"""Pretraining and SFT in one script.

  python scripts/train.py --config configs/pretrain_30m.json
  python scripts/train.py --config configs/sft_30m.json
  python scripts/train.py --config configs/pretrain_30m.json --resume          # continue from last.pt
  python scripts/train.py --config configs/tiny_debug.json --set max_steps=50  # override train keys

Writes to out_dir: last.pt (full, resumable), best.pt (weights only, for serving),
tokenizer.json (copy), metrics.jsonl.
"""

import argparse
import json
import os
import random
import shutil
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from minigpt.config import ModelConfig, load_config  # noqa: E402
from minigpt.data import PretrainData, SFTData  # noqa: E402
from minigpt.model import GPT  # noqa: E402
from minigpt.optim import build_optimizers, lr_multiplier, set_lr  # noqa: E402
from minigpt.tokenizer import ChatTokenizer  # noqa: E402


def parse_overrides(tcfg, items):
    for item in items:
        key, _, val = item.partition("=")
        if not hasattr(tcfg, key):
            sys.exit(f"unknown train key: {key}")
        try:
            val = json.loads(val)
        except json.JSONDecodeError:
            pass
        setattr(tcfg, key, val)


def pick_dtype(name, device):
    if device != "cuda" or name == "fp32":
        return None
    if name == "bf16" or (name == "auto" and torch.cuda.is_bf16_supported()):
        return torch.bfloat16
    return torch.float16  # T4 / P100 on Kaggle


def atomic_save(obj, path):
    tmp = str(path) + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


@torch.no_grad()
def evaluate(model, data, iters, ctx):
    model.eval()
    losses = []
    for _ in range(iters):
        x, y = data.batch()
        with ctx:
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--set", nargs="*", default=[], help="train overrides key=value")
    args = ap.parse_args()

    mcfg, tcfg = load_config(args.config)
    parse_overrides(tcfg, args.set)

    random.seed(tcfg.seed)
    np.random.seed(tcfg.seed)
    torch.manual_seed(tcfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp_dtype = pick_dtype(tcfg.dtype, device)
    ctx = torch.autocast(device_type="cuda", dtype=amp_dtype) if amp_dtype else nullcontext()
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    out = Path(tcfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tok = ChatTokenizer(tcfg.tokenizer)
    shutil.copy(tcfg.tokenizer, out / "tokenizer.json")

    init_state = None
    if tcfg.init_from:
        ck = torch.load(tcfg.init_from, map_location="cpu", weights_only=False)
        mcfg = ModelConfig(**ck["model_config"])  # architecture comes from the checkpoint
        init_state = ck["model"]
        print(f"init from {tcfg.init_from} (step {ck.get('step')})")
    elif mcfg.vocab_size != tok.vocab_size:
        print(f"setting vocab_size {mcfg.vocab_size} -> {tok.vocab_size} (from tokenizer)")
        mcfg.vocab_size = tok.vocab_size
    if tcfg.seq_len > mcfg.max_seq_len:
        sys.exit(f"seq_len {tcfg.seq_len} > max_seq_len {mcfg.max_seq_len}")

    model = GPT(mcfg)
    if init_state is not None:
        model.load_state_dict(init_state)
    model.to(device)
    print(f"model: {model.num_params() / 1e6:.2f}M params "
          f"({model.num_params(exclude_embeddings=True) / 1e6:.2f}M non-embedding), device={device}, amp={amp_dtype}")

    if tcfg.stage == "pretrain":
        train_data = PretrainData(tcfg.train_data, tcfg.seq_len, tcfg.batch_size, device)
        val_data = PretrainData(tcfg.val_data, tcfg.seq_len, tcfg.batch_size, device) if tcfg.val_data else None
    elif tcfg.stage == "sft":
        train_data = SFTData(tcfg.train_data, tok, tcfg.seq_len, tcfg.batch_size, device)
        val_data = SFTData(tcfg.val_data, tok, tcfg.seq_len, tcfg.batch_size, device) if tcfg.val_data else None
        print(f"sft: {len(train_data)} train conversations")
    else:
        sys.exit(f"unknown stage {tcfg.stage}")

    opts = build_optimizers(model, tcfg)
    step, best_val = 0, float("inf")
    if args.resume and (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        for opt, sd in zip(opts, ck["opts"]):
            opt.load_state_dict(sd)
        step, best_val = ck["step"], ck.get("best_val", best_val)
        print(f"resumed from step {step}")

    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)
    train_model = torch.compile(model) if tcfg.compile else model
    meta = {"model_config": asdict(mcfg), "train_config": asdict(tcfg)}

    def save():
        atomic_save({"model": model.state_dict(), "opts": [o.state_dict() for o in opts],
                     "step": step, "best_val": best_val, **meta}, out / "last.pt")

    def maybe_eval():
        nonlocal best_val
        if val_data is None:
            atomic_save({"model": model.state_dict(), "step": step, **meta}, out / "best.pt")
            return None
        val = evaluate(model, val_data, tcfg.eval_iters, ctx)
        if val < best_val:
            best_val = val
            atomic_save({"model": model.state_dict(), "step": step, "val_loss": val, **meta}, out / "best.pt")
        return val

    log = open(out / "metrics.jsonl", "a", encoding="utf-8")
    tokens_per_step = tcfg.batch_size * tcfg.grad_accum * tcfg.seq_len
    model.train()
    t0 = time.time()
    while step < tcfg.max_steps:
        mult = lr_multiplier(step, tcfg)
        set_lr(opts, mult)
        loss_acc = 0.0
        for _ in range(tcfg.grad_accum):
            x, y = train_data.batch()
            with ctx:
                _, loss = train_model(x, y)
            scaler.scale(loss / tcfg.grad_accum).backward()
            loss_acc += loss.item() / tcfg.grad_accum
        for o in opts:
            scaler.unscale_(o)
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip if tcfg.grad_clip > 0 else 1e9)
        for o in opts:
            scaler.step(o)
        scaler.update()
        for o in opts:
            o.zero_grad(set_to_none=True)
        step += 1

        rec = None
        if step % tcfg.log_interval == 0 or step == 1:
            dt = time.time() - t0
            t0 = time.time()
            n = 1 if step == 1 else tcfg.log_interval
            rec = {"step": step, "loss": round(loss_acc, 4), "lr_mult": round(mult, 4),
                   "grad_norm": round(float(gnorm), 3), "tok_per_s": int(tokens_per_step * n / max(dt, 1e-9))}
            print(" ".join(f"{k}={v}" for k, v in rec.items()), flush=True)
        if step % tcfg.eval_interval == 0 or step == tcfg.max_steps:
            val = maybe_eval()
            rec = {**(rec or {"step": step}), "val_loss": None if val is None else round(val, 4)}
            if val is not None:
                print(f"step={step} val_loss={val:.4f} best={best_val:.4f}", flush=True)
        if rec:
            log.write(json.dumps(rec) + "\n")
            log.flush()
        if step % tcfg.save_interval == 0 or step == tcfg.max_steps:
            save()

    log.close()
    print(f"done. best val {best_val:.4f}. serve with: MINIGPT_CKPT={out / 'best.pt'} "
          f"MINIGPT_TOKENIZER={out / 'tokenizer.json'}")


if __name__ == "__main__":
    main()
