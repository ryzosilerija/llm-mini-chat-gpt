"""Import a pretrained run from NorMuon-Swiglu-LayerNorm-RoPE-at-scale (model.pt + tokenizer.json)
into this repo's checkpoint format, so it can be SFT-tuned and served.

  python scripts/import_pretrained.py --run path/to/runs/pretrain --out checkpoints/imported
  # optional exact check against the original model code (path to that repo):
  python scripts/import_pretrained.py --run path/to/runs/pretrain --out checkpoints/imported \
      --ref_repo D:/Documents/GitHub/NorMuon-Swiglu-LayerNorm-RoPE-at-scale

Mapping: LayerNorm gain/bias -> weight/bias, fused wqkv -> wq/wk/wv, ffn -> mlp, norm_f -> norm,
head -> lm_head (untied). RoPE layouts are identical (rotate-half).
"""

import argparse
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from minigpt.config import ModelConfig  # noqa: E402
from minigpt.model import GPT  # noqa: E402
from minigpt.tokenizer import ChatTokenizer  # noqa: E402


def strip_prefixes(sd: dict) -> dict:
    out = {}
    for k, v in sd.items():
        for p in ("_orig_mod.", "module."):
            while k.startswith(p):
                k = k[len(p):]
        out[k] = v
    return out


def convert_state_dict(sd: dict, d_model: int) -> dict:
    sd = strip_prefixes(sd)
    out = {}
    for k, v in sd.items():
        if k == "tok_emb.weight":
            out[k] = v
        elif k == "head.weight":
            out["lm_head.weight"] = v
        elif k.startswith("norm_f."):
            out["norm." + ("weight" if k.endswith("gain") else "bias")] = v
        elif k.startswith("blocks."):
            _, i, rest = k.split(".", 2)
            pre = f"blocks.{i}."
            if rest == "attn.wqkv.weight":
                q, kk, vv = v.split(d_model, dim=0)
                out[pre + "attn.wq.weight"], out[pre + "attn.wk.weight"], out[pre + "attn.wv.weight"] = q, kk, vv
            elif rest == "attn.wo.weight":
                out[pre + rest] = v
            elif rest.startswith(("norm1.", "norm2.")):
                n, param = rest.split(".")
                out[pre + n + "." + ("weight" if param == "gain" else "bias")] = v
            elif rest.startswith("ffn."):
                out[pre + "mlp." + rest[len("ffn."):]] = v
            else:
                raise KeyError(f"unexpected key {k}")
        else:
            raise KeyError(f"unexpected key {k}")
    return {k: v.float().contiguous() for k, v in out.items()}


def model_config_from(src_cfg: dict) -> ModelConfig:
    d = src_cfg["d_model"]
    return ModelConfig(
        vocab_size=src_cfg["vocab_size"], d_model=d, n_layer=src_cfg["n_layer"], n_head=src_cfg["n_head"],
        max_seq_len=src_cfg["ctx"], rope_theta=src_cfg.get("rope_base", 10000.0),
        norm="layernorm", norm_eps=1e-5, ffn_hidden_size=int(round(8 * d / 3)), tie_embeddings=False,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="folder with model.pt and tokenizer.json")
    ap.add_argument("--out", default="checkpoints/imported")
    ap.add_argument("--ref_repo", default="", help="path to the original repo to verify logits match")
    args = ap.parse_args()

    run = Path(args.run)
    if (run / "model.pt").exists():
        blob = torch.load(run / "model.pt", map_location="cpu", weights_only=False)
        if "tokenizer" not in blob:
            sys.exit("this run is a character-level model; only BPE runs can be imported")
        tok_path = run / blob["tokenizer"]
    elif (run / "ckpt.pt").exists() and (run / "config.json").exists():
        # run stopped on --time-limit before writing model.pt: use the last resumable checkpoint
        import json

        ck = torch.load(run / "ckpt.pt", map_location="cpu", weights_only=False)
        blob = {"model": ck["model"], "config": json.loads((run / "config.json").read_text())["model"]}
        tok_path = run / "tokenizer.json"
        print(f"no model.pt; using ckpt.pt from step {ck.get('step')}")
    else:
        sys.exit(f"{run} has neither model.pt nor ckpt.pt + config.json")
    tok = ChatTokenizer(str(tok_path))

    cfg = model_config_from(blob["config"])
    if tok.vocab_size != cfg.vocab_size:
        sys.exit(f"tokenizer vocab {tok.vocab_size} != model vocab {cfg.vocab_size}")
    model = GPT(cfg)
    model.load_state_dict(convert_state_dict(blob["model"], cfg.d_model), strict=True)
    model.eval()
    print(f"imported {model.num_params() / 1e6:.2f}M params: {asdict(cfg)}")

    if args.ref_repo:
        verify(args.ref_repo, blob, model)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "model_config": asdict(cfg), "step": None,
                "train_config": {"stage": "pretrain"}, "source": str(run)}, out / "best.pt")
    shutil.copy(tok_path, out / "tokenizer.json")
    print(f"saved -> {out / 'best.pt'} and {out / 'tokenizer.json'}")


@torch.no_grad()
def verify(ref_repo: str, blob: dict, model: GPT) -> None:
    """Load the original model class from the source repo and compare logits on random input."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("ref_model", Path(ref_repo) / "minigpt" / "model.py")
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    ref_model = ref.GPT(ref.GPTConfig(**blob["config"]))
    ref_model.load_state_dict(strip_prefixes(blob["model"]))
    ref_model.float().eval()
    x = torch.randint(0, model.cfg.vocab_size, (2, min(64, model.cfg.max_seq_len)))
    a, _ = ref_model(x)
    b, _ = model(x)
    diff = (a.float() - b.float()).abs().max().item()
    print(f"max |logit diff| vs original model: {diff:.2e}")
    if diff > 1e-3:
        sys.exit("mismatch: conversion is not exact")


if __name__ == "__main__":
    main()
