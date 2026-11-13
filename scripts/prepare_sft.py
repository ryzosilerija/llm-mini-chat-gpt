"""Build SFT train/val jsonl from Hugging Face chat datasets and/or local jsonl files.

  python scripts/prepare_sft.py \
      --hf "HuggingFaceTB/smoltalk:everyday-conversations*4" "allenai/soda@40000" \
      --jsonl data/npc_sft.jsonl --out data/sft

Spec syntax: name[:subset][@limit][*repeat]. Supports chat-format rows ("messages") and
two-person dialogue rows ("dialogue" + "speakers", e.g. allenai/soda).

Each line in the output: {"messages": [{"role": "system|user|assistant", "content": "..."}]}
"""

import argparse
import json
import random
import sys
from pathlib import Path

ROLE_MAP = {"system": "system", "user": "user", "human": "user", "assistant": "assistant", "gpt": "assistant"}


def normalize(messages):
    out = []
    for m in messages:
        role = ROLE_MAP.get(m.get("role") or m.get("from"))
        content = (m.get("content") or m.get("value") or "").strip()
        if role and content:
            out.append({"role": role, "content": content})
    if not any(m["role"] == "assistant" for m in out):
        return None
    return out


def from_dialogue(row):
    """SODA-style row: {"dialogue": [utt, ...], "speakers": [name, ...]} between two people.
    The first speaker is the user; the second is the character the model plays."""
    utts, speakers = row.get("dialogue") or [], row.get("speakers") or []
    if len(utts) < 2 or len(utts) != len(speakers):
        return None
    names = list(dict.fromkeys(speakers))
    if len(names) != 2:
        return None
    user_name, bot_name = names
    # The scene description teaches the model to actually USE the system prompt (setting, mood, situation);
    # a fixed one-liner taught it to ignore system prompts entirely.
    scene = (row.get("narrative") or "").strip()
    system = f"You are {bot_name}, talking with {user_name}."
    if scene:
        system += f" Scene: {scene}"
    out = [{"role": "system", "content": system}]
    for name, text in zip(speakers, utts):
        text = (text or "").strip()
        if not text:
            return None
        role = "user" if name == user_name else "assistant"
        if out[-1]["role"] == role:          # merge consecutive turns by the same speaker
            out[-1]["content"] += " " + text
        else:
            out.append({"role": role, "content": text})
    return out if any(m["role"] == "assistant" for m in out) else None


def load_hf(spec, limit):
    """spec: name[:subset][@limit][*repeat]  e.g. HuggingFaceTB/smoltalk:systemchats-30k@8000"""
    from datasets import load_dataset

    spec, _, rep = spec.partition("*")
    spec, _, lim = spec.partition("@")
    limit = int(lim) if lim else limit
    name, _, subset = spec.partition(":")
    ds = load_dataset(name, subset or None, split="train", streaming=True)
    rows = []
    for row in ds:
        if row.get("dialogue"):
            msgs = from_dialogue(row)
        else:
            msgs = normalize(row.get("messages") or row.get("conversations") or [])
        if msgs:
            rows.append(msgs)
        if limit and len(rows) >= limit:
            break
    repeat = int(rep) if rep else 1
    print(f"{spec}: {len(rows)} conversations x{repeat}")
    return rows * repeat


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                msgs = normalize(json.loads(line)["messages"])
                if msgs:
                    rows.append(msgs)
    print(f"{path}: {len(rows)} conversations")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", nargs="*", default=[], help="dataset[:subset][@limit][*repeat] specs")
    ap.add_argument("--hf_limit", type=int, default=50000, help="max conversations per HF spec")
    ap.add_argument("--jsonl", nargs="*", default=[], help="local jsonl files (e.g. NPC data)")
    ap.add_argument("--npc_repeat", type=int, default=1, help="upsample local jsonl files")
    ap.add_argument("--val_frac", type=float, default=0.02)
    ap.add_argument("--out", default="data/sft")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = []
    for spec in args.hf:
        rows += load_hf(spec, args.hf_limit)
    for path in args.jsonl:
        rows += load_jsonl(path) * args.npc_repeat
    if not rows:
        sys.exit("no data")

    random.Random(args.seed).shuffle(rows)
    n_val = max(1, int(len(rows) * args.val_frac))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, part in (("val", rows[:n_val]), ("train", rows[n_val:])):
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for msgs in part:
                f.write(json.dumps({"messages": msgs}, ensure_ascii=False) + "\n")
    print(f"done -> {out}: train {len(rows) - n_val}, val {n_val}")


if __name__ == "__main__":
    main()
