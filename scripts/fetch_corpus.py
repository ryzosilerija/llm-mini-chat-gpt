"""Stream a Hugging Face text dataset into .txt shards (docs separated by <|endoftext|>).

Example:
  python scripts/fetch_corpus.py --dataset HuggingFaceFW/fineweb-edu --config sample-10BT \
      --out data/raw/fineweb --max_chars 2_000_000_000
Gutenberg .txt files you already have can go straight into data/raw/ as well.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from minigpt.tokenizer import EOT  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--split", default="train")
    ap.add_argument("--text_field", default="text")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_chars", type=int, default=1_000_000_000)
    ap.add_argument("--shard_chars", type=int, default=100_000_000)
    args = ap.parse_args()

    from datasets import load_dataset

    ds = load_dataset(args.dataset, args.config, split=args.split, streaming=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    total, shard, shard_chars = 0, 0, 0
    f = open(out / f"shard_{shard:04d}.txt", "w", encoding="utf-8")
    for row in ds:
        text = (row.get(args.text_field) or "").replace(EOT, "").strip()
        if not text:
            continue
        f.write(text + EOT)
        total += len(text)
        shard_chars += len(text)
        if shard_chars >= args.shard_chars:
            f.close()
            shard, shard_chars = shard + 1, 0
            f = open(out / f"shard_{shard:04d}.txt", "w", encoding="utf-8")
            print(f"{total / 1e6:.0f}M chars")
        if total >= args.max_chars:
            break
    f.close()
    print(f"done: {total / 1e6:.0f}M chars in {shard + 1} shards -> {out}")


if __name__ == "__main__":
    main()
