"""Train the byte-level BPE tokenizer.

  python scripts/train_tokenizer.py --input "data/raw/**/*.txt" --vocab_size 16384 --out checkpoints/tokenizer.json
"""

import argparse
import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from minigpt.tokenizer import train_tokenizer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", nargs="+", required=True, help="glob(s) of .txt files")
    ap.add_argument("--vocab_size", type=int, default=16384)
    ap.add_argument("--max_files", type=int, default=20, help="tokenizer only needs a sample")
    ap.add_argument("--out", default="checkpoints/tokenizer.json")
    args = ap.parse_args()

    files = sorted({f for g in args.input for f in glob.glob(g, recursive=True)})[: args.max_files]
    if not files:
        sys.exit("no input files matched")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    train_tokenizer(files, args.vocab_size, args.out)
    print(f"tokenizer ({args.vocab_size} vocab) trained on {len(files)} files -> {args.out}")


if __name__ == "__main__":
    main()
