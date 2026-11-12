"""Tokenize .txt files into flat uint16 train.bin / val.bin.

  python scripts/prepare_pretrain.py --input "data/raw/**/*.txt" --tokenizer checkpoints/tokenizer.json --out data/pretrain
Documents are split on <|endoftext|>; a file with none is treated as one document (e.g. a Gutenberg book).
"""

import argparse
import glob
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from minigpt.tokenizer import EOT, ChatTokenizer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", nargs="+", required=True)
    ap.add_argument("--tokenizer", default="checkpoints/tokenizer.json")
    ap.add_argument("--out", default="data/pretrain")
    ap.add_argument("--val_frac", type=float, default=0.005)
    ap.add_argument("--batch_docs", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tok = ChatTokenizer(args.tokenizer)
    if tok.vocab_size > 65535:
        sys.exit("vocab too large for uint16")
    files = sorted({f for g in args.input for f in glob.glob(g, recursive=True)})
    if not files:
        sys.exit("no input files matched")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    counts = {"train": 0, "val": 0}
    with open(out / "train.bin", "wb") as ftrain, open(out / "val.bin", "wb") as fval:
        for path in files:
            docs = [d.strip() for d in Path(path).read_text(encoding="utf-8", errors="ignore").split(EOT)]
            docs = [d for d in docs if d]
            for i in range(0, len(docs), args.batch_docs):
                for ids in tok.encode_batch(docs[i:i + args.batch_docs]):
                    split = "val" if rng.random() < args.val_frac else "train"
                    arr = np.array(ids + [tok.eot_id], dtype=np.uint16)
                    arr.tofile(fval if split == "val" else ftrain)
                    counts[split] += len(arr)
            print(f"{path}: train={counts['train'] / 1e6:.1f}M val={counts['val'] / 1e6:.2f}M tokens")

    if counts["val"] == 0:
        print("warning: val.bin is empty; raise --val_frac or add more documents")
    print(f"done -> {out} (train {counts['train']:,} tokens, val {counts['val']:,} tokens)")


if __name__ == "__main__":
    main()
