from __future__ import annotations

import json
import random

import numpy as np
import torch

from .chat_template import encode_chat
from .tokenizer import ChatTokenizer


def _to_device(x, y, device):
    if str(device).startswith("cuda"):
        return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    return x.to(device), y.to(device)


class PretrainData:
    """Random windows from a flat uint16 token file (docs separated by <|endoftext|>)."""

    def __init__(self, path: str, seq_len: int, batch_size: int, device):
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        if len(self.data) <= seq_len + 1:
            raise ValueError(f"{path} has only {len(self.data)} tokens, need > {seq_len + 1}")
        self.seq_len, self.batch_size, self.device = seq_len, batch_size, device

    def batch(self):
        ix = np.random.randint(0, len(self.data) - self.seq_len - 1, size=self.batch_size)
        buf = np.stack([self.data[i:i + self.seq_len + 1] for i in ix]).astype(np.int64)
        t = torch.from_numpy(buf)
        return _to_device(t[:, :-1], t[:, 1:], self.device)


class SFTData:
    """Chat examples from jsonl lines like {"messages": [{"role": ..., "content": ...}, ...]}."""

    def __init__(self, path: str, tok: ChatTokenizer, seq_len: int, batch_size: int, device):
        self.examples: list[tuple[list[int], list[int]]] = []
        skipped = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ids, labels = encode_chat(tok, json.loads(line)["messages"])
                ids, labels = ids[:seq_len + 1], labels[:seq_len + 1]
                if not any(l != -100 for l in labels[1:]):
                    skipped += 1
                    continue
                self.examples.append((ids, labels))
        if not self.examples:
            raise ValueError(f"no usable SFT examples in {path}")
        if skipped:
            print(f"[sft] skipped {skipped} examples with no assistant tokens after truncation")
        self.pad_id, self.batch_size, self.device = tok.pad_id, batch_size, device

    def __len__(self):
        return len(self.examples)

    def batch(self):
        chosen = random.choices(self.examples, k=self.batch_size)
        L = max(len(ids) for ids, _ in chosen) - 1
        x = torch.full((self.batch_size, L), self.pad_id, dtype=torch.long)
        y = torch.full((self.batch_size, L), -100, dtype=torch.long)
        for i, (ids, labels) in enumerate(chosen):
            n = len(ids) - 1
            x[i, :n] = torch.tensor(ids[:-1])
            y[i, :n] = torch.tensor(labels[1:])
        # Right padding + causal attention: real tokens never attend to pads.
        return _to_device(x, y, self.device)
