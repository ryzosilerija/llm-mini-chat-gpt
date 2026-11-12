"""Byte-level BPE tokenizer with chat special tokens (HF `tokenizers` library)."""

from __future__ import annotations

import json
from typing import Iterable

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

EOT = "<|endoftext|>"
SYSTEM = "<|system|>"
USER = "<|user|>"
ASSISTANT = "<|assistant|>"
END = "<|end|>"
PAD = "<|pad|>"
SPECIAL_TOKENS = [EOT, SYSTEM, USER, ASSISTANT, END, PAD]


def _new_tokenizer() -> Tokenizer:
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    return tok


def _trainer(vocab_size: int):
    return trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        min_frequency=2,
        show_progress=True,
    )


def train_tokenizer(files: list[str], vocab_size: int, out_path: str) -> None:
    tok = _new_tokenizer()
    tok.train(files, _trainer(vocab_size))
    tok.save(out_path)


def train_tokenizer_from_iterator(texts: Iterable[str], vocab_size: int, out_path: str) -> None:
    tok = _new_tokenizer()
    tok.train_from_iterator(texts, _trainer(vocab_size))
    tok.save(out_path)


def _is_scratch_bpe(path: str) -> bool:
    """The from-scratch BPE (minigpt/bpe.py) saves {"pattern", "merges", "special"}."""
    with open(path, encoding="utf-8") as f:
        blob = json.load(f)
    return isinstance(blob, dict) and "pattern" in blob and "special" in blob and isinstance(blob.get("merges"), list) \
        and "model" not in blob


class ChatTokenizer:
    """One interface over two formats: HF `tokenizers` json (trained here) or the
    from-scratch BPE json used by imported NorMuon checkpoints."""

    def __init__(self, path: str):
        self.scratch = _is_scratch_bpe(path)
        if self.scratch:
            from .bpe import BPETokenizer

            self.bpe = BPETokenizer.load(path)
            lookup = self.bpe.special.get
            self._vocab_size = self.bpe.vocab_size
        else:
            self.tok = Tokenizer.from_file(path)
            lookup = self.tok.token_to_id
            self._vocab_size = self.tok.get_vocab_size()

        self.special = {}
        for s in SPECIAL_TOKENS:
            i = lookup(s)
            if i is None and s != PAD:
                raise ValueError(f"tokenizer at {path} is missing special token {s}")
            if i is not None:
                self.special[s] = i
        self.eot_id = self.special[EOT]
        self.end_id = self.special[END]
        # No pad token: reuse <|endoftext|>. Pads sit after real tokens with label -100, so it never matters.
        self.pad_id = self.special.get(PAD, self.eot_id)
        self._special_ids = set(self.special.values())

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def encode(self, text: str) -> list[int]:
        """Plain text -> ids. Special-token strings in text are never turned into special ids
        for the scratch BPE; for HF tokenizers chat_template.sanitize strips them first."""
        if self.scratch:
            return self.bpe.encode(text, allow_special=False)
        return self.tok.encode(text).ids

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        if self.scratch:
            return [self.encode(t) for t in texts]
        return [e.ids for e in self.tok.encode_batch(texts)]

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        if self.scratch:
            if skip_special:
                ids = [i for i in ids if i not in self._special_ids]
            return self.bpe.decode(ids)
        return self.tok.decode(ids, skip_special_tokens=skip_special)
