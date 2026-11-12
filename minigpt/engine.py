"""Inference engine: loads a checkpoint + tokenizer, fits chat history to context, thread-safe."""

from __future__ import annotations

import threading
from typing import Iterator

import torch

from .chat_template import encode_chat
from .config import ModelConfig
from .generate import generate_ids, score_options, stream_text
from .model import GPT
from .tokenizer import ChatTokenizer


def resolve_device(device: str = "auto") -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


class Engine:
    def __init__(self, ckpt_path: str, tokenizer_path: str, device: str = "auto"):
        self.device = resolve_device(device)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.cfg = ModelConfig(**ckpt["model_config"])
        self.model = GPT(self.cfg)
        self.model.load_state_dict(ckpt["model"])
        self.model.to(self.device).eval()
        self.tok = ChatTokenizer(tokenizer_path)
        if self.tok.vocab_size != self.cfg.vocab_size:
            raise ValueError(f"tokenizer vocab {self.tok.vocab_size} != model vocab {self.cfg.vocab_size}")
        self.lock = threading.Lock()  # one forward pass at a time; small model, keep it simple
        self.info = {"params": self.model.num_params(), "max_seq_len": self.cfg.max_seq_len,
                     "device": self.device, "step": ckpt.get("step"), "stage": ckpt.get("train_config", {}).get("stage")}

    def fit_messages(self, messages: list[dict], reserve: int) -> list[int]:
        """Encode with generation prompt; drop oldest non-system turns until it fits."""
        budget = max(1, self.cfg.max_seq_len - reserve)
        system = [m for m in messages if m["role"] == "system"][:1]
        rest = [m for m in messages if m["role"] != "system"]
        while True:
            ids, _ = encode_chat(self.tok, system + rest, add_generation_prompt=True)
            if len(ids) <= budget or len(rest) <= 1:
                break
            rest = rest[1:]
        return ids[-budget:]

    def stream_chat(self, messages: list[dict], max_tokens=150, temperature=0.6, top_k=40,
                    top_p=0.9, repetition_penalty=1.1) -> Iterator[str]:
        max_tokens = max(1, min(max_tokens, self.cfg.max_seq_len // 2))
        ids = self.fit_messages(messages, reserve=max_tokens)
        with self.lock:
            gen = generate_ids(self.model, ids, max_tokens, temperature, top_k, top_p,
                               repetition_penalty, stop_ids=(self.tok.end_id, self.tok.eot_id))
            yield from stream_text(self.tok, gen)

    def chat(self, messages: list[dict], **kw) -> str:
        return "".join(self.stream_chat(messages, **kw)).strip()

    def choose(self, messages: list[dict], options: list[str]) -> list[tuple[str, float]]:
        longest = max(len(self.tok.encode(o)) for o in options) + 1
        ids = self.fit_messages(messages, reserve=longest)
        with self.lock:
            scores = score_options(self.model, self.tok, ids, options)
        return sorted(zip(options, scores), key=lambda t: t[1], reverse=True)
