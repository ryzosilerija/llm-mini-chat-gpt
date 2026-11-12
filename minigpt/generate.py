"""Sampling with KV cache, UTF-8-safe text streaming, and option scoring for NPC actions."""

from __future__ import annotations

from typing import Iterable, Iterator

import torch
import torch.nn.functional as F

from .model import GPT
from .tokenizer import ChatTokenizer


def sample_next(logits: torch.Tensor, seen: list[int], temperature=0.8, top_k=50, top_p=0.95,
                repetition_penalty=1.1) -> int:
    logits = logits.float().clone()
    if repetition_penalty != 1.0 and seen:
        idx = torch.tensor(sorted(set(seen)), device=logits.device)
        vals = logits[idx]
        logits[idx] = torch.where(vals > 0, vals / repetition_penalty, vals * repetition_penalty)
    if temperature <= 0:
        return int(logits.argmax())
    logits = logits / temperature
    if top_k and top_k < logits.numel():
        kth = torch.topk(logits, top_k).values[-1]
        logits[logits < kth] = float("-inf")
    probs = F.softmax(logits, dim=-1)
    if 0 < top_p < 1.0:
        sp, si = torch.sort(probs, descending=True)
        cum = sp.cumsum(0)
        sp[(cum - sp) > top_p] = 0.0  # keep the smallest prefix whose mass reaches top_p
        probs = torch.zeros_like(probs).scatter_(0, si, sp)
        probs = probs / probs.sum()
    return int(torch.multinomial(probs, 1))


@torch.no_grad()
def generate_ids(model: GPT, prompt_ids: list[int], max_new_tokens=150, temperature=0.8, top_k=50,
                 top_p=0.95, repetition_penalty=1.1, stop_ids: Iterable[int] = ()) -> Iterator[int]:
    model.eval()
    device = next(model.parameters()).device
    max_len = model.cfg.max_seq_len
    max_new_tokens = max(1, min(max_new_tokens, max_len - 1))
    prompt_ids = prompt_ids[-(max_len - max_new_tokens):]  # hard safety truncation
    stop = set(stop_ids)

    caches = model.new_caches()
    logits, _ = model(torch.tensor([prompt_ids], device=device), caches=caches, last_only=True)
    pos = len(prompt_ids)
    history: list[int] = []  # penalize only the model's own output, never the user's words
    for _ in range(max_new_tokens):
        nxt = sample_next(logits[0, -1], history, temperature, top_k, top_p, repetition_penalty)
        if nxt in stop:
            return
        history.append(nxt)
        yield nxt
        if pos >= max_len:
            return
        logits, _ = model(torch.tensor([[nxt]], device=device), caches=caches, start_pos=pos, last_only=True)
        pos += 1


def stream_text(tok: ChatTokenizer, ids: Iterable[int]) -> Iterator[str]:
    """Yield text deltas; holds back partial UTF-8 sequences (byte-level BPE splits them)."""
    acc: list[int] = []
    emitted = ""
    for i in ids:
        acc.append(i)
        text = tok.decode(acc)
        if text.endswith("�"):
            continue
        delta, emitted = text[len(emitted):], text
        if delta:
            yield delta


@torch.no_grad()
def score_options(model: GPT, tok: ChatTokenizer, prompt_ids: list[int], options: list[str]) -> list[float]:
    """Mean log-prob of each option (+ <|end|>) as the assistant reply. Always returns a valid choice,
    which is far more reliable for a small model than asking it to emit JSON."""
    model.eval()
    device = next(model.parameters()).device
    opt_ids = [tok.encode(o) + [tok.end_id] for o in options]
    room = model.cfg.max_seq_len - max(len(o) for o in opt_ids)
    if room < 1:
        raise ValueError("options too long for context window")
    prompt_ids = prompt_ids[-room:]
    seqs = [prompt_ids + o for o in opt_ids]
    L = max(len(s) for s in seqs)
    x = torch.full((len(seqs), L), tok.pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        x[i, :len(s)] = torch.tensor(s)
    logits, _ = model(x.to(device))
    logp = F.log_softmax(logits.float(), dim=-1)
    p = len(prompt_ids)
    scores = []
    for i, s in enumerate(seqs):
        targets = torch.tensor(s[p:], device=device)
        lp = logp[i, p - 1:len(s) - 1].gather(1, targets[:, None]).squeeze(1)
        scores.append(float(lp.mean()))
    return scores
