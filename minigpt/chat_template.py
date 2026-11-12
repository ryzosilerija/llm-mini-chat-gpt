"""Chat format:  <|system|>...<|end|><|user|>...<|end|><|assistant|>...<|end|>

Labels: only assistant content + its <|end|> are trained (-100 everywhere else).
Training uses x = ids[:-1], y = labels[1:].
"""

from __future__ import annotations

from .tokenizer import ASSISTANT, SPECIAL_TOKENS, SYSTEM, USER, ChatTokenizer

ROLE_TOKENS = {"system": SYSTEM, "user": USER, "assistant": ASSISTANT}


def sanitize(text: str) -> str:
    """Strip special-token strings so a player can't type '<|assistant|>' to hijack the format."""
    for s in SPECIAL_TOKENS:
        text = text.replace(s, "")
    return text


def encode_chat(
    tok: ChatTokenizer, messages: list[dict], add_generation_prompt: bool = False
) -> tuple[list[int], list[int]]:
    ids: list[int] = []
    labels: list[int] = []
    for m in messages:
        role = m["role"]
        if role not in ROLE_TOKENS:
            raise ValueError(f"unknown role: {role}")
        content = tok.encode(sanitize(m["content"]))
        seg = [tok.special[ROLE_TOKENS[role]], *content, tok.end_id]
        ids += seg
        if role == "assistant":
            labels += [-100, *content, tok.end_id]
        else:
            labels += [-100] * len(seg)
    if add_generation_prompt:
        ids.append(tok.special[ASSISTANT])
        labels.append(-100)
    return ids, labels
