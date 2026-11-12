from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path


@dataclass
class ModelConfig:
    vocab_size: int = 16384
    d_model: int = 512
    n_layer: int = 8
    n_head: int = 8
    ffn_mult: float = 8 / 3
    max_seq_len: int = 512
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    dropout: float = 0.0
    tie_embeddings: bool = True
    norm: str = "rmsnorm"           # "rmsnorm" | "layernorm" (with bias; used by imported NorMuon models)
    ffn_hidden_size: int = 0        # 0 = 8/3 * d_model rounded up to a multiple of 64

    def __post_init__(self):
        if self.d_model % self.n_head:
            raise ValueError("d_model must be divisible by n_head")
        if self.head_dim % 2:
            raise ValueError("head_dim must be even for RoPE")
        if self.norm not in ("rmsnorm", "layernorm"):
            raise ValueError(f"unknown norm {self.norm}")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_head

    @property
    def ffn_hidden(self) -> int:
        if self.ffn_hidden_size:
            return self.ffn_hidden_size
        h = int(self.ffn_mult * self.d_model)
        return 64 * ((h + 63) // 64)


@dataclass
class TrainConfig:
    stage: str = "pretrain"              # "pretrain" | "sft"
    out_dir: str = "checkpoints/run"
    train_data: str = ""                 # pretrain: .bin file, sft: .jsonl file
    val_data: str = ""
    tokenizer: str = "checkpoints/tokenizer.json"
    init_from: str = ""                  # checkpoint to start from (SFT starts from pretrain best.pt)

    batch_size: int = 32
    grad_accum: int = 1
    seq_len: int = 512
    max_steps: int = 10000

    optimizer: str = "muon"              # "muon" (Muon on hidden matrices + AdamW on rest) | "adamw"
    lr_muon: float = 0.02
    lr_adamw: float = 3e-3
    weight_decay: float = 0.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    grad_clip: float = 1.0

    warmup_steps: int = 200
    schedule: str = "wsd"                # "wsd" | "cosine" | "constant"
    decay_frac: float = 0.2              # wsd: last fraction of steps spent decaying
    min_lr_frac: float = 0.0

    eval_interval: int = 250
    eval_iters: int = 50
    save_interval: int = 1000
    log_interval: int = 10
    seed: int = 1337
    compile: bool = False
    dtype: str = "auto"                  # "auto" | "bf16" | "fp16" | "fp32"


def _from_dict(cls, d: dict):
    names = {f.name for f in fields(cls)}
    unknown = set(d) - names
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} keys: {sorted(unknown)}")
    return cls(**d)


def load_config(path: str | Path) -> tuple[ModelConfig, TrainConfig]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return _from_dict(ModelConfig, raw.get("model", {})), _from_dict(TrainConfig, raw.get("train", {}))
