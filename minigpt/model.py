"""Decoder-only transformer: RMSNorm, RoPE, SwiGLU, tied embeddings, KV cache."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return xf.type_as(x) * self.weight


def make_norm(cfg: ModelConfig) -> nn.Module:
    if cfg.norm == "layernorm":
        return nn.LayerNorm(cfg.d_model, eps=cfg.norm_eps)
    return RMSNorm(cfg.d_model, cfg.norm_eps)


def build_rope(max_len: int, head_dim: int, theta: float):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    freqs = torch.outer(torch.arange(max_len).float(), inv_freq)  # (T, D/2)
    return freqs.cos(), freqs.sin()


def apply_rope(x, cos, sin):
    # x: (B, H, T, D), cos/sin: (T, D/2). Half-split rotation.
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    cos, sin = cos.to(x.dtype), sin.to(x.dtype)
    return torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)


class KVCache:
    __slots__ = ("k", "v")

    def __init__(self):
        self.k = None
        self.v = None

    @property
    def length(self) -> int:
        return 0 if self.k is None else self.k.shape[2]


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d = cfg.d_model
        self.n_head, self.head_dim = cfg.n_head, cfg.head_dim
        # Separate Q/K/V/O so optimizer groups (attn_qk vs attn_vo) stay clean.
        self.wq = nn.Linear(d, d, bias=False)
        self.wk = nn.Linear(d, d, bias=False)
        self.wv = nn.Linear(d, d, bias=False)
        self.wo = nn.Linear(d, d, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x, cos, sin, cache: KVCache | None = None):
        B, T, C = x.shape
        q = self.wq(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)

        past = 0
        if cache is not None:
            past = cache.length
            if past:
                k = torch.cat([cache.k, k], dim=2)
                v = torch.cat([cache.v, v], dim=2)
            cache.k, cache.v = k, v

        if past == 0:
            y = F.scaled_dot_product_attention(
                q, k, v, is_causal=T > 1, dropout_p=self.dropout if self.training else 0.0
            )
        else:
            # Chunk appended after a cached prefix: token i may see past + i + 1 keys.
            mask = torch.ones(T, past + T, dtype=torch.bool, device=x.device).tril(diagonal=past)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

        return self.wo(y.transpose(1, 2).contiguous().view(B, T, C))


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        h = cfg.ffn_hidden
        self.w_gate = nn.Linear(cfg.d_model, h, bias=False)
        self.w_up = nn.Linear(cfg.d_model, h, bias=False)
        self.w_down = nn.Linear(h, cfg.d_model, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1 = make_norm(cfg)
        self.attn = Attention(cfg)
        self.norm2 = make_norm(cfg)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, cos, sin, cache=None):
        x = x + self.attn(self.norm1(x), cos, sin, cache)
        return x + self.mlp(self.norm2(x))


class GPT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm = make_norm(cfg)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight

        cos, sin = build_rope(cfg.max_seq_len, cfg.head_dim, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        for name, p in self.named_parameters():
            if name.endswith("wo.weight") or name.endswith("w_down.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, caches=None, start_pos: int = 0, last_only: bool = False):
        B, T = idx.shape
        if start_pos + T > self.cfg.max_seq_len:
            raise ValueError(f"sequence {start_pos + T} exceeds max_seq_len {self.cfg.max_seq_len}")
        x = self.tok_emb(idx)
        cos = self.rope_cos[start_pos:start_pos + T]
        sin = self.rope_sin[start_pos:start_pos + T]
        for i, block in enumerate(self.blocks):
            x = block(x, cos, sin, None if caches is None else caches[i])
        x = self.norm(x)

        if targets is None:
            if last_only:
                x = x[:, -1:, :]
            return self.lm_head(x), None

        logits = self.lm_head(x)
        loss = F.cross_entropy(
            logits.float().view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100
        )
        return logits, loss

    def new_caches(self) -> list[KVCache]:
        return [KVCache() for _ in self.blocks]

    def num_params(self, exclude_embeddings: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())  # parameters() dedups tied weights
        if exclude_embeddings:
            n -= self.tok_emb.weight.numel()
        return n
