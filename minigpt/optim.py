"""Muon (hidden matrices) + AdamW (embeddings, norms) and LR schedules."""

from __future__ import annotations

import math

import torch

from .config import TrainConfig


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    # fp32 on purpose: Kaggle T4/P100 have no bf16, and these models are small.
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.float()
    transposed = X.size(0) > X.size(1)
    if transposed:
        X = X.T
    X = X / (X.norm() + eps)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    return X.T if transposed else X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov,
                                      ns_steps=ns_steps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            lr, mom, wd = group["lr"], group["momentum"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(mom).add_(g)
                g = g.add(buf, alpha=mom) if group["nesterov"] else buf
                u = zeropower_via_newtonschulz5(g, group["ns_steps"])
                u *= max(1.0, p.size(0) / p.size(1)) ** 0.5
                if wd:
                    p.mul_(1 - lr * wd)
                p.add_(u.to(p.dtype), alpha=-lr)


def build_optimizers(model: torch.nn.Module, cfg: TrainConfig) -> list[torch.optim.Optimizer]:
    matrix, embed, other = [], [], []
    for name, p in model.named_parameters():  # tied lm_head is deduped into tok_emb
        if not p.requires_grad:
            continue
        if name.startswith(("tok_emb", "lm_head")):
            embed.append(p)
        elif p.ndim == 2:
            matrix.append(p)
        else:
            other.append(p)

    betas = (cfg.adam_beta1, cfg.adam_beta2)
    adam_groups = [
        {"params": embed, "lr": cfg.lr_adamw, "weight_decay": cfg.weight_decay},
        {"params": other, "lr": cfg.lr_adamw, "weight_decay": 0.0},
    ]
    if cfg.optimizer == "muon":
        opts = [Muon(matrix, lr=cfg.lr_muon, weight_decay=cfg.weight_decay),
                torch.optim.AdamW(adam_groups, betas=betas, eps=1e-8)]
    elif cfg.optimizer == "adamw":
        adam_groups.append({"params": matrix, "lr": cfg.lr_adamw, "weight_decay": cfg.weight_decay})
        opts = [torch.optim.AdamW(adam_groups, betas=betas, eps=1e-8)]
    else:
        raise ValueError(f"unknown optimizer {cfg.optimizer}")

    for opt in opts:
        for g in opt.param_groups:
            g["base_lr"] = g["lr"]
    return opts


def lr_multiplier(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup_steps:
        return (step + 1) / cfg.warmup_steps
    if cfg.schedule == "constant":
        return 1.0
    if cfg.schedule == "cosine":
        progress = min(1.0, (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps))
        return cfg.min_lr_frac + (1 - cfg.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * progress))
    if cfg.schedule == "wsd":
        decay_start = int(cfg.max_steps * (1 - cfg.decay_frac))
        if step < decay_start:
            return 1.0
        progress = min(1.0, (step - decay_start) / max(1, cfg.max_steps - decay_start))
        return cfg.min_lr_frac + (1 - cfg.min_lr_frac) * (1 - progress)
    raise ValueError(f"unknown schedule {cfg.schedule}")


def set_lr(opts: list[torch.optim.Optimizer], mult: float) -> None:
    for opt in opts:
        for g in opt.param_groups:
            g["lr"] = g["base_lr"] * mult
