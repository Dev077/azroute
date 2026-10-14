"""Replay buffer, symmetry augmentation and the training step.

Loss (per batch):
    KL(pi' || p)                         decision distribution from the search
  + (y_std - y_hat)^2                     standardised log remaining-excess, completed reroutes only
  + 0.5 * BCE(blocked)
  + 0.5 * BCE(future occupancy)           per region cell and layer, completed reroutes only
  + 0.25 * (net excess)^2                 per net, completed reroutes only
  + AdamW weight decay
"""
from __future__ import annotations

import math
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from .features import BIT_FUTURE, CROP, GLOBAL_FEATS, MAX_NETS, N_MODES, NODE_FEATS, PAIR_FEATS
from .model import RouteNet

FIELDS = {
    "maps": ((None, CROP, CROP), np.uint8),     # (L, 64, 64); filled with L at construction
    "nodes": ((MAX_NETS, NODE_FEATS), np.float16),
    "rois": ((MAX_NETS, 4), np.int16),
    "glob": ((GLOBAL_FEATS,), np.float32),
    "legal": ((MAX_NETS, N_MODES), np.bool_),
    "ld": ((None,), np.float32),                # (L + 1,)
    "pairs": ((MAX_NETS, MAX_NETS, PAIR_FEATS), np.float16),
    "pi": ((MAX_NETS, N_MODES), np.float16),
    "y": ((), np.float32),
    "blocked": ((), np.uint8),
    "fut_valid": ((), np.uint8),
    "excess": ((MAX_NETS,), np.float16),
    "spread": ((), np.float32),                 # max - min Q over the searched decisions
}


class ReplayBuffer:
    """Fixed-capacity ring buffer of training records (numpy, host memory)."""

    def __init__(self, capacity: int, layers: int):
        self.cap, self.n, self.pos = capacity, 0, 0
        self.data = {}
        for k, (shape, dt) in FIELDS.items():
            shape = tuple(layers if s is None and k == "maps" else (layers + 1 if s is None else s) for s in shape)
            self.data[k] = np.zeros((capacity,) + shape, dtype=dt)

    def add(self, batch: Dict[str, np.ndarray]) -> int:
        m = len(batch["y"])
        for i in range(m):
            for k in FIELDS:
                if k in batch:          # shards written before a field existed leave it at 0
                    self.data[k][self.pos] = batch[k][i]
                else:
                    self.data[k][self.pos] = 0
            self.pos = (self.pos + 1) % self.cap
            self.n = min(self.n + 1, self.cap)
        return m

    def sample(self, b: int, rng: np.random.Generator, prio: float = 0.0) -> Dict[str, np.ndarray]:
        """Uniform, except a `prio` share of the batch drawn in proportion to the record's spread
        (records where the searched decisions led to different outcomes)."""
        n_p = int(round(b * prio))
        idx = rng.integers(0, self.n, size=b - n_p)
        if n_p:
            w = self.data["spread"][: self.n].astype(np.float64) + 1e-3
            idx = np.concatenate([idx, rng.choice(self.n, size=n_p, p=w / w.sum())])
        return {k: v[idx] for k, v in self.data.items() if k != "spread"}


def to_device(batch: Dict[str, np.ndarray], device) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in batch.items():
        t = torch.from_numpy(np.ascontiguousarray(v))
        if t.dtype == torch.float16:
            t = t.float()
        if k == "rois":
            t = t.long()
        out[k] = t.to(device, non_blocking=True)
    return out


def augment(b: Dict[str, torch.Tensor], sym: int) -> Dict[str, torch.Tensor]:
    """Apply one of 16 symmetries: bit0 flip x, bit1 flip y, bit2 swap x/y, bit3 flip layers + swap dies.
    Delays, mode meanings and pair conflict fractions are invariant, so those targets and inputs
    are unchanged."""
    maps, nodes, rois, ld = b["maps"], b["nodes"].clone(), b["rois"].clone(), b["ld"].clone()
    valid = nodes[..., 23:24]
    last = CROP - 1
    if sym & 1:
        maps = maps.flip(-1)
        rois[..., 0], rois[..., 2] = last - b["rois"][..., 2], last - b["rois"][..., 0]
        nodes[..., 9] = (last / CROP - nodes[..., 9]) * valid[..., 0]
    if sym & 2:
        maps = maps.flip(-2)
        r = rois.clone()
        rois[..., 1], rois[..., 3] = last - r[..., 3], last - r[..., 1]
        nodes[..., 10] = (last / CROP - nodes[..., 10]) * valid[..., 0]
    if sym & 4:
        maps = maps.transpose(-1, -2)
        rois = rois[..., [1, 0, 3, 2]]
        nodes[..., [7, 8]] = nodes[..., [8, 7]]
        nodes[..., [9, 10]] = nodes[..., [10, 9]]
    if sym & 8:
        maps = maps.flip(1)
        L = maps.shape[1]
        ld[:, :L] = ld[:, :L].flip(-1)
        nodes[..., 5] = (1 - nodes[..., 5]) * valid[..., 0]
        nodes[..., 6] = (1 - nodes[..., 6]) * valid[..., 0]
    out = dict(b)
    out.update(maps=maps.contiguous(), nodes=nodes, rois=rois, ld=ld)
    return out


class Trainer:
    def __init__(self, model: RouteNet, device, lr: float = 3e-4, weight_decay: float = 1e-4,
                 warmup: int = 1000, norm_momentum: float = 0.999):
        self.model, self.device = model, device
        self.opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.base_lr, self.warmup, self.mom = lr, warmup, norm_momentum
        self.steps = 0
        self.norm_init = False
        self.use_amp = device.type == "cuda"

    def lr_at(self, progress: float) -> float:
        warm = min(1.0, (self.steps + 1) / self.warmup)
        decay = 1.0 if progress < 0.7 else (0.3 if progress < 0.9 else 0.1)
        return self.base_lr * warm * decay

    def _update_norm(self, y: torch.Tensor, ok: torch.Tensor) -> None:
        if ok.sum() < 2:
            return
        v = y[ok]
        mu, sd = v.mean(), v.std().clamp(min=0.05)
        m = self.model
        if not self.norm_init:
            m.y_mu.copy_(mu); m.y_sigma.copy_(sd); self.norm_init = True
        else:
            m.y_mu.mul_(self.mom).add_((1 - self.mom) * mu)
            m.y_sigma.mul_(self.mom).add_((1 - self.mom) * sd)

    def step(self, batch_np: Dict[str, np.ndarray], progress: float, sym: int) -> Dict[str, float]:
        m = self.model
        m.train()
        b = augment(to_device(batch_np, self.device), sym)
        legal = b["legal"]
        ok = (~torch.isnan(b["y"])) & (b["blocked"] == 0)
        self._update_norm(b["y"], ok)
        for g in self.opt.param_groups:
            g["lr"] = self.lr_at(progress)
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.use_amp):
            out = m(b["maps"], b["nodes"], b["rois"], b["glob"], legal, b["ld"], b["pairs"])
        logits = out["logits"].float()
        pi = b["pi"].flatten(1)
        lmask = legal.flatten(1)
        logp = torch.log_softmax(logits, -1).masked_fill(~lmask, 0.0)
        ent = -(pi * torch.log(pi.clamp(min=1e-12))).sum(-1)
        ce = -(pi * logp).sum(-1)
        loss_pi = ce.mean()
        y_std = ((b["y"] - m.y_mu) / m.y_sigma).nan_to_num(0.0)
        loss_v = ((out["y"].float() - y_std) ** 2 * ok).sum() / ok.sum().clamp(min=1)
        loss_b = F.binary_cross_entropy_with_logits(out["blocked"].float(), b["blocked"].float())
        region = ((b["maps"] >> 5) & 1).bool()
        fmask = region & (b["fut_valid"] > 0)[:, None, None, None]
        ftarget = ((b["maps"] >> BIT_FUTURE) & 1).float()
        fl = F.binary_cross_entropy_with_logits(out["future"].float(), ftarget, reduction="none")
        loss_f = (fl * fmask).sum() / fmask.sum().clamp(min=1)
        ex = b["excess"]
        emask = ~torch.isnan(ex) & (b["nodes"][..., 23] > 0.5)
        loss_e = (((out["excess"].float() - ex.nan_to_num(0.0)) ** 2) * emask).sum() / emask.sum().clamp(min=1)
        loss = loss_pi + loss_v + 0.5 * loss_b + 0.5 * loss_f + 0.25 * loss_e
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        self.opt.step()
        self.steps += 1
        loss, ce, ent, loss_v, loss_b, loss_f, loss_e = (t.detach() for t in (loss, ce, ent, loss_v, loss_b, loss_f, loss_e))
        return {"loss": float(loss), "policy_kl": float((ce - ent).mean()), "policy_ent": float(ent.mean()),
                "value_mse": float(loss_v), "blocked_bce": float(loss_b), "future_bce": float(loss_f),
                "excess_mse": float(loss_e), "grad_norm": float(gn), "lr": self.opt.param_groups[0]["lr"],
                "y_mu": float(m.y_mu), "y_sigma": float(m.y_sigma)}

    def state_dict(self) -> Dict:
        return {"opt": self.opt.state_dict(), "steps": self.steps, "norm_init": self.norm_init}

    def load_state_dict(self, d: Dict) -> None:
        self.opt.load_state_dict(d["opt"])
        self.steps = d["steps"]
        self.norm_init = d["norm_init"]
