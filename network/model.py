"""The network: multi-scale CNN over the region + net interaction layers over the neighbourhood.

Grid backbone (multi-scale residual CNN)
    64x64 -> 32x32 -> 16x16 -> 32x32 -> 64x64 with skip connections, so every output cell
    sees the whole region; the low-resolution blocks also add region-wide pooled context.

Net interaction layers (one node per neighbourhood net, one pair vector per ordered net pair)
    Each layer:
      1. grid read: every net attends over the 32x32 decoder features (2x2 cells per token, which
         already carry the 16x16 long-range context), biased towards its own box;
      2. pair update: pair (i, j) is updated from both nets' states and from how much their
         grid-read attention overlaps (whether they look at the same cells);
      3. net exchange: nets attend to each other with scores biased by the pair vector, and
         the message j sends i is modulated by pair (i, j);
      4. feed-forward.
    Pairs start from exact conflict features (features.PAIR_FEATS) plus box geometry.

Outputs:
  logits   (B, N*M)       decision distribution over (net, mode); illegal = -inf
  y        (B,)           standardised log remaining-excess (de-standardise with y_mu / y_sigma)
  blocked  (B,)           logit: this reroute ends blocked
  future   (B, L, 64, 64) logit: cell used by a net routed later in this reroute
  excess   (B, N)         log(final net delay / ideal)
"""
from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .features import CROP, GLOBAL_FEATS, MAX_NETS, N_MODES, NODE_FEATS, PAIR_FEATS

N_BITS = 6          # input bits 0..5 per layer (bit 6 is a training target)
GEOM_FEATS = 6
GRID = 32           # resolution of the features each net reads

PRESETS = {
    # ~10 M parameters: the run's default
    "default": dict(widths=(96, 160, 224), depths=(2, 3, 3), dim=256, pair_dim=64, heads=8, net_layers=3),
    # ~2.5 M: if network calls turn out to be the bottleneck
    "small": dict(widths=(64, 96, 160), depths=(2, 2, 3), dim=160, pair_dim=32, heads=4, net_layers=2),
    # for tests and the local smoke run
    "tiny": dict(widths=(8, 16, 16), depths=(1, 1, 1), dim=16, pair_dim=8, heads=2, net_layers=1),
}


# ----------------------------------------------------------------------------- grid backbone
class ResBlock(nn.Module):
    """3x3 conv residual block; optionally adds region-wide pooled context after the first conv."""

    def __init__(self, c: int, region_context: bool = False):
        super().__init__()
        self.c1 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(c)
        self.c2 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(c)
        self.ctx = nn.Linear(2 * c, c) if region_context else None

    def forward(self, x, mask):
        y = F.relu(self.b1(self.c1(x)))
        if self.ctx is not None:
            area = mask.sum((-1, -2)).clamp(min=1)
            mean = (y * mask).sum((-1, -2)) / area
            mx = y.masked_fill(mask < 0.5, -1e4).amax((-1, -2))
            y = y + self.ctx(torch.cat([mean, mx], 1))[..., None, None]
        return F.relu(x + self.b2(self.c2(y)))


class Backbone(nn.Module):
    def __init__(self, in_ch: int, widths: Tuple[int, int, int], depths: Tuple[int, int, int]):
        super().__init__()
        c1, c2, c3 = widths
        d1, d2, d3 = depths
        self.stem = nn.Sequential(nn.Conv2d(in_ch, c1, 3, padding=1, bias=False), nn.BatchNorm2d(c1), nn.ReLU())
        self.enc1 = nn.ModuleList([ResBlock(c1) for _ in range(d1)])
        self.down1 = nn.Sequential(nn.Conv2d(c1, c2, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(c2), nn.ReLU())
        self.enc2 = nn.ModuleList([ResBlock(c2, True) for _ in range(d2)])
        self.down2 = nn.Sequential(nn.Conv2d(c2, c3, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(c3), nn.ReLU())
        self.enc3 = nn.ModuleList([ResBlock(c3, True) for _ in range(d3)])
        self.up2 = nn.Sequential(nn.Conv2d(c3, c2, 1, bias=False), nn.BatchNorm2d(c2))
        self.dec2 = ResBlock(c2, True)
        self.up1 = nn.Sequential(nn.Conv2d(c2, c1, 1, bias=False), nn.BatchNorm2d(c1))
        self.dec1 = ResBlock(c1)

    def forward(self, x, m64):
        m32 = F.max_pool2d(m64, 2)
        m16 = F.max_pool2d(m32, 2)
        h1 = self.stem(x)
        for b in self.enc1:
            h1 = b(h1, m64)
        h2 = self.down1(h1)
        for b in self.enc2:
            h2 = b(h2, m32)
        h3 = self.down2(h2)
        for b in self.enc3:
            h3 = b(h3, m16)
        u2 = self.dec2(F.relu(h2 + F.interpolate(self.up2(h3), scale_factor=2, mode="nearest")), m32)
        u1 = self.dec1(F.relu(h1 + F.interpolate(self.up1(u2), scale_factor=2, mode="nearest")), m64)
        # full-resolution features; 32x32 decoder features (read by the nets) + mask; 16x16 features + mask
        return u1, u2, m32, h3, m16


# ----------------------------------------------------------------------------- net interaction
def _geometry(rois: torch.Tensor) -> torch.Tensor:
    """(B, N, 4) canvas boxes -> (B, N, N, 6) pair geometry (i = row / receiver, j = column / sender)."""
    r = rois.float()
    cx, cy = (r[..., 0] + r[..., 2]) / 2, (r[..., 1] + r[..., 3]) / 2
    w, h = r[..., 2] - r[..., 0] + 1, r[..., 3] - r[..., 1] + 1
    ix = (torch.minimum(r[..., :, None, 2], r[..., None, :, 2]) - torch.maximum(r[..., :, None, 0], r[..., None, :, 0]) + 1).clamp(min=0)
    iy = (torch.minimum(r[..., :, None, 3], r[..., None, :, 3]) - torch.maximum(r[..., :, None, 1], r[..., None, :, 1]) + 1).clamp(min=0)
    inter = ix * iy
    area = w * h
    union = area[..., :, None] + area[..., None, :] - inter
    dx = (cx[..., None, :] - cx[..., :, None]) / CROP
    dy = (cy[..., None, :] - cy[..., :, None]) / CROP
    return torch.stack([inter / union.clamp(min=1), dx, dy, dx.abs() + dy.abs(),
                        torch.log(area[..., None, :].clamp(min=1) / area[..., :, None].clamp(min=1)) / 4,
                        inter / area[..., :, None].clamp(min=1)], dim=-1)


def _box_bias(rois: torch.Tensor, grid: int) -> torch.Tensor:
    """(B, N, 4) canvas boxes -> (B, N, grid*grid) 1.0 inside the box (expanded by one grid cell)."""
    s = CROP // grid
    ar = torch.arange(grid, device=rois.device)
    x0, y0, x1, y1 = [(rois[..., i, None].long() // s) for i in range(4)]
    mx = ((ar >= x0 - 1) & (ar <= x1 + 1)).float()
    my = ((ar >= y0 - 1) & (ar <= y1 + 1)).float()
    return (my.unsqueeze(-1) * mx.unsqueeze(-2)).flatten(-2)


class NetInteractionLayer(nn.Module):
    def __init__(self, dim: int, pair_dim: int, heads: int):
        super().__init__()
        self.h, self.dh = heads, dim // heads
        # 1. grid read
        self.ln_g = nn.LayerNorm(dim)
        self.gq = nn.Linear(dim, dim)
        self.gk = nn.Linear(dim, dim)
        self.gv = nn.Linear(dim, dim)
        self.go = nn.Linear(dim, dim)
        self.outside = nn.Parameter(torch.full((heads,), -2.0))   # learned penalty for cells outside the box
        # 2. pair update
        self.ln_pn = nn.LayerNorm(dim)
        self.p_i = nn.Linear(dim, pair_dim)
        self.p_j = nn.Linear(dim, pair_dim)
        self.p_ov = nn.Linear(heads, pair_dim)
        self.ln_p = nn.LayerNorm(pair_dim)
        self.p_ff = nn.Sequential(nn.Linear(pair_dim, 2 * pair_dim), nn.GELU(), nn.Linear(2 * pair_dim, pair_dim))
        # 3. net exchange
        self.ln_n = nn.LayerNorm(dim)
        self.nq = nn.Linear(dim, dim)
        self.nk = nn.Linear(dim, dim)
        self.nv = nn.Linear(dim, dim)
        self.ln_pb = nn.LayerNorm(pair_dim)
        self.pb = nn.Linear(pair_dim, heads)          # pair -> attention bias
        self.pv = nn.Linear(pair_dim, dim)            # pair -> message modulation
        self.no = nn.Linear(dim, dim)
        # 4. feed-forward
        self.ln_f = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(self, h, z, grid, grid_mask, box, valid):
        B, N, D = h.shape
        H, dh = self.h, self.dh
        # 1. grid read: (B, H, N, K)
        x = self.ln_g(h)
        q = self.gq(x).view(B, N, H, dh).transpose(1, 2)
        k = self.gk(grid).view(B, -1, H, dh).transpose(1, 2)
        v = self.gv(grid).view(B, -1, H, dh).transpose(1, 2)
        s = q @ k.transpose(-1, -2) / math.sqrt(dh)
        s = s + (1 - box[:, None]) * self.outside[None, :, None, None]
        s = s.masked_fill(grid_mask[:, None, None, :] < 0.5, float("-inf"))
        att = torch.softmax(s, -1)
        h = h + self.go((att @ v).transpose(1, 2).reshape(B, N, D))
        # 2. pair update
        x = self.ln_pn(h)
        overlap = torch.einsum("bhik,bhjk->bijh", att, att)            # shared attention mass
        z = z + self.p_i(x)[:, :, None] + self.p_j(x)[:, None, :] + self.p_ov(overlap)
        z = z + self.p_ff(self.ln_p(z))
        # 3. net exchange (row i receives from column j)
        x = self.ln_n(h)
        zb = self.ln_pb(z)
        q = self.nq(x).view(B, N, H, dh)
        k = self.nk(x).view(B, N, H, dh)
        s = torch.einsum("bihd,bjhd->bijh", q, k) / math.sqrt(dh) + self.pb(zb)
        s = s.masked_fill(~valid[:, None, :, None], float("-inf"))
        a = torch.softmax(s, 2)                                          # over senders j
        msg = self.nv(x)[:, None, :, :] + self.pv(zb)                   # (B, i, j, D)
        msg = msg.view(B, N, N, H, dh)
        out = torch.einsum("bijh,bijhd->bihd", a, msg).reshape(B, N, D)
        h = h + self.no(out)
        # 4. feed-forward
        h = h + self.ff(self.ln_f(h))
        return h, z


# ----------------------------------------------------------------------------- full network
class RouteNet(nn.Module):
    def __init__(self, layers: int = 6, widths=(96, 160, 224), depths=(2, 3, 3), dim: int = 256,
                 pair_dim: int = 64, heads: int = 8, net_layers: int = 3):
        super().__init__()
        self.L = layers
        self.config = dict(layers=layers, widths=tuple(widths), depths=tuple(depths), dim=dim,
                           pair_dim=pair_dim, heads=heads, net_layers=net_layers)
        c1, c2, c3 = widths
        in_ch = N_BITS * layers + layers + 1 + 1 + 2   # bits, layer delays, via delay, demand, coords
        self.backbone = Backbone(in_ch, widths, depths)
        self.future = nn.Sequential(nn.Conv2d(c1, c1, 3, padding=1), nn.ReLU(), nn.Conv2d(c1, layers, 1))
        self.grid_proj = nn.Linear(c2, dim)
        self.grid_pos = nn.Parameter(torch.zeros(1, GRID * GRID, dim))
        self.node_in = nn.Sequential(nn.Linear(NODE_FEATS + GLOBAL_FEATS + c1, dim), nn.GELU(), nn.Linear(dim, dim))
        self.pair_in = nn.Sequential(nn.Linear(GEOM_FEATS + 2 * PAIR_FEATS, pair_dim), nn.GELU(),
                                     nn.Linear(pair_dim, pair_dim))
        self.layers = nn.ModuleList([NetInteractionLayer(dim, pair_dim, heads) for _ in range(net_layers)])
        self.ln_out = nn.LayerNorm(dim)
        self.policy = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, N_MODES))
        self.excess = nn.Sequential(nn.Linear(dim, dim // 2), nn.GELU(), nn.Linear(dim // 2, 1))
        self.pool_q = nn.Linear(dim, 1)
        self.trunk = nn.Sequential(nn.Linear(2 * dim + c3 + GLOBAL_FEATS, dim), nn.GELU(),
                                   nn.Linear(dim, dim), nn.GELU())
        self.value = nn.Linear(dim, 1)
        self.blocked = nn.Linear(dim, 1)
        self.register_buffer("y_mu", torch.zeros(()))
        self.register_buffer("y_sigma", torch.ones(()))

    @staticmethod
    def preset(name: str, layers: int = 6) -> "RouteNet":
        return RouteNet(layers=layers, **PRESETS[name])

    def build_input(self, maps, nodes, rois, ld):
        B = maps.shape[0]
        sh = torch.arange(N_BITS, device=maps.device, dtype=torch.uint8).view(1, 1, N_BITS, 1, 1)
        bits = ((maps.unsqueeze(2) >> sh) & 1).float().reshape(B, self.L * N_BITS, CROP, CROP)
        region = ((maps[:, :1] >> 5) & 1).float()
        planes = ld[:, :, None, None] * region                     # L layer delays + via delay
        roi = _roi_masks(rois)
        demand = (roi * nodes[..., 0:1, None] * nodes[..., 21:22, None]).sum(1, keepdim=True)
        ar = torch.linspace(-1, 1, CROP, device=maps.device)
        coords = torch.stack(torch.meshgrid(ar, ar, indexing="ij")).unsqueeze(0).expand(B, -1, -1, -1)
        return torch.cat([bits, planes, demand, coords], 1), roi, region

    def forward(self, maps, nodes, rois, glob, legal, ld, pairs) -> Dict[str, torch.Tensor]:
        x, roi, region = self.build_input(maps, nodes, rois, ld)
        f64, f32, m32, f16, m16 = self.backbone(x, region)
        valid = nodes[..., 23] > 0.5                                 # (B, N)
        area = roi.sum((-1, -2)).clamp(min=1)
        pooled = torch.einsum("bnhw,bchw->bnc", roi, f64) / area[..., None]
        B, N = nodes.shape[:2]
        h = self.node_in(torch.cat([nodes, glob[:, None, :].expand(-1, N, -1), pooled], -1))
        conflict = pairs.float()
        z = self.pair_in(torch.cat([_geometry(rois), conflict, conflict.transpose(1, 2)], -1))
        grid = self.grid_proj(f32.flatten(2).transpose(1, 2)) + self.grid_pos
        gmask = m32.flatten(1)
        box = _box_bias(rois, GRID)
        for layer in self.layers:
            h, z = layer(h, z, grid, gmask, box, valid)
        h = self.ln_out(h)
        logits = self.policy(h).masked_fill(~legal, float("-inf")).flatten(1)
        w = self.pool_q(h).squeeze(-1).masked_fill(~valid, float("-inf"))
        attn_pool = (torch.softmax(w, -1)[..., None] * h).sum(1)
        mx = h.masked_fill(~valid[..., None], -1e4).max(1).values
        gpool = (f16 * m16).sum((-1, -2)) / m16.sum((-1, -2)).clamp(min=1)
        t = self.trunk(torch.cat([attn_pool, mx, gpool, glob], -1))
        return {"logits": logits, "y": self.value(t).squeeze(-1), "blocked": self.blocked(t).squeeze(-1),
                "future": self.future(f64), "excess": self.excess(h).squeeze(-1)}


def _roi_masks(rois: torch.Tensor) -> torch.Tensor:
    """(B, N, 4) inclusive canvas boxes -> (B, N, CROP, CROP) float masks."""
    ar = torch.arange(CROP, device=rois.device)
    x0, y0, x1, y1 = [rois[..., i, None].long() for i in range(4)]
    mx = ((ar >= x0) & (ar <= x1)).float()
    my = ((ar >= y0) & (ar <= y1)).float()
    return my.unsqueeze(-1) * mx.unsqueeze(-2)


BATCH_KEYS = ["maps", "nodes", "rois", "glob", "legal", "ld", "pairs"]


def collate(feats: List[Dict[str, np.ndarray]], device) -> Dict[str, torch.Tensor]:
    out = {}
    for k in BATCH_KEYS:
        t = torch.from_numpy(np.stack([f[k] for f in feats]))
        if k in ("nodes", "glob", "ld", "pairs"):
            t = t.float()
        elif k == "rois":
            t = t.long()
        out[k] = t.to(device, non_blocking=True)
    return out


def save_model(model: RouteNet, path: str, extra: Dict = None) -> None:
    import os
    tmp = path + ".tmp"
    torch.save({"config": model.config, "state": model.state_dict(), **(extra or {})}, tmp)
    os.replace(tmp, path)


def load_model(path: str, device) -> Tuple[RouteNet, Dict]:
    ck = torch.load(path, map_location=device, weights_only=False)
    m = RouteNet(**ck["config"]).to(device)
    m.load_state_dict(ck["state"])
    return m, ck
