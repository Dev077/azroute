"""Heterogeneous GATv2 that gives every net a priority score; nets are routed
highest score first."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, HeteroConv

from .graph import EDGE_TYPES, N_DIM, R_DIM

NODE_DIMS = {"net": N_DIM, "region": R_DIM}


class OrderGAT(nn.Module):
    """Residual + LayerNorm GATv2 layers over nets and regions, then an MLP from
    each net's embedding to its score."""

    def __init__(self, hidden: int = 128, heads: int = 4, layers: int = 4):
        super().__init__()
        assert hidden % heads == 0
        self.config = dict(hidden=hidden, heads=heads, layers=layers)
        self.proj = nn.ModuleDict({t: nn.Linear(d, hidden) for t, d in NODE_DIMS.items()})
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(layers):
            self.convs.append(HeteroConv({
                et: GATv2Conv(hidden, hidden // heads, heads=heads, edge_dim=1, add_self_loops=False)
                for et in EDGE_TYPES}, aggr="sum"))
            self.norms.append(nn.ModuleDict({t: nn.LayerNorm(hidden) for t in NODE_DIMS}))
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, data) -> torch.Tensor:
        """One score per net (in ``data["net"]`` order)."""
        x = {t: F.gelu(self.proj[t](data[t].x)) for t in NODE_DIMS}
        ei = {et: data[et].edge_index for et in EDGE_TYPES}
        ea = {et: data[et].edge_attr for et in EDGE_TYPES}
        for conv, norm in zip(self.convs, self.norms):
            out = conv(x, ei, edge_attr_dict=ea)
            x = {t: norm[t](x[t] + F.gelu(out[t])) if t in out else x[t] for t in x}
        return self.head(x["net"]).squeeze(1)


def save_checkpoint(path, model, optimizer=None, extra=None):
    torch.save({"config": model.config, "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict() if optimizer else None,
                "extra": extra or {}}, path)


def load_checkpoint(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    model = OrderGAT(**ck["config"]).to(device)
    model.load_state_dict(ck["state_dict"])
    return model, ck
