"""Heterogeneous GATv2 with a policy head (one logit per candidate route) and a
value head (predicted challenge score of the finished routing, plus the
probability that it finishes legally)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, HeteroConv, global_max_pool, global_mean_pool
from torch_geometric.utils import softmax as segment_softmax

from .graph import C_DIM, EDGE_TYPES, G_DIM, N_DIM, R_DIM

NODE_DIMS = {"net": N_DIM, "cand": C_DIM, "region": R_DIM}


def _mlp(i, h, o):
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Linear(h, h), nn.GELU(), nn.Linear(h, o))


class RouteGAT(nn.Module):
    def __init__(self, hidden: int = 152, heads: int = 4, layers: int = 6,
                 dropout: float = 0.0):
        super().__init__()
        assert hidden % heads == 0
        self.config = dict(hidden=hidden, heads=heads, layers=layers, dropout=dropout)
        self.proj = nn.ModuleDict({t: nn.Linear(d, hidden) for t, d in NODE_DIMS.items()})
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(layers):
            self.convs.append(HeteroConv({
                et: GATv2Conv(hidden, hidden // heads, heads=heads, edge_dim=1,
                              add_self_loops=False, dropout=dropout)
                for et in EDGE_TYPES}, aggr="sum"))
            self.norms.append(nn.ModuleDict({t: nn.LayerNorm(hidden) for t in NODE_DIMS}))
        self.policy = _mlp(2 * hidden, hidden, 1)
        self.value = _mlp(4 * hidden + G_DIM, hidden, 2)  # [score, legal logit]

    def forward(self, data):
        x = {t: F.gelu(self.proj[t](data[t].x)) for t in NODE_DIMS}
        ei = {et: data[et].edge_index for et in EDGE_TYPES}
        ea = {et: data[et].edge_attr for et in EDGE_TYPES}
        for conv, norm in zip(self.convs, self.norms):
            out = conv(x, ei, edge_attr_dict=ea)
            x = {t: norm[t](x[t] + F.gelu(out[t])) if t in out else x[t] for t in x}

        # policy: each candidate next to the net it would route
        of = data["cand", "of", "net"].edge_index
        net_of_cand = torch.empty_like(of[0])
        net_of_cand[of[0]] = of[1]
        logits = self.policy(torch.cat([x["cand"], x["net"][net_of_cand]], dim=1)).squeeze(1)

        # value: pooled graph summary
        g = data.g
        B = g.size(0)
        batch = {t: getattr(data[t], "batch", None) for t in NODE_DIMS}
        for t in batch:
            if batch[t] is None:
                batch[t] = torch.zeros(x[t].size(0), dtype=torch.long, device=g.device)
        pooled = torch.cat([
            global_mean_pool(x["net"], batch["net"], B),
            global_mean_pool(x["region"], batch["region"], B),
            global_mean_pool(x["cand"], batch["cand"], B),
            global_max_pool(x["cand"], batch["cand"], B),
            g,
        ], dim=1)
        v = self.value(pooled)
        return logits, v[:, 0], v[:, 1], batch["cand"]

    @torch.no_grad()
    def evaluate(self, data, device):
        """Single state -> (prior over candidates, predicted score, P(legal))."""
        self.eval()
        if str(device) != "cpu":
            data = data.clone().to(device)  # keep the cached CPU graph on the CPU
        logits, value, legal, cb = self(data)
        prior = segment_softmax(logits, cb).cpu().tolist()
        return prior, max(0.0, float(value[0])), float(torch.sigmoid(legal[0]))


def save_checkpoint(path, model, optimizer=None, extra=None):
    torch.save({"config": model.config, "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict() if optimizer else None,
                "extra": extra or {}}, path)


def load_checkpoint(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    model = RouteGAT(**ck["config"]).to(device)
    model.load_state_dict(ck["state_dict"])
    return model, ck
