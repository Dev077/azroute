"""Heterogeneous GATv2 with a policy head (one logit per kick on offer) and a
value head (how much better, in percent, the best board reachable from here in
the moves left is than the current one)."""
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
    """Stability by construction: every layer is residual + LayerNorm (no vanishing
    signal through depth); dropout on attention and on each layer's messages
    (regularisation, off in eval mode, i.e. during search)."""

    def __init__(self, hidden: int = 152, heads: int = 4, layers: int = 6,
                 dropout: float = 0.1):
        super().__init__()
        assert hidden % heads == 0
        self.config = dict(hidden=hidden, heads=heads, layers=layers, dropout=dropout)
        self.drop = nn.Dropout(dropout)
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
        self.value = _mlp(4 * hidden + G_DIM, hidden, 1)  # gain still to come, %
        self.compiled = False
        self._core_fn = self._core

    # -- torch.compile ------------------------------------------------------------
    def set_compiled(self, on: bool = True) -> None:
        """Compile the tensor core (GAT layers + heads) with torch.compile. Only
        ``_core`` is compiled, so the HeteroData unpacking stays in Python and
        checkpoints are unaffected. ``dynamic=True`` because every board has a
        different number of nodes and edges."""
        if on:
            import torch._dynamo as dynamo
            dynamo.config.cache_size_limit = max(dynamo.config.cache_size_limit, 64)
            self._core_fn = torch.compile(self._core, dynamic=True)
        else:
            self._core_fn = self._core
        self.compiled = on

    def forward(self, data):
        x = {t: data[t].x for t in NODE_DIMS}
        ei = {et: data[et].edge_index for et in EDGE_TYPES}
        ea = {et: data[et].edge_attr for et in EDGE_TYPES}
        g = data.g
        batch = {}
        for t in NODE_DIMS:
            b = getattr(data[t], "batch", None)
            batch[t] = b if b is not None else torch.zeros(x[t].size(0), dtype=torch.long,
                                                           device=g.device)
        of = data["cand", "of", "net"].edge_index
        logits, v = self._core_fn(x, ei, ea, of, g, batch["net"], batch["region"], batch["cand"])
        return logits, v[:, 0], batch["cand"]

    def _core(self, x_in, ei, ea, of, g, b_net, b_region, b_cand):
        x = {t: F.gelu(self.proj[t](x_in[t])) for t in NODE_DIMS}
        for conv, norm in zip(self.convs, self.norms):
            out = conv(x, ei, edge_attr_dict=ea)
            x = {t: norm[t](x[t] + self.drop(F.gelu(out[t]))) if t in out else x[t] for t in x}

        # policy: each candidate next to the net it would route
        net_of_cand = torch.empty_like(of[0])
        net_of_cand[of[0]] = of[1]
        logits = self.policy(torch.cat([x["cand"], x["net"][net_of_cand]], dim=1)).squeeze(1)

        # value: pooled graph summary
        B = g.size(0)
        pooled = torch.cat([
            global_mean_pool(x["net"], b_net, B),
            global_mean_pool(x["region"], b_region, B),
            global_mean_pool(x["cand"], b_cand, B),
            global_max_pool(x["cand"], b_cand, B),
            g,
        ], dim=1)
        return logits, self.value(pooled)

    @torch.no_grad()
    def evaluate(self, data, device):
        """Single state -> (prior over the moves, predicted gain still to come in %)."""
        self.eval()
        if str(device) != "cpu":
            data = data.clone().to(device)  # keep the cached CPU graph on the CPU
        logits, value, cb = self(data)
        policy = segment_softmax(logits, cb).cpu().tolist()
        return policy, float(value[0])


def save_checkpoint(path, model, optimizer=None, extra=None):
    torch.save({"config": model.config, "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict() if optimizer else None,
                "extra": extra or {}}, path)


def load_checkpoint(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    model = RouteGAT(**ck["config"]).to(device)
    model.load_state_dict(ck["state_dict"])
    return model, ck
