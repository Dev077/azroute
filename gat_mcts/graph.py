"""Turn a routing (and the kicks on offer) into a heterogeneous graph for the GAT.

Node types
  net     every net, with its current route
  cand    one node per move: kick net N at strength k (these are the actions)
  region  a tile x tile patch of one routing layer, with its congestion

Edge types (each with a 1-d edge weight, and a reverse edge so information flows
both ways)
  cand -of-> net, net -has-> cand
  cand -uses-> region          where the kicked net's ideal route runs (weight: fraction)
  cand -hits-> net             nets whose wires that ideal route crosses (weight: share of it)
  net -pin-> region            where the net's pins sit
  net -route-> region          where the net's wires sit (weight: fraction)
  region -adj-> region         neighbouring tiles on a layer (1) and across a via (0)
"""
from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import torch
from torch_geometric.data import HeteroData

from .problem import K, KICKS, Board, Move, State

N_DIM = 11
C_DIM = 7 + K
R_DIM = 8
G_DIM = 9

EDGE_TYPES = [
    ("cand", "of", "net"), ("net", "has", "cand"),
    ("cand", "uses", "region"), ("region", "used_by", "cand"),
    ("cand", "hits", "net"), ("net", "hit_by", "cand"),
    ("net", "pin", "region"), ("region", "pin_of", "net"),
    ("net", "route", "region"), ("region", "route_of", "net"),
    ("region", "adj", "region"),
]

_region_adj_cache: list = [None, None]  # [problem, (edge_index, edge_attr)]


def _region_adjacency(prob):
    if _region_adj_cache[0] is not prob:
        TX, TY, L = prob.TX, prob.TY, prob.L
        src, dst, w = [], [], []
        for z in range(L):
            for ty in range(TY):
                for tx in range(TX):
                    r = (z * TY + ty) * TX + tx
                    for dx, dy, dz, wt in ((1, 0, 0, 1.0), (0, 1, 0, 1.0), (0, 0, 1, 0.0)):
                        nx, ny, nz = tx + dx, ty + dy, z + dz
                        if nx < TX and ny < TY and nz < L:
                            q = (nz * TY + ny) * TX + nx
                            src += [r, q]; dst += [q, r]; w += [wt, wt]
        _region_adj_cache[0] = prob
        _region_adj_cache[1] = (torch.tensor([src, dst], dtype=torch.long),
                                torch.tensor(w, dtype=torch.float32).unsqueeze(1))
    return _region_adj_cache[1]


def _edges(pairs: List[Tuple[int, int]], weights: List[float]):
    if not pairs:
        return torch.zeros((2, 0), dtype=torch.long), torch.zeros((0, 1))
    ei = torch.tensor(pairs, dtype=torch.long).t().contiguous()
    return ei, torch.tensor(weights, dtype=torch.float32).unsqueeze(1)


def build_graph(board: Board, state: State, actions: Sequence[Move], depth: int,
                best_delay: int) -> HeteroData:
    """``actions`` are the moves on offer, in action order; ``depth`` is how many
    moves of the episode have been played; ``best_delay`` is the best board's
    total delay so far this episode."""
    prob = board.prob
    data = HeteroData()
    n, R, L = prob.n, prob.R, prob.L
    lb, routes = prob.lb, state.routes
    size = prob.region_size
    kick_nets = sorted({nid for nid, _k in actions})
    hits: Dict[int, Dict[int, int]] = {nid: state.ideal_hits(prob, nid) for nid in kick_nets}
    gap = [routes[i].delay / lb[i] - 1.0 for i in range(n)]

    # ---- region nodes ----
    occ = state.region_occ
    usage = [0.0] * R
    for nid in kick_nets:
        for r, cnt in prob.ideal[nid].regions:
            usage[r] += cnt
    TX, TY = prob.TX, prob.TY
    rx = []
    for r in range(R):
        z = prob.region_z(r)
        ty, tx = divmod(r % (TX * TY), TX)
        rx.append([
            z / max(1, L - 1),
            1.0 - prob.cheapness[z],
            1.0 if z in (0, L - 1) else 0.0,
            (tx + 0.5) / TX,
            (ty + 0.5) / TY,
            occ[r] / size[r],
            prob.region_pins[r] / size[r],
            math.log1p(usage[r] / size[r]),
        ])
    data["region"].x = torch.tensor(rx, dtype=torch.float32)

    # ---- net nodes ----
    start = board.start.routes
    kickable = set(kick_nets)
    hit_count = [0] * n
    for hs in hits.values():
        for o in hs:
            hit_count[o] += 1
    nx = []
    for nid in range(n):
        x0, x1, y0, y1, _, _ = prob.bbox[nid]
        pins = len(prob.net_pins[nid])
        c = routes[nid]
        nx.append([
            math.log1p(pins - 1) / math.log(8),
            prob.cross[nid],
            (x1 - x0) / prob.W,
            (y1 - y0) / prob.H,
            math.log1p(lb[nid] / prob.mean_lb),
            gap[nid],
            math.log1p((c.delay - lb[nid]) / prob.mean_lb),
            c.vias / (pins * max(1, L - 1)),
            1.0 if nid in kickable else 0.0,
            0.0 if c is start[nid] else 1.0,          # changed since the episode began
            math.log1p(hit_count[nid]) / math.log(8),  # how many kicks would push it aside
        ])
    data["net"].x = torch.tensor(nx, dtype=torch.float32)

    # ---- move nodes ----
    cx = []
    for nid, k in actions:
        hs = hits[nid]
        ideal = prob.ideal[nid]
        crossed = sum(hs.values())
        onehot = [0.0] * K
        onehot[k] = 1.0
        cx.append([
            gap[nid],
            (routes[nid].delay - lb[nid]) / prob.mean_lb,   # most it could save
            len(hs) / 4.0,
            crossed / len(ideal.vids),
            sum(gap[o] for o in hs) / 4.0,                 # how bent the crossed nets already are
            sum(1 for o in hs if o in kickable) / 4.0,
            KICKS[k] / KICKS[-1],
        ] + onehot)
    data["cand"].x = torch.tensor(cx, dtype=torch.float32).reshape(-1, C_DIM)

    # ---- edges ----
    of = [(i, nid) for i, (nid, _k) in enumerate(actions)]
    ei, ea = _edges(of, [1.0] * len(of))
    data["cand", "of", "net"].edge_index, data["cand", "of", "net"].edge_attr = ei, ea
    data["net", "has", "cand"].edge_index, data["net", "has", "cand"].edge_attr = ei.flip(0), ea

    uses, uw = [], []
    for i, (nid, _k) in enumerate(actions):
        for r, cnt in prob.ideal[nid].regions:
            uses.append((i, r)); uw.append(cnt / size[r])
    ei, ea = _edges(uses, uw)
    data["cand", "uses", "region"].edge_index, data["cand", "uses", "region"].edge_attr = ei, ea
    data["region", "used_by", "cand"].edge_index, data["region", "used_by", "cand"].edge_attr = ei.flip(0), ea

    hit_e, hw = [], []
    for i, (nid, _k) in enumerate(actions):
        nv = len(prob.ideal[nid].vids)
        for o, cnt in hits[nid].items():
            hit_e.append((i, o)); hw.append(cnt / nv)
    ei, ea = _edges(hit_e, hw)
    data["cand", "hits", "net"].edge_index, data["cand", "hits", "net"].edge_attr = ei, ea
    data["net", "hit_by", "cand"].edge_index, data["net", "hit_by", "cand"].edge_attr = ei.flip(0), ea

    pin_e = []
    for nid in range(n):
        for r in {prob.region_of[v] for v in prob.net_pins[nid]}:
            pin_e.append((nid, r))
    ei, ea = _edges(pin_e, [1.0] * len(pin_e))
    data["net", "pin", "region"].edge_index, data["net", "pin", "region"].edge_attr = ei, ea
    data["region", "pin_of", "net"].edge_index, data["region", "pin_of", "net"].edge_attr = ei.flip(0), ea

    route_e, rw = [], []
    for nid, c in routes.items():
        for r, cnt in c.regions:
            route_e.append((nid, r)); rw.append(cnt / size[r])
    ei, ea = _edges(route_e, rw)
    data["net", "route", "region"].edge_index, data["net", "route", "region"].edge_attr = ei, ea
    data["region", "route_of", "net"].edge_index, data["region", "route_of", "net"].edge_attr = ei.flip(0), ea

    ei, ea = _region_adjacency(prob)
    data["region", "adj", "region"].edge_index, data["region", "adj", "region"].edge_attr = ei, ea

    # ---- graph-level features ----
    data.g = torch.tensor([[
        (board.moves - depth) / max(1, board.moves),          # moves left
        prob.lb_total / state.delay,                            # how close to the lower bound
        prob.baseline_total / prob.lb_total,                    # the best score this case could ever reach
        100.0 * (board.start.delay / state.delay - 1.0),        # gain so far this episode (%)
        100.0 * (state.delay / best_delay - 1.0),               # how far above the best board so far (%)
        math.log(max(2, n)) / math.log(1000),
        math.log(prob.W * prob.H / 1600.0 + 1.0),
        sum(occ) / prob.V,
        sum(1 for i in range(n) if routes[i].delay == lb[i]) / n,
    ]], dtype=torch.float32)
    return data
