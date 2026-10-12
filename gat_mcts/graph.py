"""Turn a routing state into a heterogeneous graph for the GAT.

Node types
  net     every net (routed, in the frontier, or still waiting)
  cand    one node per candidate route of the frontier nets: these are the actions
  region  a tile x tile patch of one routing layer, with its congestion

Edge types (each with a 1-d edge weight, and a reverse edge so information flows
both ways)
  cand -of-> net, net -has-> cand
  cand -uses-> region          weight: fraction of the region the candidate uses
  net -pin-> region            where the net's pins sit
  net -route-> region          where a routed net's wires sit (weight: fraction)
  region -adj-> region         neighbouring tiles on a layer (1) and across a via (0)
  cand -conflict-> cand        candidates of different nets sharing a vertex
"""
from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import torch
from torch_geometric.data import HeteroData

from .problem import K, Candidate, Problem, State

N_DIM = 11
C_DIM = 8 + K
R_DIM = 9
G_DIM = 8

EDGE_TYPES = [
    ("cand", "of", "net"), ("net", "has", "cand"),
    ("cand", "uses", "region"), ("region", "used_by", "cand"),
    ("net", "pin", "region"), ("region", "pin_of", "net"),
    ("net", "route", "region"), ("region", "route_of", "net"),
    ("region", "adj", "region"),
    ("cand", "conflict", "cand"),
]

_region_adj_cache: list = [None, None]  # [problem, (edge_index, edge_attr)]


def _region_adjacency(prob: Problem):
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


def build_graph(prob: Problem, state: State, frontier: Sequence[int],
                actions: Sequence[Candidate]) -> HeteroData:
    """``actions`` must be the frontier nets' candidates, in action order."""
    data = HeteroData()
    n, R, L = prob.n, prob.R, prob.L
    lb = prob.lb
    front = set(frontier)
    by_net: Dict[int, List[Candidate]] = {}
    for c in actions:
        by_net.setdefault(c.net, []).append(c)
    best = {nid: min(c.delay for c in cs) for nid, cs in by_net.items()}

    # ---- region nodes ----
    dem = state.demand(prob)
    occ = state.region_occ
    size = prob.region_size
    usage = [0.0] * R
    for c in actions:
        for r, cnt in c.regions:
            usage[r] += cnt
    TX, TY = prob.TX, prob.TY
    rx = []
    for r in range(R):
        z = prob.region_z(r)
        rem = r % (TX * TY)
        ty, tx = divmod(rem, TX)
        rx.append([
            z / max(1, L - 1),
            1.0 - prob.cheapness[z],
            1.0 if z in (0, L - 1) else 0.0,
            (tx + 0.5) / TX,
            (ty + 0.5) / TY,
            occ[r] / size[r],
            prob.region_pins[r] / size[r],
            min(4.0, dem[r] * 4.0),
            math.log1p(usage[r] / size[r]),
        ])
    data["region"].x = torch.tensor(rx, dtype=torch.float32)

    # ---- net nodes ----
    nx = []
    for nid in range(n):
        x0, x1, y0, y1, _, _ = prob.bbox[nid]
        routed = nid in state.routes
        cs = by_net.get(nid, ())
        nx.append([
            math.log1p(len(prob.net_pins[nid]) - 1) / math.log(8),
            prob.cross[nid],
            (x1 - x0) / prob.W,
            (y1 - y0) / prob.H,
            math.log1p(lb[nid] / prob.mean_lb),
            1.0 if routed else 0.0,
            1.0 if nid in front else 0.0,
            (state.routes[nid].delay / lb[nid] - 1.0) if routed else 0.0,
            len(cs) / K,
            (best[nid] / lb[nid] - 1.0) if cs else 0.0,
            prob.rank[nid] / max(1, n - 1),
        ])
    data["net"].x = torch.tensor(nx, dtype=torch.float32)

    # ---- candidate nodes + conflicts ----
    vert_cands: Dict[int, List[int]] = {}
    for i, c in enumerate(actions):
        for v in c.vids:
            vert_cands.setdefault(v, []).append(i)
    shared: Dict[Tuple[int, int], int] = {}
    for idxs in vert_cands.values():
        if len(idxs) < 2:
            continue
        for a in idxs:
            for b in idxs:
                if a != b and actions[a].net != actions[b].net:
                    shared[(a, b)] = shared.get((a, b), 0) + 1
    conflict_amount = [0] * len(actions)
    conflict_nets: List[set] = [set() for _ in actions]
    for (a, b), cnt in shared.items():
        conflict_amount[a] += cnt
        conflict_nets[a].add(actions[b].net)

    unrouted = state.unrouted
    cx = []
    for i, c in enumerate(actions):
        pins = prob.net_pins[c.net]
        x0, x1, y0, y1, z0, z1 = prob.bbox[c.net]
        hpwl = (x1 - x0) + (y1 - y0) + (z1 - z0)
        inplane = [(a, b) for a, b in c.edges if b - a != prob.wh]
        cheap = (sum(prob.cheapness[a // prob.wh] for a, _ in inplane) / len(inplane)
                 if inplane else 0.0)
        hug = set()
        for v in c.vids:
            for o in prob.pin_adj.get(v, ()):
                if o != c.net and o in unrouted:
                    hug.add(o)
        onehot = [0.0] * K
        onehot[c.profile] = 1.0
        cx.append([
            c.delay / lb[c.net] - 1.0,
            (c.delay - best[c.net]) / lb[c.net],
            len(c.edges) / (hpwl + len(pins)),
            c.vias / (len(pins) * max(1, L - 1)),
            cheap,
            conflict_amount[i] / len(c.vids),
            len(conflict_nets[i]) / max(1, prob.frontier_size),
            len(hug) / len(pins),
        ] + onehot)
    data["cand"].x = torch.tensor(cx, dtype=torch.float32).reshape(-1, C_DIM)

    # ---- edges ----
    of = [(i, c.net) for i, c in enumerate(actions)]
    ei, ea = _edges(of, [1.0] * len(of))
    data["cand", "of", "net"].edge_index, data["cand", "of", "net"].edge_attr = ei, ea
    data["net", "has", "cand"].edge_index, data["net", "has", "cand"].edge_attr = ei.flip(0), ea

    uses, uw = [], []
    for i, c in enumerate(actions):
        for r, cnt in c.regions:
            uses.append((i, r)); uw.append(cnt / size[r])
    ei, ea = _edges(uses, uw)
    data["cand", "uses", "region"].edge_index, data["cand", "uses", "region"].edge_attr = ei, ea
    data["region", "used_by", "cand"].edge_index, data["region", "used_by", "cand"].edge_attr = ei.flip(0), ea

    pin_e = []
    for nid in range(n):
        for r in {prob.region_of[v] for v in prob.net_pins[nid]}:
            pin_e.append((nid, r))
    ei, ea = _edges(pin_e, [1.0] * len(pin_e))
    data["net", "pin", "region"].edge_index, data["net", "pin", "region"].edge_attr = ei, ea
    data["region", "pin_of", "net"].edge_index, data["region", "pin_of", "net"].edge_attr = ei.flip(0), ea

    route_e, rw = [], []
    for nid, c in state.routes.items():
        for r, cnt in c.regions:
            route_e.append((nid, r)); rw.append(cnt / size[r])
    ei, ea = _edges(route_e, rw)
    data["net", "route", "region"].edge_index, data["net", "route", "region"].edge_attr = ei, ea
    data["region", "route_of", "net"].edge_index, data["region", "route_of", "net"].edge_attr = ei.flip(0), ea

    ei, ea = _region_adjacency(prob)
    data["region", "adj", "region"].edge_index, data["region", "adj", "region"].edge_attr = ei, ea

    pairs = list(shared.keys())
    ei, ea = _edges(pairs, [shared[p] / len(actions[p[0]].vids) for p in pairs])
    data["cand", "conflict", "cand"].edge_index, data["cand", "conflict", "cand"].edge_attr = ei, ea

    # ---- graph-level features ----
    routed_lb = sum(lb[i] for i in state.routes)
    data.g = torch.tensor([[
        len(state.routes) / n,
        math.log(max(2, n)) / math.log(1000),
        (state.delay / routed_lb - 1.0) if routed_lb else 0.0,
        prob.baseline_total / prob.lb_total,   # the best score this case could ever reach
        sum(occ) / prob.V,
        sum(dem) / R,
        math.log(prob.W * prob.H / 1600.0 + 1.0),
        sum(state.rips.values()) / n,
    ]], dtype=torch.float32)
    return data
