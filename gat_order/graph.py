"""Turn a board's netlist into a heterogeneous graph for the GAT (one per board;
the order is decided before any routing, so nothing in it changes).

Node types
  net     every net: its size, bounding box, how crowded its area is, and its
          place in the router's default order (largest bounding box first)
  region  a tile x tile patch of the die (all layers): pin density and the
          wiring demand of the nets whose bounding boxes cover it

Edge types (each with a 1-d weight)
  net -covers-> region, region -covered_by-> net   weight: 1 / tiles the net covers
  net -overlaps-> net                              weight: shared bbox area / smaller bbox area
  region -adj-> region                             neighbouring tiles
"""
from __future__ import annotations

import math
from typing import List, Tuple

import torch
from torch_geometric.data import HeteroData

from m3d.model import Instance

from .router import default_order

N_DIM = 12
R_DIM = 6
EDGE_TYPES = [
    ("net", "covers", "region"), ("region", "covered_by", "net"),
    ("net", "overlaps", "net"),
    ("region", "adj", "region"),
]


def _edges(pairs: List[Tuple[int, int]], weights: List[float]):
    if not pairs:
        return torch.zeros((2, 0), dtype=torch.long), torch.zeros((0, 1))
    ei = torch.tensor(pairs, dtype=torch.long).t().contiguous()
    return ei, torch.tensor(weights, dtype=torch.float32).unsqueeze(1)


def build_graph(inst: Instance) -> HeteroData:
    W, H, L = inst.width, inst.height, inst.layers
    pins = inst.pin_by_id()
    n = len(inst.nets)
    boxes = []
    for net in inst.nets:
        ps = [pins[p] for p in net.pins()]
        boxes.append((min(p.x for p in ps), max(p.x for p in ps),
                      min(p.y for p in ps), max(p.y for p in ps),
                      len({p.z for p in ps}) > 1, len(ps) - 1))
    rank = {nid: r for r, nid in enumerate(default_order(inst))}

    T = max(2, math.ceil(max(W, H) / 8))
    TX, TY = math.ceil(W / T), math.ceil(H / T)
    R = TX * TY
    tile_pins = [0] * R
    for p in inst.pins:
        tile_pins[(p.y // T) * TX + p.x // T] += 1
    demand, count = [0.0] * R, [0] * R
    covers, cw = [], []
    for i, (x0, x1, y0, y1, _c, _s) in enumerate(boxes):
        dens = ((x1 - x0) + (y1 - y0) + 1) / ((x1 - x0 + 1) * (y1 - y0 + 1))  # RUDY
        tiles = [ty * TX + tx for ty in range(y0 // T, y1 // T + 1) for tx in range(x0 // T, x1 // T + 1)]
        for r in tiles:
            demand[r] += dens
            count[r] += 1
            covers.append((i, r)); cw.append(1.0 / len(tiles))

    # pins of other nets inside each net's box, and bounding-box overlaps
    net_of_pin = {p: i for i, net in enumerate(inst.nets) for p in net.pins()}
    inside = [0] * n
    for p in inst.pins:
        for i, (x0, x1, y0, y1, _c, _s) in enumerate(boxes):
            if x0 <= p.x <= x1 and y0 <= p.y <= y1 and net_of_pin.get(p.id) != i:
                inside[i] += 1
    over, ow, n_over = [], [], [0] * n
    for i in range(n):
        a = boxes[i]
        for j in range(i + 1, n):
            b = boxes[j]
            dx = min(a[1], b[1]) - max(a[0], b[0]) + 1
            dy = min(a[3], b[3]) - max(a[2], b[2]) + 1
            if dx > 0 and dy > 0:
                small = min((a[1] - a[0] + 1) * (a[3] - a[2] + 1), (b[1] - b[0] + 1) * (b[3] - b[2] + 1))
                w = dx * dy / small
                over += [(i, j), (j, i)]; ow += [w, w]
                n_over[i] += 1; n_over[j] += 1

    data = HeteroData()
    nx = []
    for i, net in enumerate(inst.nets):
        x0, x1, y0, y1, cross, sinks = boxes[i]
        area = (x1 - x0 + 1) * (y1 - y0 + 1)
        nx.append([
            math.log1p(sinks) / math.log(8),
            1.0 if cross else 0.0,
            (x1 - x0) / W,
            (y1 - y0) / H,
            ((x1 - x0) + (y1 - y0)) / (W + H),
            area / (W * H),
            math.log1p(inside[i] / area * 10),
            n_over[i] / max(1, n - 1),
            rank[net.id] / max(1, n - 1),
            (x0 + x1) / 2 / W,
            (y0 + y1) / 2 / H,
            math.log(max(2, n)) / math.log(1000),
        ])
    data["net"].x = torch.tensor(nx, dtype=torch.float32)
    rx = []
    for r in range(R):
        ty, tx = divmod(r, TX)
        rx.append([(tx + 0.5) / TX, (ty + 0.5) / TY, tile_pins[r] / (T * T),
                   math.log1p(demand[r]), count[r] / max(1, n), L / 10.0])
    data["region"].x = torch.tensor(rx, dtype=torch.float32)

    ei, ea = _edges(covers, cw)
    data["net", "covers", "region"].edge_index, data["net", "covers", "region"].edge_attr = ei, ea
    data["region", "covered_by", "net"].edge_index, data["region", "covered_by", "net"].edge_attr = ei.flip(0), ea
    ei, ea = _edges(over, ow)
    data["net", "overlaps", "net"].edge_index, data["net", "overlaps", "net"].edge_attr = ei, ea
    adj = []
    for r in range(R):
        ty, tx = divmod(r, TX)
        if tx + 1 < TX:
            adj += [(r, r + 1), (r + 1, r)]
        if ty + 1 < TY:
            adj += [(r, r + TX), (r + TX, r)]
    ei, ea = _edges(adj, [1.0] * len(adj))
    data["region", "adj", "region"].edge_index, data["region", "adj", "region"].edge_attr = ei, ea
    return data
