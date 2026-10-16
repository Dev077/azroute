"""Network inputs for a partially rerouted neighbourhood.

The neighbourhood's region is cropped from the grid onto a CROP x CROP canvas per layer
(regions larger than CROP are max-pooled by an integer factor `scale`). Each canvas cell
is a bit field:

    bit 0  occupied by a fixed net (outside the neighbourhood)
    bit 1  occupied by a neighbourhood net already rerouted
    bit 2  pin of a fixed net
    bit 3  pin of a pending net
    bit 4  previous route of a pending net (what KEEP would reinstall)
    bit 5  inside the region (valid cell)
    bit 6  training target only: used by a net routed later in this reroute

Each neighbourhood net is a node with NODE_FEATS features and its pin bounding box in
canvas units; GLOBAL_FEATS describe the whole reroute.

Every ordered pair (i, j) of neighbourhood nets has PAIR_FEATS exact conflict features,
each a fraction of net i's ideal tree (its minimum-delay tree inside the region, ignoring
other nets' routes):

    0  cells of i's ideal tree currently occupied by j
    1  cells of i's ideal tree on j's previous route, while j is pending (what KEEP on j brings back)
    2  cells of i's ideal tree in the via columns above j's pins
"""
from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np

from routing.modes import MODE_NAMES

CROP = 64
MAX_NETS = 16
N_MODES = len(MODE_NAMES)
NODE_FEATS = 24
GLOBAL_FEATS = 6
PAIR_FEATS = 3
BIT_FUTURE = 6
MODE_INDEX = {m: i for i, m in enumerate(MODE_NAMES)}


def geometry(hood) -> Tuple[Tuple[int, int, int, int], int, int, int]:
    """(box, scale, canvas height, canvas width) of a neighbourhood's region."""
    x0, y0, x1, y1 = hood.region
    w, h = x1 - x0 + 1, y1 - y0 + 1
    s = max(1, math.ceil(max(w, h) / CROP))
    return (x0, y0, x1, y1), s, math.ceil(h / s), math.ceil(w / s)


def to_canvas(P, hood, vids) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Canvas (z, cy, cx) indices of the given vertex ids that lie inside the region."""
    (x0, y0, x1, y1), s, _, _ = geometry(hood)
    v = np.asarray(vids, dtype=np.int64)
    z, r = np.divmod(v, P.WH)
    y, x = np.divmod(r, P.W)
    keep = (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
    return z[keep], (y[keep] - y0) // s, (x[keep] - x0) // s


def _pool(bits: np.ndarray, s: int) -> np.ndarray:
    if s == 1:
        return bits
    L, h, w = bits.shape
    H, W = math.ceil(h / s) * s, math.ceil(w / s) * s
    pad = np.zeros((L, H, W), dtype=bits.dtype)
    pad[:, :h, :w] = bits
    return np.bitwise_or.reduce(np.bitwise_or.reduce(pad.reshape(L, H // s, s, W // s, s), axis=4), axis=2)


def _hood_cache(P, st) -> Dict:
    """Per-neighbourhood constants, computed once and shared by every state of a reroute."""
    c = st.hood.cache
    if "ready" in c:
        return c
    (x0, y0, x1, y1), s, hh, ww = geometry(st.hood)
    c["old_route"] = {}
    for n in st.hood.nets:
        verts = {a for e in st.old_edges.get(n, []) for a in e} or {P.drivers[n]}
        c["old_route"][n] = to_canvas(P, st.hood, sorted(verts))
    rois, area = {}, {}
    for n in st.hood.nets:
        bx0, by0, bx1, by1 = P.pin_bbox[n]
        rois[n] = ((max(bx0, x0) - x0) // s, (max(by0, y0) - y0) // s,
                   (min(bx1, x1) - x0) // s, (min(by1, y1) - y0) // s)
        area[n] = (bx1 - bx0 + 1) * (by1 - by0 + 1)
    c["roi"], c["area"] = rois, area
    c["ideal_total"] = float(sum(P.ideal[n] for n in st.hood.nets))
    # ideal trees inside the region, and the static pair conflicts derived from them
    nets = st.hood.nets[:MAX_NETS]
    ideal, idx, old, pin_xy = {}, {}, {}, {}
    for n in nets:
        v = np.asarray(P.router.ideal_tree(n, st.hood.region) or [P.drivers[n]], dtype=np.int64)
        ideal[n] = v
        z, r = np.divmod(v, P.WH)
        y, x = np.divmod(r, P.W)
        idx[n] = (z, y - y0, x - x0)
        old[n] = np.unique(np.asarray([a for e in st.old_edges.get(n, []) for a in e] or [P.drivers[n]],
                                      dtype=np.int64))
        pin_xy[n] = np.asarray([v % P.WH for v in P.pin_vids[n]], dtype=np.int64)
    blk_old = np.zeros((MAX_NETS, MAX_NETS), dtype=np.float32)
    pin_col = np.zeros((MAX_NETS, MAX_NETS), dtype=np.float32)
    for i, ni in enumerate(nets):
        L = len(ideal[ni])
        ixy = ideal[ni] % P.WH
        for j, nj in enumerate(nets):
            if i != j:
                blk_old[i, j] = np.isin(ideal[ni], old[nj]).sum() / L
                pin_col[i, j] = np.isin(ixy, pin_xy[nj]).sum() / L
    c["ideal_idx"], c["blk_old"], c["pin_col"] = idx, blk_old, pin_col
    c["ready"] = True
    return c


def encode(P, st) -> Dict[str, np.ndarray]:
    c = _hood_cache(P, st)
    (x0, y0, x1, y1), s, hh, ww = geometry(st.hood)
    own, pin = P.router.crop(x0, y0, x1, y1)
    hood = np.array(st.hood.nets, dtype=np.int32)
    pending = np.array(st.pending, dtype=np.int32)
    in_hood_own = np.isin(own, hood)
    bits = ((own >= 0) & ~in_hood_own).astype(np.uint8)
    bits |= in_hood_own.astype(np.uint8) << 1
    bits |= ((pin >= 0) & ~np.isin(pin, hood)).astype(np.uint8) << 2
    bits |= np.isin(pin, pending).astype(np.uint8) << 3
    bits = _pool(bits, s)
    for n in st.pending:
        z, cy, cx = c["old_route"][n]
        bits[z, cy, cx] |= 1 << 4
    bits |= 1 << 5
    maps = np.zeros((P.L, CROP, CROP), dtype=np.uint8)
    maps[:, :hh, :ww] = bits[:, :CROP, :CROP]

    nodes = np.zeros((MAX_NETS, NODE_FEATS), dtype=np.float32)
    rois = np.zeros((MAX_NETS, 4), dtype=np.int16)
    legal = np.zeros((MAX_NETS, N_MODES), dtype=bool)
    pend = set(st.pending)
    last_mode = {n: m for n, m in st.done}
    ideal_total = c["ideal_total"]
    r = P.router
    for k, n in enumerate(st.hood.nets[:MAX_NETS]):
        bx0, by0, bx1, by1 = c["roi"][n]
        rois[k] = (bx0, by0, bx1, by1)
        ideal = max(1.0, P.ideal[n])
        f = nodes[k]
        f[0] = n in pend
        f[1] = n not in pend
        f[2] = n == st.hood.seed
        f[3] = math.log1p(P.n_sinks[n]) / 3
        f[4] = P.cross_die[n]
        f[5] = P.driver_top[n]
        f[6] = P.sinks_top[n]
        f[7] = (bx1 - bx0 + 1) / CROP
        f[8] = (by1 - by0 + 1) / CROP
        f[9] = (bx0 + bx1) / 2 / CROP
        f[10] = (by0 + by1) / 2 / CROP
        f[11] = P.ideal[n] / max(1.0, ideal_total)
        f[12] = math.log(max(1, st.old_net_delay[n]) / ideal)
        if n not in pend and r.is_routed(n):
            f[13] = math.log(max(1, r.net_delay(n)) / ideal)
            f[14 + MODE_INDEX[last_mode[n]]] = 1.0
        f[21] = min(1.0, 4.0 / c["area"][n])          # bbox demand density
        f[22] = math.log(ideal) / 10
        f[23] = 1.0                                     # real node (not padding)
        if n in pend:
            legal[k] = True
    ideal_pending = sum(P.ideal[n] for n in st.pending)
    glob = np.array([
        math.log(max(1, st.old_delay) / max(1.0, ideal_total)),
        (st.old_delay - ideal_total) / max(1.0, ideal_total),
        (st.rerouted_delay + ideal_pending) / max(1, st.old_delay),
        s / 8.0,
        len(st.pending) / MAX_NETS,
        len(st.hood.nets) / MAX_NETS,
    ], dtype=np.float32)
    # pair conflicts: dynamic occupancy of each net's ideal tree, plus the cached static ones
    nets = st.hood.nets[:MAX_NETS]
    blk_now = np.zeros((MAX_NETS, MAX_NETS), dtype=np.float32)
    pos = {n: j for j, n in enumerate(nets)}
    for i, ni in enumerate(nets):
        z, yy, xx = c["ideal_idx"][ni]
        owners = own[z, yy, xx]
        hit = owners[(owners >= 0) & (owners != ni)]
        if hit.size:
            for o, cnt in zip(*np.unique(hit, return_counts=True)):
                if o in pos:
                    blk_now[i, pos[o]] = cnt / len(owners)
    pend_col = np.array([n in pend for n in nets] + [False] * (MAX_NETS - len(nets)))
    pairs = np.stack([blk_now, c["blk_old"] * pend_col[None, :], c["pin_col"]], -1)
    dmax = float(max(max(P.layer_delay), P.inst.via_delay))
    ld = np.array(list(P.layer_delay) + [P.inst.via_delay], dtype=np.float32) / dmax
    return {"maps": maps, "nodes": nodes, "rois": rois, "glob": glob, "legal": legal, "ld": ld,
            "pairs": pairs.astype(np.float32)}


def decision_index(st) -> np.ndarray:
    """Flat (node * N_MODES + mode) index of every decision in st.legal_decisions() order."""
    pos = {n: k for k, n in enumerate(st.hood.nets)}
    return np.array([pos[n] * N_MODES + MODE_INDEX[m] for n, m in st.legal_decisions()], dtype=np.int64)


def future_bits(P, st_hood, nets_after: List[int]) -> np.ndarray:
    """Canvas mask of cells used by the final routes of `nets_after` (training target, bit 6)."""
    _, s, hh, ww = geometry(st_hood)
    out = np.zeros((P.L, CROP, CROP), dtype=bool)
    for n in nets_after:
        if P.router.is_routed(n):
            z, cy, cx = to_canvas(P, st_hood, P.router.route_vertices(n))
            ok = (cy < CROP) & (cx < CROP)
            out[z[ok], cy[ok], cx[ok]] = True
    return out


def price_full(P, hood, price_crop: np.ndarray) -> np.ndarray:
    """Expand a (L, CROP, CROP) price canvas back to a per-vertex array for the router core."""
    (x0, y0, x1, y1), s, hh, ww = geometry(hood)
    w, h = x1 - x0 + 1, y1 - y0 + 1
    up = price_crop[:, :hh, :ww]
    if s > 1:
        up = np.repeat(np.repeat(up, s, axis=1), s, axis=2)
    full = np.zeros((P.L, P.H, P.W), dtype=np.float32)
    full[:, y0:y1 + 1, x0:x1 + 1] = up[:, :h, :w]
    return full.reshape(-1)
