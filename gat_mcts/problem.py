"""Routing state, candidate routes and exact delay for the GAT + MCTS router.

Vertex ids follow ``m3d.grid.Grid``: ``vid = (z*height + y)*width + x``.

Ownership is tracked per vertex only. Every pin vertex is owned by its net from
the start, so "never pass through another net's pin" falls out of the same
ownership check, and a route only ever uses edges between vertices it owns, so
exclusive vertices also mean exclusive edges.

Nets are referred to by their *index* in ``inst.nets`` (0..n-1); the instance's
own net ids are only used when writing a submission.
"""
from __future__ import annotations

import heapq
import math
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from m3d.model import Instance, NetRoute, Submission

FREE = -1
INF = float("inf")

# Candidate-route profiles. Each unrouted net in the frontier gets (up to) one
# candidate per profile; identical trees are deduplicated.
#   ref        this net's route in the negotiated router's legal solution (offered
#              while still free: choosing ref for every net reproduces the baseline)
#   spt        exact shortest-path tree: the best delay this net can get right now
#   spt_hist   SPT that avoids what the negotiated router found contended (its history costs)
#   spt_cong   SPT that avoids congested / in-demand regions and other nets' pin access
#   spt_outer  SPT that prefers the expensive outer layers (leaves the cheap middle free)
#   steiner    tree grown sink by sink minimising wirelength (fewest resources)
#   spt_noise  SPT under randomly perturbed region costs (diversity)
PROFILES = ("ref", "spt", "spt_hist", "spt_cong", "spt_outer", "steiner", "spt_noise")
K = len(PROFILES)

CONG_GAMMA = 2.0       # spt_cong: cost multiplier per unit of congestion
PIN_ACCESS_PEN = 4.0   # spt_cong: additive cost for hugging another unrouted net's pin
OUTER_GAMMA = 2.5      # spt_outer: extra multiplier on the cheapest layer
REPAIR_PEN = 20.0      # repair: cost of crossing another net's wire (x its rip count + 1)
HIST_W = 2.0           # spt_hist: cost per unit of negotiated history on a vertex


@dataclass(frozen=True)
class Candidate:
    net: int                          # net index
    profile: int                      # index into PROFILES
    vids: frozenset                   # every vertex of the tree (pins included)
    edges: Tuple[Tuple[int, int], ...]  # (a, b) with a < b
    delay: int                        # exact sum of driver->sink path delays
    vias: int
    regions: Tuple[Tuple[int, int], ...]  # (region index, vertices in it)


class Problem:
    """Static, per-instance data shared by every state of one episode."""

    def __init__(self, inst: Instance, baseline_total: int,
                 reference: Optional[Submission] = None,
                 history: Optional[Dict[int, float]] = None,
                 frontier: int = 4, rip_limit: int = 8, seed: int = 0):
        self.inst = inst
        self.baseline_total = int(baseline_total)
        self.frontier_size = frontier
        self.rip_limit = rip_limit
        self.seed = seed
        W, H, L = inst.width, inst.height, inst.layers
        self.W, self.H, self.L = W, H, L
        self.wh = W * H
        self.V = W * H * L
        self.layer_delay = list(inst.layer_delay)
        self.via = inst.via_delay
        self.n = len(inst.nets)
        self.margin = max(4, int(0.15 * max(W, H)))

        # pins
        def vid(x, y, z):
            return (z * H + y) * W + x
        self.vid = vid
        self._build_tables()
        pin_vid = {p.id: vid(p.x, p.y, p.z) for p in inst.pins}
        self.net_pins: List[List[int]] = [[pin_vid[p] for p in net.pins()] for net in inst.nets]
        self.pin_owner = [FREE] * self.V
        for i, pins in enumerate(self.net_pins):
            for v in pins:
                self.pin_owner[v] = i
        # vertices next to another net's pin: vid -> nets whose pin is adjacent
        self.pin_adj: Dict[int, Set[int]] = {}
        for i, pins in enumerate(self.net_pins):
            for p in pins:
                for q, _w in self.neighbors(p):
                    if self.pin_owner[q] == FREE:
                        self.pin_adj.setdefault(q, set()).add(i)

        # net geometry + static priority (bbox span incl. z, largest first, like
        # the reference routers)
        self.bbox = []
        for pins in self.net_pins:
            cs = [self.coord(v) for v in pins]
            xs = [c[0] for c in cs]; ys = [c[1] for c in cs]; zs = [c[2] for c in cs]
            self.bbox.append((min(xs), max(xs), min(ys), max(ys), min(zs), max(zs)))
        span = [(b[1] - b[0]) + (b[3] - b[2]) + (b[5] - b[4]) for b in self.bbox]
        self.order = sorted(range(self.n), key=lambda i: (-span[i], i))
        self.rank = {nid: r for r, nid in enumerate(self.order)}
        self.cross = [1.0 if len({self.coord(v)[2] for v in pins}) > 1 else 0.0
                      for pins in self.net_pins]

        # regions: T x T tiles per layer, ~10 tiles a side
        T = max(2, math.ceil(max(W, H) / 10))
        self.tile = T
        self.TX, self.TY = math.ceil(W / T), math.ceil(H / T)
        self.R = self.TX * self.TY * L
        self.region_of = [0] * self.V
        self.region_size = [0] * self.R
        for z in range(L):
            for y in range(H):
                for x in range(W):
                    r = (z * self.TY + y // T) * self.TX + x // T
                    self.region_of[vid(x, y, z)] = r
                    self.region_size[r] += 1
        self.region_pins = [0] * self.R
        for pins in self.net_pins:
            for v in pins:
                self.region_pins[self.region_of[v]] += 1
        lo, hi = min(self.layer_delay), max(self.layer_delay)
        self.cheapness = [((hi - d) / (hi - lo) if hi > lo else 0.0) for d in self.layer_delay]
        self.outer_mult = [1.0 + OUTER_GAMMA * self.cheapness[self.region_z(r)]
                           for r in range(self.R)]

        # per-net lower bound: best delay with only the pins as obstacles
        self.lb = []
        for i in range(self.n):
            c = route_spt(self, self.pin_owner, i, profile=0)
            if c is None:
                raise RuntimeError(f"net index {i} cannot be routed even on an empty grid")
            self.lb.append(c.delay)
        self.lb_total = sum(self.lb)
        self.mean_lb = self.lb_total / max(1, self.n)

        # negotiated-router knowledge (both optional)
        self.history = history or {}
        self.ref: List[Optional[Candidate]] = [None] * self.n
        if reference is not None:
            idx = {net.id: i for i, net in enumerate(inst.nets)}
            p_ref = PROFILES.index("ref")
            for r in reference.routes:
                nid = idx[r.net]
                edges, vids = set(), set(self.net_pins[nid])
                for u, v in r.edges:
                    a, b = vid(*u), vid(*v)
                    edges.add((a, b) if a < b else (b, a))
                    vids.update((a, b))
                self.ref[nid] = _finish(self, nid, p_ref, edges, vids)

    # -- geometry -------------------------------------------------------------
    def coord(self, v: int) -> Tuple[int, int, int]:
        z, r = divmod(v, self.wh)
        y, x = divmod(r, self.W)
        return x, y, z

    def region_z(self, r: int) -> int:
        return r // (self.TX * self.TY)

    def neighbors(self, v: int):
        """``(neighbour vid, edge delay)`` for every legal move from ``v``."""
        return self.nbrs[v]

    def _build_tables(self) -> None:
        W, H, L, wh = self.W, self.H, self.L, self.wh
        self.xs = [0] * self.V
        self.ys = [0] * self.V
        self.nbrs: List[Tuple[Tuple[int, int], ...]] = [()] * self.V
        for z in range(L):
            ld = self.layer_delay[z]
            for y in range(H):
                for x in range(W):
                    v = (z * H + y) * W + x
                    self.xs[v], self.ys[v] = x, y
                    out = []
                    if x + 1 < W: out.append((v + 1, ld))
                    if x > 0: out.append((v - 1, ld))
                    if y + 1 < H: out.append((v + W, ld))
                    if y > 0: out.append((v - W, ld))
                    if z + 1 < L: out.append((v + wh, self.via))
                    if z > 0: out.append((v - wh, self.via))
                    self.nbrs[v] = tuple(out)

    def edge_delay(self, a: int, b: int) -> int:
        za, zb = a // self.wh, b // self.wh
        return self.via if za != zb else self.layer_delay[za]

    def window(self, nid: int, full: bool = False):
        if full:
            return 0, self.W - 1, 0, self.H - 1
        x0, x1, y0, y1, _, _ = self.bbox[nid]
        m = self.margin
        return max(0, x0 - m), min(self.W - 1, x1 + m), max(0, y0 - m), min(self.H - 1, y1 + m)

    def score(self, total_delay: int) -> float:
        """The challenge's per-case score: baseline_total / total_delay."""
        return self.baseline_total / total_delay


# --------------------------------------------------------------------------- #
# Shortest-path search
# --------------------------------------------------------------------------- #
def _dijkstra(prob: Problem, owner: Sequence[int], nid: int, seeds: Dict[int, float],
              targets: Set[int], bounds, mult=None, vpen=None, blocked=None,
              first_only: bool = False, soft=None):
    """Dijkstra over vertices free or owned by ``nid``, inside ``bounds`` (x0, x1, y0, y1).

    ``mult`` is a per-region multiplier on edge delay, ``vpen`` an additive
    per-vertex cost. With ``soft`` (owner net -> extra cost), other nets' wires
    may be crossed at that cost; other nets' pins never can. Returns
    ``(found_targets, prev)``; ``prev[v]`` is -1 for seeds and unreached vertices.
    """
    pin_owner, NB, X, Y, region = prob.pin_owner, prob.nbrs, prob.xs, prob.ys, prob.region_of
    x0, x1, y0, y1 = bounds
    clip = not (x0 == 0 and y0 == 0 and x1 == prob.W - 1 and y1 == prob.H - 1)
    V = prob.V
    dist = [INF] * V
    prev = [-1] * V
    done = bytearray(V)
    heap = []
    for v, d in seeds.items():
        dist[v] = d
        heap.append((d, v))
    heapq.heapify(heap)
    found: List[int] = []
    need = len(targets)
    pop, push = heapq.heappop, heapq.heappush
    while heap:
        d, u = pop(heap)
        if done[u]:
            continue
        done[u] = 1
        if u in targets:
            found.append(u)
            if first_only or len(found) == need:
                break
        for v, w in NB[u]:
            if done[v]:
                continue
            if clip and not (x0 <= X[v] <= x1 and y0 <= Y[v] <= y1):
                continue
            o = owner[v]
            if o != FREE and o != nid:
                if soft is None or pin_owner[v] != FREE:
                    continue
                c = soft(o)
            else:
                c = 0.0
            if blocked is not None and v in blocked:
                continue
            c += w * mult[region[v]] if mult is not None else w
            if vpen is not None:
                c += vpen(v)
            nd = d + c
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                push(heap, (nd, v))
    return found, prev


def _finish(prob: Problem, nid: int, profile: int, edges: Set[Tuple[int, int]],
            vids: Set[int]) -> Candidate:
    delay = tree_delay(prob, prob.net_pins[nid], edges)
    vias = sum(1 for a, b in edges if b - a == prob.wh)
    counts: Dict[int, int] = {}
    for v in vids:
        r = prob.region_of[v]
        counts[r] = counts.get(r, 0) + 1
    return Candidate(nid, profile, frozenset(vids), tuple(sorted(edges)), delay, vias,
                     tuple(sorted(counts.items())))


def tree_delay(prob: Problem, pins: Sequence[int], edges: Iterable[Tuple[int, int]]) -> int:
    """Exact objective for one net: sum over sinks of the driver->sink tree path delay."""
    adj: Dict[int, List[int]] = {}
    for a, b in edges:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    driver = pins[0]
    dist = {driver: 0}
    stack = [driver]
    while stack:
        u = stack.pop()
        for v in adj.get(u, ()):
            if v not in dist:
                dist[v] = dist[u] + prob.edge_delay(u, v)
                stack.append(v)
    return sum(dist[s] for s in pins[1:])


def route_spt(prob: Problem, owner: Sequence[int], nid: int, profile: int = 0,
              mult=None, vpen=None, soft=None) -> Optional[Candidate]:
    """Shortest-path tree from the driver (exact best delay when costs are true delays)."""
    pins = prob.net_pins[nid]
    driver, sinks = pins[0], set(pins[1:])
    for full in (False, True):
        found, prev = _dijkstra(prob, owner, nid, {driver: 0.0}, sinks,
                                prob.window(nid, full), mult, vpen, soft=soft)
        if len(found) == len(sinks):
            break
    else:
        return None
    vids = {driver}
    edges: Set[Tuple[int, int]] = set()
    for t in sinks:
        v = t
        while v not in vids:
            vids.add(v)
            p = prev[v]
            edges.add((p, v) if p < v else (v, p))
            v = p
    return _finish(prob, nid, profile, edges, vids)


def route_grow(prob: Problem, owner: Sequence[int], nid: int, beta: float, profile: int,
               mult=None) -> Optional[Candidate]:
    """Grow a tree sink by sink. Tree vertices seed the search at ``beta`` x their
    delay from the driver: beta=0 is a wirelength-minimising Steiner-style tree,
    beta=1 approaches the shortest-path tree."""
    pins = prob.net_pins[nid]
    driver = pins[0]
    remaining = set(pins[1:])
    dtree = {driver: 0}
    edges: Set[Tuple[int, int]] = set()
    while remaining:
        seeds = {v: beta * d for v, d in dtree.items()}
        for full in (False, True):
            found, prev = _dijkstra(prob, owner, nid, seeds, remaining,
                                    prob.window(nid, full), mult, None,
                                    blocked=dtree, first_only=True)
            if found:
                break
        else:
            return None
        path = [found[0]]
        while prev[path[-1]] != -1:
            path.append(prev[path[-1]])
        path.reverse()  # path[0] is a tree vertex
        for a, b in zip(path, path[1:]):
            dtree[b] = dtree[a] + prob.edge_delay(a, b)
            edges.add((a, b) if a < b else (b, a))
        remaining.difference_update(path)
    return _finish(prob, nid, profile, edges, set(dtree))


# --------------------------------------------------------------------------- #
# State
# --------------------------------------------------------------------------- #
class State:
    """One partial routing. Copy-on-apply: ``apply`` returns a new state."""

    __slots__ = ("owner", "routes", "unrouted", "cache", "region_occ", "delay",
                 "step", "rips", "failed", "_demand")

    def __init__(self, owner, routes, unrouted, cache, region_occ, delay, step,
                 rips=None, failed=False):
        self.owner: List[int] = owner
        self.routes: Dict[int, Candidate] = routes
        self.unrouted: Set[int] = unrouted
        # net -> (valid candidates, profiles already tried)
        self.cache: Dict[int, Tuple[Tuple[Candidate, ...], frozenset]] = cache
        self.region_occ: List[int] = region_occ
        self.delay: int = delay
        self.step: int = step
        self.rips: Dict[int, int] = rips if rips is not None else {}  # repairs per net
        self.failed: bool = failed   # a net needed repair more than rip_limit times
        self._demand = None

    @staticmethod
    def initial(prob: Problem) -> "State":
        occ = list(prob.region_pins)
        return State(list(prob.pin_owner), {}, set(range(prob.n)), {}, occ, 0, 0)

    def copy(self) -> "State":
        return State(list(self.owner), dict(self.routes), set(self.unrouted),
                     dict(self.cache), list(self.region_occ), self.delay, self.step,
                     dict(self.rips), self.failed)

    @property
    def done(self) -> bool:
        return not self.unrouted

    # -- mutation (used in place by cleanup; via apply() by the search) ----------
    def lay(self, prob: Problem, cand: Candidate) -> None:
        nid = cand.net
        owner, occ, region = self.owner, self.region_occ, prob.region_of
        for v in cand.vids:
            if owner[v] == FREE:
                owner[v] = nid
                occ[region[v]] += 1
        self.routes[nid] = cand
        self.unrouted.discard(nid)
        self.delay += cand.delay
        self.cache.pop(nid, None)
        self._demand = None
        # drop other nets' candidates that now collide with this route
        for other, (cands, tried) in list(self.cache.items()):
            keep = tuple(c for c in cands if c.vids.isdisjoint(cand.vids))
            if len(keep) != len(cands):
                self.cache[other] = (keep, frozenset(c.profile for c in keep))

    def rip(self, prob: Problem, nid: int) -> Candidate:
        cand = self.routes.pop(nid)
        owner, occ, region, pin_owner = self.owner, self.region_occ, prob.region_of, prob.pin_owner
        for v in cand.vids:
            if pin_owner[v] == FREE:
                owner[v] = FREE
                occ[region[v]] -= 1
        self.unrouted.add(nid)
        self.delay -= cand.delay
        self.cache.clear()
        self._demand = None
        return cand

    def apply(self, prob: Problem, cand: Candidate) -> "State":
        """Lay ``cand`` on a copy, then repair any net it cut off."""
        s = self.copy()
        s.lay(prob, cand)
        s.step += 1
        s.repair(prob)
        return s

    def repair(self, prob: Problem) -> bool:
        """Rip-up repair, so the episode never dead-ends.

        Any unrouted net that can no longer reach all its pins is routed through
        other nets' wires at a penalty (PathFinder-style: the penalty grows with how
        often the crossed net has already been ripped), so it disturbs as few nets
        as possible; the nets it crosses are ripped up and go back to the unrouted
        pool. A net needing this more than ``prob.rip_limit`` times marks the state
        failed (scored 0, like an illegal submission).
        """
        rips = self.rips

        def soft(o):
            return REPAIR_PEN * (1 + rips.get(o, 0))
        while not self.failed:
            dead = self.dead_nets(prob)
            if not dead:
                return True
            for nid in dead:
                if nid not in self.unrouted:
                    continue
                self.rips[nid] = self.rips.get(nid, 0) + 1
                if self.rips[nid] > prob.rip_limit:
                    self.failed = True
                    break
                route = route_spt(prob, self.owner, nid, soft=soft)
                owner = self.owner
                for b in {owner[v] for v in route.vids if owner[v] not in (FREE, nid)}:
                    self.rips[b] = self.rips.get(b, 0) + 1
                    self.rip(prob, b)
                self.lay(prob, route)
        return False

    # -- search helpers --------------------------------------------------------
    def frontier(self, prob: Problem) -> List[int]:
        """The next few unrouted nets in priority order: the nets MCTS may pick from."""
        out = []
        for nid in prob.order:
            if nid in self.unrouted:
                out.append(nid)
                if len(out) == prob.frontier_size:
                    break
        return out

    def finish_greedy(self, prob: Problem) -> None:
        """Route every remaining net in place, cheaply: its negotiated (ref) route if
        still free, else the history-aware shortest-path tree, repairing as needed.
        Used to score MCTS leaves by rollout."""
        hist = prob.history

        def vpen(v):
            return HIST_W * hist.get(v, 0.0)
        while not self.done and not self.failed:
            nid = next(i for i in prob.order if i in self.unrouted)
            r = prob.ref[nid]
            if r is None or not all(self.owner[v] in (FREE, nid) for v in r.vids):
                r = route_spt(prob, self.owner, nid, vpen=vpen if hist else None)
            if r is None:  # cut off: repair() routes it through other nets' wires
                self.repair(prob)
                continue
            self.lay(prob, r)
            self.repair(prob)

    def actions(self, prob: Problem, rng_seed: int = 0):
        """``(frontier, candidates of all frontier nets)``; candidates is None if a
        frontier net has none (cannot happen after ``repair``)."""
        frontier = self.frontier(prob)
        out: List[Candidate] = []
        for nid in frontier:
            cands = self.candidates(prob, nid, rng_seed=rng_seed)
            if not cands:
                return frontier, None
            out.extend(cands)
        return frontier, out

    def demand(self, prob: Problem) -> List[float]:
        """Per-region future demand from unrouted nets (RUDY: wirelength / bbox area,
        spread uniformly over the bbox, shared over all layers)."""
        if self._demand is not None:
            return self._demand
        T, TX, TY = prob.tile, prob.TX, prob.TY
        diff = [[0.0] * (TX + 1) for _ in range(TY + 1)]
        for nid in self.unrouted:
            x0, x1, y0, y1, _, _ = prob.bbox[nid]
            tx0, tx1, ty0, ty1 = x0 // T, x1 // T, y0 // T, y1 // T
            ntiles = (tx1 - tx0 + 1) * (ty1 - ty0 + 1)
            dens = prob.lb[nid] / (ntiles * T * T * prob.L * max(1, len(prob.net_pins[nid]) - 1))
            diff[ty0][tx0] += dens; diff[ty0][tx1 + 1] -= dens
            diff[ty1 + 1][tx0] -= dens; diff[ty1 + 1][tx1 + 1] += dens
        for ty in range(TY + 1):
            for tx in range(1, TX + 1):
                diff[ty][tx] += diff[ty][tx - 1]
        for ty in range(1, TY + 1):
            for tx in range(TX + 1):
                diff[ty][tx] += diff[ty - 1][tx]
        plane = [diff[ty][tx] for ty in range(TY) for tx in range(TX)]
        self._demand = plane * prob.L
        return self._demand

    def candidates(self, prob: Problem, nid: int, rng_seed: int = 0) -> Tuple[Candidate, ...]:
        """Candidate routes for ``nid`` (cached; only missing profiles are recomputed)."""
        cands, tried = self.cache.get(nid, ((), frozenset()))
        if len(tried) == K:
            return cands
        out = list(cands)
        seen = {c.edges for c in out}
        owner = self.owner
        for p in range(K):
            if p in tried:
                continue
            name = PROFILES[p]
            if name == "ref":
                r = prob.ref[nid]
                if r is not None and all(owner[v] in (FREE, nid) for v in r.vids):
                    if r.edges not in seen:
                        seen.add(r.edges)
                        out.append(r)
                continue
            if name == "spt":
                c = route_spt(prob, owner, nid, p)
            elif name == "spt_hist":
                if not prob.history:
                    continue
                hist = prob.history
                c = route_spt(prob, owner, nid, p,
                              vpen=lambda v: HIST_W * hist.get(v, 0.0))
            elif name == "spt_cong":
                dem, occ, size = self.demand(prob), self.region_occ, prob.region_size
                mult = [1.0 + CONG_GAMMA * (occ[r] / size[r] + dem[r]) for r in range(prob.R)]
                unrouted, pin_adj = self.unrouted, prob.pin_adj

                def vpen(v, _nid=nid):
                    nets = pin_adj.get(v)
                    if nets and any(o != _nid and o in unrouted for o in nets):
                        return PIN_ACCESS_PEN
                    return 0.0
                c = route_spt(prob, owner, nid, p, mult, vpen)
            elif name == "spt_outer":
                c = route_spt(prob, owner, nid, p, prob.outer_mult)
            elif name == "steiner":
                c = route_grow(prob, owner, nid, 0.0, p)
            else:  # spt_noise
                rng = random.Random(hash((prob.seed, rng_seed, nid, self.step)))
                mult = [1.0 + 0.5 * rng.random() for _ in range(prob.R)]
                c = route_spt(prob, owner, nid, p, mult)
            if c is None:
                # unreachable under one cost is unreachable under all
                self.cache[nid] = ((), frozenset(range(K)))
                return ()
            if c.edges not in seen:
                seen.add(c.edges)
                out.append(c)
        out.sort(key=lambda c: c.profile)
        res = tuple(out)
        self.cache[nid] = (res, frozenset(range(K)))
        return res

    def dead_nets(self, prob: Problem) -> List[int]:
        """Unrouted nets whose pins can no longer be connected at all.

        Exact: an unrouted net may only use free vertices and its own pins, so it
        is routable iff all its pins fall in one component of that graph.
        """
        owner, V = self.owner, prob.V
        comp = [-1] * V
        label = 0
        for s in range(V):
            if owner[s] != FREE or comp[s] != -1:
                continue
            comp[s] = label
            stack = [s]
            while stack:
                u = stack.pop()
                for v, _w in prob.neighbors(u):
                    if owner[v] == FREE and comp[v] == -1:
                        comp[v] = label
                        stack.append(v)
            label += 1
        dead = []
        for nid in self.unrouted:
            pins = prob.net_pins[nid]
            pinset = set(pins)
            # union-find over the pins: two pins join through a shared free
            # component or by being adjacent
            parent = {p: p for p in pins}

            def find(a):
                while parent[a] != a:
                    parent[a] = parent[parent[a]]
                    a = parent[a]
                return a
            comp_pin: Dict[int, int] = {}
            for p in pins:
                for q, _w in prob.neighbors(p):
                    if q in pinset:
                        parent[find(p)] = find(q)
                    elif owner[q] == FREE:
                        c = comp[q]
                        if c in comp_pin:
                            parent[find(p)] = find(comp_pin[c])
                        else:
                            comp_pin[c] = p
            if len({find(p) for p in pins}) > 1:
                dead.append(nid)
        return dead

    # -- output ------------------------------------------------------------------
    def to_submission(self, prob: Problem) -> Submission:
        routes = []
        for i, net in enumerate(prob.inst.nets):
            cand = self.routes[i]
            edges = [(prob.coord(a), prob.coord(b)) for a, b in cand.edges]
            routes.append(NetRoute(net=net.id, edges=edges))
        return Submission(instance=prob.inst.name, routes=routes)

    @staticmethod
    def from_submission(prob: Problem, sub: Submission) -> "State":
        """Rebuild a complete state from a legal submission (e.g. a fallback router's)."""
        s = State.initial(prob)
        idx = {net.id: i for i, net in enumerate(prob.inst.nets)}
        for r in sub.routes:
            nid = idx[r.net]
            edges = set()
            vids = set(prob.net_pins[nid])
            for u, v in r.edges:
                a, b = prob.vid(*u), prob.vid(*v)
                edges.add((a, b) if a < b else (b, a))
                vids.update((a, b))
            s.lay(prob, _finish(prob, nid, 0, edges, vids))
        return s


def negotiate(inst: Instance) -> Tuple[Optional[Submission], Dict[int, float]]:
    """Run the toolkit's negotiated-congestion router; return its legal solution
    (None if it fails) and its per-vertex history costs (vid -> history)."""
    from m3d.negotiated import Negotiated
    neg = Negotiated(inst)
    sub, _stats = neg.run()
    return sub, {v: h for v, h in neg.h_v.items() if h > 0}
