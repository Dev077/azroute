"""Routing state, kick moves and exact delay for the GAT + MCTS router.

The search improves a complete legal routing. It starts from the negotiated
router's solution after the cleanup pass (a local optimum of single-net
rerouting), and every move is a *kick*: re-route one net through other nets'
wires at a crossing penalty (one of ``KICKS``), rip up the nets it crosses,
re-route them, then let the nets next to the freed space take shorter routes
(``settle``). A kick may make the board worse; the search keeps the best board
it reaches. Kicks are what get past the cleanup pass: on the hard tier about one
kick in ten, followed by the local settle, beats cleanup's result.

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
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from m3d.model import Instance, NetRoute, Submission

FREE = -1
INF = float("inf")

# Kick strengths: the cost of crossing one vertex of another net's wire when the
# kicked net is re-routed. Low = take a shorter route and push many nets aside;
# high = only cross where it saves a lot. Each finds different improvements.
KICKS = (3.0, 6.0, 12.0)
K = len(KICKS)


@dataclass(frozen=True)
class Candidate:
    """One net's route."""
    net: int                          # net index
    vids: frozenset                   # every vertex of the tree (pins included)
    edges: Tuple[Tuple[int, int], ...]  # (a, b) with a < b
    delay: int                        # exact sum of driver->sink path delays
    vias: int
    regions: Tuple[Tuple[int, int], ...]  # (region index, vertices in it)


Move = Tuple[int, int]   # (net index, kick index into KICKS)


class Problem:
    """Static, per-instance data shared by every state of one episode."""

    def __init__(self, inst: Instance, baseline_total: int,
                 reference: Optional[Submission] = None):
        self.inst = inst
        self.baseline_total = int(baseline_total)
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
        self.windows = [self.window(i) for i in range(self.n)]

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
        self.min_step = lo   # cheapest delay of one in-plane step (for lower bounds)

        # per-net ideal route (only the pins as obstacles) and its delay, the lower bound
        self.ideal: List[Candidate] = []
        for i in range(self.n):
            c = route_spt(self, self.pin_owner, i)
            if c is None:
                raise RuntimeError(f"net index {i} cannot be routed even on an empty grid")
            self.ideal.append(c)
        self.lb = [c.delay for c in self.ideal]
        self.lb_total = sum(self.lb)
        self.mean_lb = self.lb_total / max(1, self.n)

        # the negotiated router's legal solution (optional), as routes
        self.ref: List[Optional[Candidate]] = [None] * self.n
        if reference is not None:
            idx = {net.id: i for i, net in enumerate(inst.nets)}
            for r in reference.routes:
                nid = idx[r.net]
                edges, vids = set(), set(self.net_pins[nid])
                for u, v in r.edges:
                    a, b = vid(*u), vid(*v)
                    edges.add((a, b) if a < b else (b, a))
                    vids.update((a, b))
                self.ref[nid] = _finish(self, nid, edges, vids)

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


def _finish(prob: Problem, nid: int, edges: Set[Tuple[int, int]], vids: Set[int]) -> Candidate:
    delay = tree_delay(prob, prob.net_pins[nid], edges)
    vias = sum(1 for a, b in edges if b - a == prob.wh)
    counts: Dict[int, int] = {}
    for v in vids:
        r = prob.region_of[v]
        counts[r] = counts.get(r, 0) + 1
    return Candidate(nid, frozenset(vids), tuple(sorted(edges)), delay, vias,
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


def route_spt(prob: Problem, owner: Sequence[int], nid: int,
              mult=None, vpen=None, soft=None, bounds=None) -> Optional[Candidate]:
    """Shortest-path tree from the driver (exact best delay when costs are true delays).
    Searched in the net's window, then the whole grid; or only inside ``bounds``."""
    pins = prob.net_pins[nid]
    driver, sinks = pins[0], set(pins[1:])
    tries = [bounds] if bounds is not None else [prob.window(nid), prob.window(nid, True)]
    for b in tries:
        found, prev = _dijkstra(prob, owner, nid, {driver: 0.0}, sinks, b, mult, vpen, soft=soft)
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
    return _finish(prob, nid, edges, vids)


# --------------------------------------------------------------------------- #
# State
# --------------------------------------------------------------------------- #
class State:
    """A routing: complete and legal between moves. Copy-on-kick: ``kick`` returns
    a new state; ``lay``/``rip`` change it in place (cleanup, kicks)."""

    __slots__ = ("owner", "routes", "unrouted", "region_occ", "delay")

    def __init__(self, owner, routes, unrouted, region_occ, delay):
        self.owner: List[int] = owner
        self.routes: Dict[int, Candidate] = routes
        self.unrouted: Set[int] = unrouted
        self.region_occ: List[int] = region_occ
        self.delay: int = delay

    @staticmethod
    def initial(prob: Problem) -> "State":
        return State(list(prob.pin_owner), {}, set(range(prob.n)), list(prob.region_pins), 0)

    def copy(self) -> "State":
        return State(list(self.owner), dict(self.routes), set(self.unrouted),
                     list(self.region_occ), self.delay)

    @property
    def done(self) -> bool:
        return not self.unrouted

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

    def rip(self, prob: Problem, nid: int) -> Candidate:
        cand = self.routes.pop(nid)
        owner, occ, region, pin_owner = self.owner, self.region_occ, prob.region_of, prob.pin_owner
        for v in cand.vids:
            if pin_owner[v] == FREE:
                owner[v] = FREE
                occ[region[v]] -= 1
        self.unrouted.add(nid)
        self.delay -= cand.delay
        return cand

    def detour(self, prob: Problem, nid: int) -> int:
        """Delay above the net's lower bound (0 = it already has its ideal delay)."""
        return self.routes[nid].delay - prob.lb[nid]

    def ideal_hits(self, prob: Problem, nid: int) -> Dict[int, int]:
        """Nets whose wires the net's ideal route runs through -> vertices shared."""
        owner = self.owner
        out: Dict[int, int] = {}
        for v in prob.ideal[nid].vids:
            o = owner[v]
            if o != FREE and o != nid:
                out[o] = out.get(o, 0) + 1
        return out

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
            s.lay(prob, _finish(prob, nid, edges, vids))
        return s


# --------------------------------------------------------------------------- #
# Moves
# --------------------------------------------------------------------------- #
def _sink_delays(prob: Problem, cand: Candidate) -> List[Tuple[int, int]]:
    """(sink, driver->sink delay along the net's tree) for each sink."""
    adj: Dict[int, List[int]] = {}
    for a, b in cand.edges:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    pins = prob.net_pins[cand.net]
    dist = {pins[0]: 0}
    stack = [pins[0]]
    while stack:
        u = stack.pop()
        for v in adj.get(u, ()):
            if v not in dist:
                dist[v] = dist[u] + prob.edge_delay(u, v)
                stack.append(v)
    return [(s, dist[s]) for s in pins[1:]]


def _could_shorten(prob: Problem, cand: Candidate, freed: List[Tuple[int, int, int]]) -> bool:
    """Whether any freed vertex (x, y, z) could lie on a path shorter than the
    net's current one to some sink: a lower bound on delay (cheapest layer per
    step, a via per layer change) through it must beat that sink's delay. A net
    that was already routed as well as it could be can only improve through
    newly freed space, so the others need no Dijkstra."""
    c, via, W, wh = prob.min_step, prob.via, prob.W, prob.wh
    sinks = []
    for s, d in _sink_delays(prob, cand):
        z, r = divmod(s, wh)
        y, x = divmod(r, W)
        sinks.append((x, y, z, d))
    far = max(d for *_xyz, d in sinks)
    z0, r = divmod(prob.net_pins[cand.net][0], wh)
    y0, x0 = divmod(r, W)
    for fx, fy, fz in freed:
        a = c * (abs(fx - x0) + abs(fy - y0)) + via * abs(fz - z0)
        if a >= far:
            continue
        for sx, sy, sz, d in sinks:
            if a + c * (abs(fx - sx) + abs(fy - sy)) + via * abs(fz - sz) < d:
                return True
    return False


def settle(prob: Problem, state: State, freed: Set[int], budget: int) -> int:
    """Let nets next to newly freed vertices take shorter routes, in place.

    Only nets whose search window overlaps the freed space, and which could
    possibly get shorter through it (``_could_shorten``), are retried (worst
    detour first); a net that improves frees space of its own, which is handled
    the same way. At most ``budget`` reroutes are tried. Returns how many."""
    tries = 0
    xs, ys, wins, wh = prob.xs, prob.ys, prob.windows, prob.wh
    while freed and tries < budget:
        fx0 = min(xs[v] for v in freed); fx1 = max(xs[v] for v in freed)
        fy0 = min(ys[v] for v in freed); fy1 = max(ys[v] for v in freed)
        nets = [i for i, r in state.routes.items() if r.delay > prob.lb[i]
                and not (wins[i][1] < fx0 or wins[i][0] > fx1 or wins[i][3] < fy0 or wins[i][2] > fy1)]
        nets.sort(key=lambda i: (prob.lb[i] - state.routes[i].delay, i))
        pts = [(xs[v], ys[v], v // wh) for v in freed]
        freed = set()
        for nid in nets:
            if tries >= budget:
                break
            if not _could_shorten(prob, state.routes[nid], pts):
                continue
            tries += 1
            cur = state.routes[nid]
            # a box around the pins and the current route: the current route
            # still fits, so the reroute is never worse, and the search stays small
            x0, x1, y0, y1, _, _ = prob.bbox[nid]
            rx = [xs[v] for v in cur.vids]; ry = [ys[v] for v in cur.vids]
            box = (max(0, min(x0, min(rx)) - 2), min(prob.W - 1, max(x1, max(rx)) + 2),
                   max(0, min(y0, min(ry)) - 2), min(prob.H - 1, max(y1, max(ry)) + 2))
            state.rip(prob, nid)
            c = route_spt(prob, state.owner, nid, bounds=box)
            if c is not None and c.delay < cur.delay:
                state.lay(prob, c)
                freed |= cur.vids - c.vids
            else:
                state.lay(prob, cur)
    return tries


def kick(prob: Problem, state: State, move: Move, settle_budget: int) -> Optional[State]:
    """Apply a kick to a copy of ``state``: re-route the net crossing other nets'
    wires at ``KICKS[k]`` per vertex, rip up the nets it crosses and re-route them
    (largest first), then ``settle``. Returns None if a crossed net can no longer
    be routed, or if the kick would change nothing."""
    nid, k = move
    pen = KICKS[k]
    s = state.copy()
    old = [s.rip(prob, nid)]
    r = route_spt(prob, s.owner, nid, soft=lambda o: pen)
    if r is None:  # cannot happen: its own old route is still free
        return None
    owner = s.owner
    hit = {owner[v] for v in r.vids if owner[v] != FREE and owner[v] != nid}
    if not hit and r.edges == old[0].edges:
        return None
    for b in hit:
        old.append(s.rip(prob, b))
    s.lay(prob, r)
    for b in sorted(hit, key=lambda i: prob.rank[i]):
        c = route_spt(prob, s.owner, b)
        if c is None:
            return None
        s.lay(prob, c)
    freed = {v for o in old for v in o.vids if owner[v] == FREE}
    settle(prob, s, freed, settle_budget)
    return s


class Board:
    """One search problem: a starting routing, the nets that may be kicked (all
    of them, or a window of a large board), and the move budget of an episode."""

    def __init__(self, prob: Problem, start: State, moves: int, settle_budget: int = 40,
                 nets: Optional[Sequence[int]] = None):
        assert start.done
        self.prob = prob
        self.start = start
        self.moves = moves
        self.settle_budget = settle_budget
        self.nets = sorted(nets) if nets is not None else list(range(prob.n))

    def actions(self, state: State) -> List[Move]:
        """Every kick of every kickable net that isn't already at its lower bound."""
        prob = self.prob
        return [(nid, k) for nid in self.nets if state.routes[nid].delay > prob.lb[nid]
                for k in range(K)]

    def apply(self, state: State, move: Move) -> Optional[State]:
        return kick(self.prob, state, move, self.settle_budget)


def window_nets(prob: Problem, size: int, rng) -> List[int]:
    """``size`` nets clustered around a random one: the kickable nets of a large board."""
    def centre(i):
        x0, x1, y0, y1, _, _ = prob.bbox[i]
        return (x0 + x1) / 2, (y0 + y1) / 2
    cx, cy = centre(rng.randrange(prob.n))
    near = sorted(range(prob.n), key=lambda i: ((centre(i)[0] - cx) ** 2 + (centre(i)[1] - cy) ** 2, i))
    return near[:size]


def nets_around(prob: Problem, nets: Sequence[int]) -> List[int]:
    """``nets`` plus every net whose search window overlaps theirs: the nets a kick
    in ``nets`` (and its settle) can move."""
    w = prob.windows
    x0 = min(w[i][0] for i in nets); x1 = max(w[i][1] for i in nets)
    y0 = min(w[i][2] for i in nets); y1 = max(w[i][3] for i in nets)
    return [i for i in range(prob.n)
            if not (w[i][1] < x0 or w[i][0] > x1 or w[i][3] < y0 or w[i][2] > y1)]


def negotiate(inst: Instance, order: str = "bbox_desc") -> Tuple[Optional[Submission], Dict[int, float]]:
    """Run the toolkit's negotiated-congestion router; return its legal solution
    (None if it fails) and its per-vertex history costs (vid -> history)."""
    from m3d.negotiated import Negotiated
    neg = Negotiated(inst, order=order)
    sub, _stats = neg.run()
    return sub, {v: h for v, h in neg.h_v.items() if h > 0}
