"""Improvement loop: repeatedly rip up a neighbourhood of nets and reroute it.

A neighbourhood is a badly detoured seed net plus the nets that block its ideal
tree (and, one level further, the nets blocking those). It is rerouted inside a
bounded region, one routing decision (net, mode) at a time. The reroute is accepted
when the neighbourhood's total delay does not increase; otherwise the previous routes
are restored. The routing is legal at every point, and every accepted reroute is
logged (including any price canvas used) so it can be replayed exactly.
"""
from __future__ import annotations

import base64
import random
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from routing.modes import KEEP, MODE_NAMES
from routing.problem import Box, Problem

Decision = Tuple[int, str]  # (net index, mode name)


@dataclass
class Neighbourhood:
    seed: int
    nets: List[int]
    region: Box
    cache: Dict = field(default_factory=dict, repr=False)


@dataclass
class RerouteState:
    """A partially rerouted neighbourhood."""
    problem: Problem
    hood: Neighbourhood
    old_delay: int                    # neighbourhood delay before the rip-up
    pending: List[int]                # nets still to route
    done: List[Decision] = field(default_factory=list)
    rerouted_delay: int = 0           # exact delay of the nets already rerouted
    blocked: bool = False
    old_net_delay: Dict[int, int] = field(default_factory=dict)
    old_edges: Dict[int, list] = field(default_factory=dict)
    prices: List[Optional[str]] = field(default_factory=list)  # per decision: packed price canvas or None

    @property
    def ideal_pending(self) -> int:
        return sum(self.problem.ideal[i] for i in self.pending)

    def legal_decisions(self) -> List[Decision]:
        return [(n, m) for n in self.pending for m in MODE_NAMES]

    def clone(self) -> "RerouteState":
        return RerouteState(self.problem, self.hood, self.old_delay, list(self.pending), list(self.done),
                            self.rerouted_delay, self.blocked, self.old_net_delay, self.old_edges,
                            list(self.prices))


def pack_price(crop_u8: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(crop_u8, dtype=np.uint8).tobytes()).decode()


def unpack_price(s: str, L: int) -> np.ndarray:
    from network.features import CROP
    return np.frombuffer(base64.b64decode(s), dtype=np.uint8).reshape(L, CROP, CROP)


class Acceptance:
    """Which finished reroutes to keep, judged on the case's total delay before and after.
    The default keeps a reroute only if the total does not increase."""
    name = "strict"

    def ok(self, before: int, after: int, best: int) -> bool:
        return after <= before

    def tick(self, total: int) -> None:
        """Called after every finished reroute with the (possibly unchanged) total."""


class RecordToRecord(Acceptance):
    """Also keep a worse reroute while the total stays within `tau` of the best total found,
    with `tau` shrinking linearly to 0 over `seconds` (record-to-record travel)."""
    name = "rrt"

    def __init__(self, tau: float, seconds: float):
        self.tau, self.seconds, self.t0 = tau, seconds, time.time()

    def ok(self, before: int, after: int, best: int) -> bool:
        left = max(0.0, 1.0 - (time.time() - self.t0) / self.seconds)
        return after <= before or after <= best * (1.0 + self.tau * left)


class LateAcceptance(Acceptance):
    """Also keep a worse reroute if the new total is no worse than the total `length`
    finished reroutes ago (late acceptance hill climbing)."""
    name = "lahc"

    def __init__(self, length: int):
        self.length, self.hist, self.v = length, [], 0

    def ok(self, before: int, after: int, best: int) -> bool:
        if not self.hist:
            self.hist = [before] * self.length
        return after <= before or after <= self.hist[self.v % self.length]

    def tick(self, total: int) -> None:
        if not self.hist:
            self.hist = [total] * self.length
        self.hist[self.v % self.length] = total
        self.v += 1


class Improver:
    def __init__(self, problem: Problem, rng: random.Random, max_nets: int = 6,
                 seed_margin: int = 3, region_margin: int = 6, accept: Optional[Acceptance] = None):
        self.P = problem
        self.rng = rng
        self.max_nets = max_nets
        self.seed_margin = seed_margin
        self.region_margin = region_margin
        self.fail_count: Dict[int, int] = {}
        self.log: List[dict] = []      # accepted reroutes, for exact replay
        self.accept = accept or Acceptance()
        # the best routing seen: while the current routing is the best, best_snap is None; it is
        # saved only when a worse reroute is kept (never under the default acceptance)
        self.total = problem.router.total_delay()
        self.best_total = self.total
        self.best_snap: Optional[list] = None
        self.best_log_len = 0

    # -- neighbourhood selection ------------------------------------------------
    def excess(self, i: int) -> int:
        return self.P.router.net_delay(i) - self.P.ideal[i]

    def pick_seed(self) -> Optional[int]:
        cand, w = [], []
        for i in range(self.P.N):
            e = self.excess(i)
            if e > 0:
                cand.append(i)
                w.append(e / (1 + self.fail_count.get(i, 0)))
        if not cand:
            return None
        return self.rng.choices(cand, weights=w, k=1)[0]

    def neighbourhood(self, seed: int) -> Neighbourhood:
        P = self.P
        nets = [seed]
        frontier = [seed]
        while frontier and len(nets) < self.max_nets:
            nxt = []
            for n in frontier:
                box = P.grow(P.pin_bbox[n], self.seed_margin)
                bl = P.router.blockers(n, box)
                self.rng.shuffle(bl)
                for b in bl:
                    if b not in nets and len(nets) < self.max_nets:
                        nets.append(b)
                        nxt.append(b)
            frontier = nxt
        region = P.pin_bbox[nets[0]]
        for n in nets[1:]:
            region = P.union(region, P.pin_bbox[n])
        return Neighbourhood(seed, nets, P.grow(region, self.region_margin))

    # -- rerouting ----------------------------------------------------------------
    def begin(self, hood: Neighbourhood) -> RerouteState:
        r = self.P.router
        old_net = {i: r.net_delay(i) for i in hood.nets}
        old_edges = {i: r.route_edges(i) for i in hood.nets}
        for i in hood.nets:
            r.rip_up(i)
        return RerouteState(self.P, hood, sum(old_net.values()), pending=list(hood.nets),
                            old_net_delay=old_net, old_edges=old_edges)

    def apply(self, st: RerouteState, d: Decision, price_crop: Optional[np.ndarray] = None) -> bool:
        """Route one net of the neighbourhood. Falls back to DELAY if the mode is blocked.

        LEARNED_PRICE uses `price_crop` (uint8 canvas, 255 = 1.0) when given, else the
        rule-based bounding-box demand of the other pending nets."""
        net, mode = d
        r = self.P.router
        reserve = [n for n in st.pending if n != net]
        packed = None
        if mode == KEEP:
            ok = r.try_set_route(net, st.old_edges[net])
        else:
            price = None
            if mode == "LEARNED_PRICE":
                if price_crop is not None:
                    from network.features import price_full
                    packed = pack_price(price_crop)
                    price = price_full(self.P, st.hood, price_crop.astype(np.float32) / 255.0)
                else:
                    price = self.demand_price(st, net)
            ok = r.route_net(net, self.P.modes[mode], st.hood.region, price, reserve)
        if not ok and mode != "DELAY":
            ok = r.route_net(net, self.P.modes["DELAY"], st.hood.region, None, reserve)
        st.pending.remove(net)
        st.done.append(d)
        st.prices.append(packed)
        if not ok:
            st.blocked = True
            return False
        st.rerouted_delay += r.net_delay(net)
        return True

    def demand_price(self, st: RerouteState, exclude: int) -> np.ndarray:
        """Rule-based price: bounding-box demand of the other pending nets."""
        P = self.P
        price = np.zeros((P.L, P.H, P.W), dtype=np.float32)
        for n in st.pending:
            if n == exclude:
                continue
            x0, y0, x1, y1 = P.pin_bbox[n]
            area = (x1 - x0 + 1) * (y1 - y0 + 1)
            price[:, y0:y1 + 1, x0:x1 + 1] += min(1.0, 4.0 / area)
        return price.reshape(-1)

    def finish(self, st: RerouteState, cp: int) -> Tuple[bool, int]:
        """Accept (commit + log) or reject (rollback) a finished reroute. Returns (accepted, new delay)."""
        r = self.P.router
        hood = st.hood
        if st.blocked or st.pending:
            r.rollback(cp)
            self.fail_count[hood.seed] = self.fail_count.get(hood.seed, 0) + 1
            self.accept.tick(self.total)
            return False, -1
        new = st.rerouted_delay
        after = self.total + new - st.old_delay
        if self.accept.ok(self.total, after, self.best_total):
            if after > self.best_total and self.best_snap is None:
                # leaving the best routing: save it (the current routes with the old ones of this neighbourhood)
                snap = self.P.snapshot()
                for n in hood.nets:
                    snap[n] = list(st.old_edges[n])
                self.best_snap = snap
            r.commit()
            if new < st.old_delay:
                self.fail_count.pop(hood.seed, None)
            self.log.append({"nets": hood.nets, "region": list(hood.region),
                             "decisions": [[n, m] for n, m in st.done], "prices": st.prices})
            self.total = after
            if after <= self.best_total:
                self.best_total, self.best_snap, self.best_log_len = after, None, len(self.log)
            self.accept.tick(self.total)
            return True, new
        r.rollback(cp)
        self.fail_count[hood.seed] = self.fail_count.get(hood.seed, 0) + 1
        self.accept.tick(self.total)
        return False, new

    def restore_best(self) -> None:
        """Go back to the best routing seen, and drop the logged reroutes made after it."""
        if self.best_snap is not None:
            self.P.restore(self.best_snap)
            self.total, self.best_snap = self.best_total, None
            del self.log[self.best_log_len:]

    def reroute(self, hood: Neighbourhood, policy: "Policy") -> Tuple[bool, int, int]:
        """Rip up and reroute a neighbourhood with a synchronous policy. Returns (accepted, old, new)."""
        cp = self.P.router.checkpoint()
        st = self.begin(hood)
        policy(self, st)
        ok, new = self.finish(st, cp)
        return ok, st.old_delay, new


Policy = Callable[[Improver, RerouteState], None]


def greedy_policy(imp: Improver, st: RerouteState) -> None:
    """Rule-based policy: random net order, DELAY mode for every net."""
    order = list(st.pending)
    imp.rng.shuffle(order)
    for n in order:
        if not imp.apply(st, (n, "DELAY")):
            break


def greedy_until_stall(imp: Improver, stall_tries: int, max_seconds: float, k: int = 3) -> int:
    """Fast small-neighbourhood rerouting until `stall_tries` reroutes in a row bring no gain."""
    saved = imp.max_nets
    imp.max_nets = k
    t0, since, gained = time.time(), 0, 0
    while since < stall_tries and time.time() - t0 < max_seconds:
        seed = imp.pick_seed()
        if seed is None:
            break
        ok, old, new = imp.reroute(imp.neighbourhood(seed), greedy_policy)
        if ok and new < old:
            gained += old - new
            since = 0
        else:
            since += 1
    imp.max_nets = saved
    return gained


def replay(problem: Problem, log: Sequence[dict]) -> None:
    """Re-apply logged reroutes on a problem in its initial routing; reproduces the routes exactly."""
    imp = Improver(problem, random.Random(0))
    for entry in log:
        hood = Neighbourhood(entry["nets"][0], list(entry["nets"]), tuple(entry["region"]))
        st = imp.begin(hood)
        prices = entry.get("prices") or [None] * len(entry["decisions"])
        for (net, mode), pr in zip(entry["decisions"], prices):
            crop = unpack_price(pr, problem.L) if pr else None
            if not imp.apply(st, (net, mode), crop):
                raise RuntimeError("replay diverged: decision blocked")
        problem.router.commit()
