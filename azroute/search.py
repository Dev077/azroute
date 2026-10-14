"""Search over routing decisions inside one neighbourhood reroute.

Gumbel root selection with sequential halving, deterministic selection below the
root, and the improved decision distribution pi' = softmax(logits + sigma(completed Q))
as the training target (Danihelka et al., ICLR 2022). Q values are rescaled to [0, 1]
by their min and max at each node.

Every node is a partially rerouted neighbourhood. Its score q is the share of the
neighbourhood's room for improvement the finished reroute is expected to recover:

    q = (old_delay - estimated_total) / slack,   slack = max(old_delay - ideal, 0.02 * ideal)

clipped to [-1, 1]; a blocked reroute scores -1. The estimated total is exact for the
nets already rerouted plus the evaluator's estimate for the pending ones.

The search is a generator: whenever it needs evaluations it yields a list of requests and
expects one (logits over st.legal_decisions(), q, price canvas or None) per request back.
With `defer=False` a request is the live state (its routes are installed while the evaluator
runs; the network-free evaluator needs that) and the list has one entry. With `defer=True`
(network evaluation) a request is a Leaf whose features were encoded before its routes were
rolled back, so every simulation of one sequential-halving round (one per candidate) goes out
in a single list. The candidates' subtrees are disjoint, so this changes no decision; it
turns ~n_sims round trips per decision into one per round.
"""
from __future__ import annotations

import math
import random
from typing import Callable, Dict, Generator, List, Optional, Tuple

import numpy as np

from .features import MAX_NETS, N_MODES, decision_index, encode, future_bits
from .lns import Decision, Improver, Neighbourhood, RerouteState

C_VISIT = 50.0
C_SCALE = 0.1
BLOCKED_NET_PENALTY = 2.0   # completion evaluator: an unroutable pending net counts as 2x its old delay

EvalResult = Tuple[np.ndarray, float, Optional[np.ndarray]]


class Leaf:
    """A state to evaluate later: `feats` were encoded while its routes were installed."""
    __slots__ = ("st", "feats")

    def __init__(self, st: RerouteState, feats: Dict[str, np.ndarray]):
        self.st, self.feats = st, feats


def slack(st: RerouteState) -> float:
    ideal = sum(st.problem.ideal[i] for i in st.hood.nets)
    return max(st.old_delay - ideal, 0.02 * ideal, 1.0)


def score_total(st: RerouteState, total: float) -> float:
    return max(-1.0, min(1.0, (st.old_delay - total) / slack(st)))


def exact_score(st: RerouteState) -> float:
    return -1.0 if st.blocked else score_total(st, st.rerouted_delay)


def completion_evaluator(imp: Improver) -> Callable[[RerouteState], EvalResult]:
    """Network-free evaluator: uniform prior; value from completing the pending nets (KEEP their
    previous route if its cells are free, else DELAY; largest ideal delay first), then undoing it.
    Unroutable pending nets are charged a penalty so the search keeps a gradient."""
    def evaluate(st: RerouteState) -> EvalResult:
        r = imp.P.router
        cp = r.checkpoint()
        total = st.rerouted_delay
        reserve = list(st.pending)
        for n in sorted(st.pending, key=lambda i: -imp.P.ideal[i]):
            reserve.remove(n)
            if r.try_set_route(n, st.old_edges[n]) or \
                    r.route_net(n, imp.P.modes["DELAY"], st.hood.region, None, reserve):
                total += r.net_delay(n)
            else:
                total += BLOCKED_NET_PENALTY * st.old_net_delay[n]
        r.rollback(cp)
        return np.zeros(len(st.pending) * N_MODES), score_total(st, total), None
    return evaluate


def drive(gen: Generator, evaluate: Callable[[RerouteState], EvalResult]):
    """Run a search generator to completion with a synchronous per-request evaluator."""
    try:
        req = next(gen)
        while True:
            req = gen.send([evaluate(x) for x in req])
    except StopIteration as stop:
        return stop.value


class Node:
    __slots__ = ("actions", "logits", "value", "price", "children", "N", "W", "terminal")

    def __init__(self):
        self.actions: List[Decision] = []
        self.logits: Optional[np.ndarray] = None
        self.value = 0.0
        self.price: Optional[np.ndarray] = None
        self.children: Dict[Decision, "Node"] = {}
        self.N = 0
        self.W = 0.0
        self.terminal = False

    @property
    def expanded(self) -> bool:
        return self.logits is not None or self.terminal

    def q(self) -> float:
        return self.W / self.N if self.N else 0.0


def _softmax(x: np.ndarray) -> np.ndarray:
    z = np.exp(x - x.max())
    return z / z.sum()


def sigma(q01: np.ndarray, max_n: int) -> np.ndarray:
    return (C_VISIT + max_n) * C_SCALE * q01


def minmax(q: np.ndarray) -> np.ndarray:
    lo, hi = q.min(), q.max()
    return (q - lo) / (hi - lo) if hi - lo > 1e-9 else np.zeros_like(q)


def completed_q(node: Node) -> np.ndarray:
    """Q per action; unvisited actions get the prior-weighted mix of the node value and visited Qs."""
    probs = _softmax(node.logits)
    qs = np.empty(len(node.actions))
    visited = np.zeros(len(node.actions), dtype=bool)
    tot_n = 0
    for i, a in enumerate(node.actions):
        c = node.children.get(a)
        if c is not None and c.N > 0:
            qs[i] = c.q(); visited[i] = True; tot_n += c.N
    if visited.any():
        pv = probs[visited]
        mix = (pv * qs[visited]).sum() / max(pv.sum(), 1e-12)
        v_mix = (node.value + tot_n * mix) / (1 + tot_n)
    else:
        v_mix = node.value
    qs[~visited] = v_mix
    return qs


def improved_policy(node: Node) -> np.ndarray:
    max_n = max((c.N for c in node.children.values()), default=0)
    return _softmax(node.logits + sigma(minmax(completed_q(node)), max_n))


class Search:
    """Search-guided neighbourhood reroute. `reroute(hood)` is a generator; its return value
    is (accepted, old_delay, new_delay, training records)."""

    def __init__(self, imp: Improver, n_full: int = 64, n_fast: int = 8, p_full: float = 0.25,
                 max_considered: int = 16, gumbel_scale: float = 1.0, record: bool = True,
                 max_considered_fast: Optional[int] = None,
                 rng: Optional[random.Random] = None, defer: bool = False, explore_fast: bool = True):
        self.imp = imp
        self.defer = defer
        self.explore_fast = explore_fast
        self.n_full, self.n_fast, self.p_full = n_full, n_fast, p_full
        self.m = max_considered
        self.m_fast = max_considered_fast or max_considered
        self.gumbel_scale = gumbel_scale
        self.record = record
        self.rng = rng or random.Random(imp.rng.randrange(2 ** 32))
        self.np_rng = np.random.default_rng(self.rng.randrange(2 ** 32))

    # -- tree mechanics -----------------------------------------------------------
    @staticmethod
    def _set(node: Node, res: EvalResult) -> None:
        logits, v, price = res
        node.logits = np.asarray(logits, dtype=np.float64)
        node.value = float(v)
        node.price = price

    def _request(self, st: RerouteState):
        return Leaf(st, encode(self.imp.P, st)) if self.defer else st

    def _expand(self, node: Node, st: RerouteState):
        """Evaluate a node whose state is installed in the router right now."""
        if st.blocked or not st.pending:
            node.terminal = True
            node.value = exact_score(st)
            return node.value
        node.actions = st.legal_decisions()
        (res,) = yield [self._request(st)]
        self._set(node, res)
        return node.value

    def _apply(self, node: Node, st: RerouteState, a: Decision) -> bool:
        return self.imp.apply(st, a, node.price if a[1] == "LEARNED_PRICE" else None)

    def _simulate(self, root: Node, st: RerouteState, firsts: List[Decision]):
        """One simulation per first decision. Deferred leaves of all of them are evaluated in one
        request; the paths are disjoint below the root, so the order of backups does not matter."""
        r = self.imp.P.router
        pending, done = [], []
        for first in firsts:
            cp = r.checkpoint()
            s = st.clone()
            node, a, path = root, first, []
            while True:
                child = node.children.get(a)
                if child is None:
                    child = node.children[a] = Node()
                self._apply(node, s, a)
                path.append(child)
                if child.terminal:
                    done.append((path, child.value))
                    break
                if not child.expanded:
                    if s.blocked or not s.pending:
                        child.terminal = True
                        child.value = exact_score(s)
                        done.append((path, child.value))
                    elif self.defer:
                        child.actions = s.legal_decisions()
                        pending.append((child, path, Leaf(s, encode(self.imp.P, s))))
                    else:
                        value = yield from self._expand(child, s)
                        done.append((path, value))
                    break
                node = child
                pi = improved_policy(node)
                ns = np.array([node.children[x].N if x in node.children else 0 for x in node.actions])
                a = node.actions[int(np.argmax(pi - ns / (1 + ns.sum())))]
            r.rollback(cp)
        if pending:
            results = yield [leaf for _, _, leaf in pending]
            for (child, path, _), res in zip(pending, results):
                self._set(child, res)
                done.append((path, child.value))
        for path, value in done:
            root.N += 1
            for n in path:
                n.N += 1
                n.W += value

    def _choose(self, root: Node, st: RerouteState, n_sims: int, gscale: float, m: int):
        """Gumbel top-m sampling plus sequential halving; the final pick drops the Gumbel noise
        (the reroute is only kept if it is no worse, so the executed decision should exploit)."""
        k = len(root.actions)
        if k == 1:
            return root.actions[0]
        g = gscale * self.np_rng.gumbel(size=k)
        cand = list(np.argsort(-(g + root.logits))[: min(m, k)])
        phases = max(1, math.ceil(math.log2(len(cand))))
        for _ in range(phases):
            per = max(1, n_sims // (phases * len(cand)))
            for _ in range(per):
                if self.defer:
                    yield from self._simulate(root, st, [root.actions[i] for i in cand])
                else:
                    for i in cand:
                        yield from self._simulate(root, st, [root.actions[i]])
            if len(cand) == 1:
                break
            max_n = max(c.N for c in root.children.values())
            cq = minmax(completed_q(root))
            score = g[cand] + root.logits[cand] + sigma(cq[cand], max_n)
            cand = [cand[j] for j in np.argsort(-score)[: max(1, math.ceil(len(cand) / 2))]]
        max_n = max(c.N for c in root.children.values())
        cq = minmax(completed_q(root))
        final = root.logits + sigma(cq, max_n)
        return root.actions[max(cand, key=lambda i: final[i])]

    # -- one neighbourhood reroute --------------------------------------------------
    def reroute(self, hood: Neighbourhood):
        imp, P = self.imp, self.imp.P
        cp = P.router.checkpoint()
        st = imp.begin(hood)
        root = Node()
        yield from self._expand(root, st)
        pending_records = []
        while st.pending and not st.blocked:
            full = self.rng.random() < self.p_full
            # Gumbel noise picks which decisions are considered and, through the halving, the one
            # executed; a worse reroute is rejected, so exploring on fast searches too is cheap
            explore = (full and self.record) or self.explore_fast
            a = yield from self._choose(root, st, self.n_full if full else self.n_fast,
                                        self.gumbel_scale if explore else 0.0, self.m if full else self.m_fast)
            if full and self.record and root.logits is not None:
                feats = encode(P, st)
                pi = np.zeros(MAX_NETS * N_MODES, dtype=np.float32)
                pi[decision_index(st)] = improved_policy(root)
                # how much the searched decisions' outcomes differed: ~0 means nothing to learn
                qs = [c.q() for c in root.children.values() if c.N > 0]
                spread = max(qs) - min(qs) if len(qs) > 1 else 0.0
                pending_records.append((feats, pi.reshape(MAX_NETS, N_MODES), st.rerouted_delay,
                                        list(st.pending), spread))
            self._apply(root, st, a)
            nxt = root.children.get(a)
            if nxt is None or not nxt.expanded:
                nxt = Node()
                yield from self._expand(nxt, st)
            root = nxt
        records = self._targets(st, pending_records)
        ok, new = imp.finish(st, cp)
        return ok, st.old_delay, new, records

    def _targets(self, st: RerouteState, pending_records) -> List[Dict[str, np.ndarray]]:
        """Attach outcome targets (computed from the final routes, before accept/reject)."""
        P = self.imp.P
        out = []
        final_total = st.rerouted_delay
        net_ex = np.full(MAX_NETS, np.nan, dtype=np.float32)
        if not st.blocked:
            for k, n in enumerate(st.hood.nets[:MAX_NETS]):
                net_ex[k] = math.log(max(1, P.router.net_delay(n)) / max(1, P.ideal[n]))
        for feats, pi, rerouted_t, pending_t, spread in pending_records:
            rec = dict(feats)
            rec["pi"] = pi.astype(np.float16)
            rec["spread"] = np.float32(spread)
            if st.blocked:
                rec["y"] = np.float32(np.nan)
                rec["blocked"] = np.uint8(1)
                rec["fut_valid"] = np.uint8(0)
            else:
                lb = max(1, sum(P.ideal[n] for n in pending_t))
                rec["y"] = np.float32(math.log(max(1, final_total - rerouted_t) / lb))
                rec["blocked"] = np.uint8(0)
                rec["fut_valid"] = np.uint8(1)
                rec["maps"] = rec["maps"] | (future_bits(P, st.hood, pending_t).astype(np.uint8) << 6)
            rec["excess"] = net_ex.astype(np.float16)
            out.append(rec)
        return out
