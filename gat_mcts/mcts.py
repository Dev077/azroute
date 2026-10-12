"""AlphaZero-style MCTS over "route net N with candidate C" moves.

Single-player search, so Q values are min-max normalised over the tree before
PUCT (as in MuZero). Leaves are scored by the GAT's value head, or, with
``evaluator="rollout"``, by actually finishing the board with the rip-up router
and scoring it (no network needed: a check of how far the search can go with
an honest, if greedy, look at the outcome).
"""
from __future__ import annotations

import math
import random
from typing import List, Optional

import numpy as np

from .graph import build_graph
from .problem import Candidate, Problem, State


class Node:
    __slots__ = ("state", "actions", "prior", "children", "N", "W", "expanded",
                 "terminal", "value", "graph", "legal_p")

    def __init__(self, state: State):
        self.state = state
        self.actions: List[Candidate] = []
        self.prior: List[float] = []
        self.children: List[Optional["Node"]] = []
        self.N = 0
        self.W = 0.0
        self.expanded = False
        self.terminal = False
        self.value = 0.0      # evaluation at expansion (exact if terminal)
        self.graph = None
        self.legal_p = 1.0

    def q(self) -> float:
        return self.W / self.N if self.N else 0.0


class MCTS:
    def __init__(self, prob: Problem, model=None, device="cpu", c_puct: float = 1.5,
                 evaluator: str = "net",
                 dirichlet_alpha: float = 0.3, noise_frac: float = 0.25,
                 rng: Optional[random.Random] = None):
        assert evaluator in ("net", "rollout")
        assert evaluator == "rollout" or model is not None
        self.prob = prob
        self.model = model
        self.device = device
        self.c_puct = c_puct
        self.evaluator = evaluator
        self.alpha = dirichlet_alpha
        self.noise_frac = noise_frac
        self.rng = rng or random.Random(0)
        self.reset()

    def reset(self) -> None:
        """Forget the Q range (call at the start of every episode)."""
        self.qmin, self.qmax = math.inf, -math.inf

    # -- expansion -------------------------------------------------------------
    def expand(self, node: Node) -> float:
        prob, s = self.prob, node.state
        node.expanded = True
        if s.failed:
            return self._dead(node)
        if s.done:
            node.terminal = True
            node.value = prob.score(s.delay)
            return node.value
        frontier, actions = s.actions(prob, rng_seed=self.rng.randrange(1 << 30))
        if actions is None:  # cannot happen after repair(); kept as a guard
            return self._dead(node)
        node.actions = actions
        node.children = [None] * len(actions)
        if self.evaluator == "net":
            node.graph = build_graph(prob, s, frontier, actions)
            prior, value, legal_p = self.model.evaluate(node.graph, self.device)
            node.prior, node.value, node.legal_p = prior, value, legal_p
        else:
            # heuristic prior: prefer the negotiated route, then less detour
            ws = [math.exp(-4.0 * (c.delay / prob.lb[c.net] - 1.0)) * (3.0 if c.profile == 0 else 1.0)
                  for c in actions]
            z = sum(ws)
            node.prior = [w / z for w in ws]
            node.value = self._rollout(s)
        return node.value

    def _dead(self, node: Node) -> float:
        node.terminal = True
        node.value = 0.0  # a failed routing is illegal, and illegal scores 0 in the challenge
        node.actions, node.children, node.prior = [], [], []
        return 0.0

    def _rollout(self, state: State) -> float:
        s = state.copy()
        s.finish_greedy(self.prob)
        return self.prob.score(s.delay) if s.done and not s.failed else 0.0

    # -- search ------------------------------------------------------------------
    def add_noise(self, node: Node) -> None:
        if len(node.prior) < 2:
            return
        noise = np.random.default_rng(self.rng.randrange(1 << 30)).dirichlet(
            [self.alpha] * len(node.prior))
        f = self.noise_frac
        node.prior = [(1 - f) * p + f * float(n) for p, n in zip(node.prior, noise)]

    def _norm(self, q: float) -> float:
        if self.qmax > self.qmin:
            return (q - self.qmin) / (self.qmax - self.qmin)
        return 0.5

    def _select(self, node: Node) -> int:
        sqrt_n = math.sqrt(max(1, node.N))
        fpu = self._norm(node.q())
        best, best_i = -math.inf, 0
        for i, p in enumerate(node.prior):
            ch = node.children[i]
            if ch is not None and ch.N:
                q, n = self._norm(ch.q()), ch.N
            else:
                q, n = fpu, 0
            u = q + self.c_puct * p * sqrt_n / (1 + n)
            if u > best:
                best, best_i = u, i
        return best_i

    def search(self, root: Node, sims: int) -> None:
        if not root.expanded:
            v = self.expand(root)
            self._backup([root], v)
        if root.terminal:
            return
        for _ in range(sims):
            node, path = root, [root]
            while True:
                if node.terminal:
                    v = node.value
                    break
                i = self._select(node)
                child = node.children[i]
                if child is None:
                    child = Node(node.state.apply(self.prob, node.actions[i]))
                    node.children[i] = child
                path.append(child)
                node = child
                if not child.expanded:
                    v = self.expand(child)
                    break
            self._backup(path, v)

    def _backup(self, path: List[Node], v: float) -> None:
        self.qmin, self.qmax = min(self.qmin, v), max(self.qmax, v)
        for n in path:
            n.N += 1
            n.W += v

    # -- acting ------------------------------------------------------------------
    def visit_policy(self, root: Node) -> List[float]:
        visits = [ch.N if ch is not None else 0 for ch in root.children]
        tot = sum(visits)
        if tot == 0:
            return list(root.prior)
        return [v / tot for v in visits]

    def choose(self, root: Node, temperature: float) -> int:
        pi = self.visit_policy(root)
        if temperature <= 1e-6:
            return max(range(len(pi)), key=lambda i: (pi[i], root.prior[i]))
        w = [p ** (1.0 / temperature) for p in pi]
        return self.rng.choices(range(len(pi)), weights=w)[0]

    def step(self, root: Node, i: int) -> Node:
        """Commit action ``i``; the child becomes the new root (its subtree is kept)."""
        child = root.children[i]
        if child is None:
            child = Node(root.state.apply(self.prob, root.actions[i]))
        return child
