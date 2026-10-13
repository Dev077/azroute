"""AlphaZero-style MCTS over kick moves on a complete routing.

The objective of an episode is the best board it reaches, so the return of a
path is the highest score along it (each node's score is exact: every board in
the tree is a real, legal routing), or at the leaf the value head's estimate of
the best still to come. Single-player search, so Q values are min-max
normalised over the tree before PUCT (as in MuZero). A kick that cannot be
applied (a crossed net has nowhere left to go, or nothing would change) is
masked out the first time the search tries it.

``evaluator="heuristic"`` runs the same search without a network: priors favour
the nets with the largest detour, and a leaf is worth its own score. It is the
baseline the trained network is compared against.
"""
from __future__ import annotations

import math
import random
from typing import List, Optional

import numpy as np

from .graph import build_graph
from .problem import Board, Move, State


class Node:
    __slots__ = ("state", "depth", "score", "best_delay", "actions", "prior", "children", "dead",
                 "N", "W", "expanded", "terminal", "value", "graph")

    def __init__(self, state: State, depth: int, score: float, best_delay: int):
        self.state = state
        self.depth = depth                 # moves played since the episode began
        self.score = score                 # this board's challenge score (exact)
        self.best_delay = best_delay       # best board's delay on the way here
        self.actions: List[Move] = []
        self.prior: List[float] = []
        self.children: List[Optional["Node"]] = []
        self.dead: List[bool] = []         # kick could not be applied
        self.N = 0
        self.W = 0.0
        self.expanded = False
        self.terminal = False
        self.value = score                 # estimate of the best score reachable from here
        self.graph = None

    def q(self) -> float:
        return self.W / self.N if self.N else self.value


class MCTS:
    def __init__(self, board: Board, model=None, device="cpu", c_puct: float = 1.5,
                 evaluator: str = "net",
                 dirichlet_alpha: float = 0.3, noise_frac: float = 0.25,
                 rng: Optional[random.Random] = None):
        assert evaluator in ("net", "heuristic")
        assert evaluator == "heuristic" or model is not None
        self.board = board
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

    def root(self, state: State) -> Node:
        return Node(state, 0, self.board.prob.score(state.delay), state.delay)

    # -- expansion -------------------------------------------------------------
    def expand(self, node: Node) -> float:
        board, s = self.board, node.state
        node.expanded = True
        actions = board.actions(s) if node.depth < board.moves else []
        if not actions:
            node.terminal = True
            node.value = node.score
            return node.value
        node.actions = actions
        node.children = [None] * len(actions)
        node.dead = [False] * len(actions)
        if self.evaluator == "net":
            node.graph = build_graph(board, s, actions, node.depth, node.best_delay)
            prior, gain = self.model.evaluate(node.graph, self.device)
            node.prior = prior
            node.value = node.score * (1.0 + max(0.0, gain) / 100.0)
        else:
            prob = board.prob
            ws = [1.0 + 4.0 * (s.routes[nid].delay / prob.lb[nid] - 1.0) for nid, _k in actions]
            z = sum(ws)
            node.prior = [w / z for w in ws]
            node.value = node.score
        return node.value

    def child(self, node: Node, i: int) -> Optional[Node]:
        """The node after move ``i`` (applied on first use); None if it can't be applied."""
        if node.dead[i]:
            return None
        ch = node.children[i]
        if ch is None:
            s = self.board.apply(node.state, node.actions[i])
            if s is None:
                node.dead[i] = True
                node.prior[i] = 0.0
                return None
            ch = Node(s, node.depth + 1, self.board.prob.score(s.delay),
                      min(node.best_delay, s.delay))
            node.children[i] = ch
        return ch

    # -- search ------------------------------------------------------------------
    def add_noise(self, node: Node) -> None:
        live = [i for i in range(len(node.prior)) if not node.dead[i]]
        if len(live) < 2:
            return
        noise = np.random.default_rng(self.rng.randrange(1 << 30)).dirichlet([self.alpha] * len(live))
        f = self.noise_frac
        for i, nz in zip(live, noise):
            node.prior[i] = (1 - f) * node.prior[i] + f * float(nz)

    def _norm(self, q: float) -> float:
        if self.qmax > self.qmin:
            return (q - self.qmin) / (self.qmax - self.qmin)
        return 0.5

    def _select(self, node: Node) -> Optional[int]:
        sqrt_n = math.sqrt(max(1, node.N))
        fpu = self._norm(node.q())
        best, best_i = -math.inf, None
        for i, p in enumerate(node.prior):
            if node.dead[i]:
                continue
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
            self._backup([root], self.expand(root))
        if root.terminal:
            return
        for _ in range(sims):
            node, path = root, [root]
            while True:
                if node.terminal:
                    v = node.value
                    break
                i = self._select(node)
                if i is None:            # every move from here failed: nothing left to try
                    node.terminal, node.value = True, node.score
                    v = node.value
                    break
                child = self.child(node, i)
                if child is None:        # masked; pick again at the same node
                    continue
                path.append(child)
                node = child
                if not child.expanded:
                    v = self.expand(child)
                    break
            self._backup(path, v)

    def _backup(self, path: List[Node], v: float) -> None:
        """Each node gets the best score from itself down to the leaf (or the leaf's estimate)."""
        ret = v
        for n in reversed(path):
            ret = max(ret, n.score)
            self.qmin, self.qmax = min(self.qmin, ret), max(self.qmax, ret)
            n.N += 1
            n.W += ret

    # -- acting ------------------------------------------------------------------
    def visit_policy(self, root: Node) -> List[float]:
        visits = [ch.N if ch is not None else 0 for ch in root.children]
        tot = sum(visits)
        return [v / tot for v in visits] if tot else [0.0] * len(visits)

    def choose(self, root: Node, temperature: float) -> Optional[int]:
        """Most-visited move (temperature 0) or one sampled from the visit counts;
        None if no move from the root could be applied."""
        pi = self.visit_policy(root)
        if not any(pi):
            return None
        if temperature <= 1e-6:
            return max(range(len(pi)), key=lambda i: (pi[i], root.prior[i]))
        w = [p ** (1.0 / temperature) for p in pi]
        return self.rng.choices(range(len(pi)), weights=w)[0]

    def step(self, root: Node, i: int) -> Node:
        """Commit move ``i`` (already applied by the search); its subtree is kept."""
        return root.children[i]
