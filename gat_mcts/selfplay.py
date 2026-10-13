"""Play one improvement episode with MCTS: self-play for training, or solving a case.

An episode starts from a legal routing (the negotiated router's, after the
cleanup pass) and plays ``board.moves`` kicks; its result is the best board it
reached. Every recorded state is labelled with the visit policy of its search
and with how much better, in percent, the best board from that point on was.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import List, Optional

import torch

from m3d.checker import check
from m3d.model import Submission
from m3d.negotiated import route_negotiated
from m3d.scorer import score_case

from .cleanup import cleanup
from .graph import build_graph
from .mcts import MCTS
from .problem import Board, Problem, State, window_nets


@dataclass
class Episode:
    start_score: float           # the board the episode began from
    best_state: State            # the best board it reached (legal)
    best_score: float
    moves: int
    seconds: float
    samples: list = field(default_factory=list)   # (HeteroData, policy target, gain target %)

    @property
    def gain(self) -> float:
        """Improvement over the starting board, in percent."""
        return 100.0 * (self.best_score / self.start_score - 1.0)


def negotiated_state(prob: Problem) -> State:
    """The negotiated router's legal solution as a state."""
    if all(r is not None for r in prob.ref):
        state = State.initial(prob)
        for r in prob.ref:
            state.lay(prob, r)
        return state
    sub, _ = route_negotiated(prob.inst)
    if sub is None:
        raise RuntimeError(f"{prob.inst.name}: no legal routing found")
    return State.from_submission(prob, sub)


def start_state(prob: Problem, seconds: float, nets: Optional[List[int]] = None) -> State:
    """The negotiated solution after the cleanup pass (only ``nets`` rerouted, if
    given): the local optimum the kicks start from."""
    state = negotiated_state(prob)
    cleanup(prob, state, seconds=seconds, nets=nets)
    return state


def _gain_targets(traj: List[float], steps: List[int]) -> List[float]:
    """For a state recorded at step t: % by which the best board from t on beats it."""
    best_after = traj[:]
    for t in range(len(traj) - 2, -1, -1):
        best_after[t] = max(traj[t], best_after[t + 1])
    return [100.0 * (best_after[t] / traj[t] - 1.0) for t in steps]


def _check_legal(prob: Problem, state: State) -> float:
    """The toolkit's own checker + scorer, so results are exactly the challenge's numbers."""
    cs = score_case(prob.inst, state.to_submission(prob), prob.baseline_total)
    assert cs.legal and cs.total_delay == state.delay, cs.reasons
    return cs.ratio


def play(board: Board, mcts: MCTS, sims: int, train: bool, temp_moves: float = 0.3,
         verbose: bool = False) -> Episode:
    """Play ``board.moves`` kicks from ``board.start``. With ``train`` the root gets
    Dirichlet noise, moves are sampled from the visit counts for the first
    ``temp_moves`` fraction of the episode (otherwise the most-visited move is
    played), and each root is recorded with its visit policy."""
    t0 = time.time()
    mcts.reset()
    root = mcts.root(board.start)
    best = root
    traj, steps, raw = [root.score], [], []
    temp_until = int(temp_moves * board.moves)
    moves = 0
    while True:
        if not root.expanded:
            mcts.expand(root)
        if root.terminal:
            break
        if train:
            mcts.add_noise(root)
        mcts.search(root, sims)
        i = mcts.choose(root, 1.0 if (train and moves < temp_until) else 0.0)
        if i is None:
            break
        if train:
            g = root.graph if root.graph is not None else build_graph(
                board, root.state, root.actions, root.depth, root.best_delay)
            raw.append((g, mcts.visit_policy(root)))
            steps.append(moves)
        root = mcts.step(root, i)
        moves += 1
        traj.append(root.score)
        if root.score > best.score:
            best = root
        if verbose:
            print(f"    move {moves}/{board.moves}  score {root.score:.4f}  best {best.score:.4f}", flush=True)
    score = _check_legal(board.prob, best.state)
    samples = [(g, pi, z) for (g, pi), z in zip(raw, _gain_targets(traj, steps))]
    return Episode(traj[0], best.state, score, moves, time.time() - t0, samples)


def label_samples(ep: Episode):
    """Attach the training targets (policy, gain %) to each graph."""
    out = []
    for g, pi, z in ep.samples:
        g = g.clone()
        g["cand"].pi = torch.tensor(pi, dtype=torch.float32)
        g.z = torch.tensor([z], dtype=torch.float32)
        out.append(g)
    return out


@dataclass
class Solution:
    submission: Submission
    delay_start: int      # negotiated router + cleanup: what the search starts from
    delay_search: int     # best board the search reached
    delay_final: int      # after a last cleanup pass (what is submitted)
    rounds: int
    seconds: float


def solve(prob: Problem, model, device: str, sims: int, moves: int, rounds: int,
          c_puct: float = 1.5, evaluator: str = "net", settle_budget: int = 40,
          window: int = 32, full_max_nets: int = 110, start_seconds: float = 120.0,
          cleanup_seconds: float = 30.0, seed: int = 0, verbose: bool = False) -> Solution:
    """Negotiated solution + cleanup, then up to ``rounds`` greedy MCTS episodes of
    ``moves`` kicks, each starting from the best board so far, then cleanup. The
    result is never worse than negotiated + cleanup. A board with more than
    ``full_max_nets`` nets kicks a fresh ``window`` of nets each round."""
    t0 = time.time()
    rng = random.Random(seed)
    best = start_state(prob, start_seconds)
    delay_start = best.delay
    played = 0
    for _ in range(max(1, rounds)):
        nets = window_nets(prob, window, rng) if window and prob.n > full_max_nets else None
        board = Board(prob, best, moves, settle_budget, nets)
        mcts = MCTS(board, model, device, c_puct=c_puct, evaluator=evaluator, rng=rng)
        ep = play(board, mcts, sims, train=False, verbose=verbose)
        played += 1
        if ep.best_state.delay < best.delay:
            best = ep.best_state
        elif nets is None:
            break  # a whole round without a better board: the next would replay it
    delay_search = best.delay
    final = best.copy()
    if cleanup_seconds > 0:
        cleanup(prob, final, seconds=cleanup_seconds, verbose=verbose)
    sub = final.to_submission(prob)
    res = check(prob.inst, sub)
    assert res.legal and res.total_delay == final.delay, res.reasons
    return Solution(sub, delay_start, delay_search, final.delay, played, time.time() - t0)
