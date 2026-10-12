"""Play one routing episode with MCTS: self-play for training, or solving a case."""
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
from .mcts import MCTS, Node
from .problem import PROFILES, Problem, State


@dataclass
class Episode:
    state: State                 # final state (complete unless dead)
    legal: bool                  # finished every net without a dead end
    score: float                 # challenge score of the search's own result (0 if dead)
    moves: int
    seconds: float
    samples: list = field(default_factory=list)   # (HeteroData, visit policy) per decision


def play(prob: Problem, mcts: MCTS, sims: int, train: bool, temp_moves: float = 0.3,
         verbose: bool = False) -> Episode:
    """Route the board from empty. With ``train`` the root gets Dirichlet noise and
    moves are sampled from the visit counts for the first ``temp_moves`` fraction of
    the episode; otherwise the most-visited move is played."""
    t0 = time.time()
    mcts.reset()
    root = Node(State.initial(prob))
    samples = []
    moves = 0
    temp_until = int(temp_moves * prob.n)
    while True:
        if not root.expanded:
            mcts.expand(root)
        if root.terminal:
            break
        if len(root.actions) > 1:
            if train:
                mcts.add_noise(root)
            mcts.search(root, sims)
            pi = mcts.visit_policy(root)
            if train and root.graph is not None:
                samples.append((root.graph, pi))
            i = mcts.choose(root, 1.0 if (train and moves < temp_until) else 0.0)
        else:
            i = 0
        root = mcts.step(root, i)
        moves += 1
        if verbose and moves % 20 == 0:
            print(f"    move {moves}/{prob.n}  delay so far {root.state.delay}", flush=True)
    state = root.state
    legal, score = _final_score(prob, state)
    return Episode(state, legal, score, moves, time.time() - t0, samples)


def _final_score(prob: Problem, state: State):
    """(legal, challenge score). Scored by the toolkit's own checker + scorer, so
    training targets are exactly the challenge's numbers; a failed routing is 0."""
    if not state.done or state.failed:
        return False, 0.0
    cs = score_case(prob.inst, state.to_submission(prob), prob.baseline_total)
    assert cs.legal and cs.total_delay == state.delay, cs.reasons
    return True, cs.ratio


def _guide(prob: Problem, frontier, actions) -> int:
    """Warm-start teacher: route the first frontier net with its negotiated (ref)
    route, else the history-aware SPT, else its lowest-delay candidate."""
    first = [i for i, c in enumerate(actions) if c.net == frontier[0]]
    for name in ("ref", "spt_hist"):
        p = PROFILES.index(name)
        for i in first:
            if actions[i].profile == p:
                return i
    return min(first, key=lambda i: actions[i].delay)


def play_guided(prob: Problem, eps: float, rng: random.Random) -> Episode:
    """Warm-start episode, no search: follow the teacher, but with probability
    ``eps`` play a random candidate instead. Policy targets are the teacher's
    move; the value target is the score the perturbed episode actually reached,
    so the value head learns which deviations end in failure and which pay off."""
    t0 = time.time()
    state = State.initial(prob)
    samples = []
    moves = 0
    while not state.done and not state.failed:
        frontier, actions = state.actions(prob, rng_seed=rng.randrange(1 << 30))
        if actions is None:
            break
        g = _guide(prob, frontier, actions)
        if len(actions) > 1:
            pi = [0.0] * len(actions)
            pi[g] = 1.0
            samples.append((build_graph(prob, state, frontier, actions), pi))
        i = g if rng.random() >= eps else rng.randrange(len(actions))
        state = state.apply(prob, actions[i])
        moves += 1
    legal, score = _final_score(prob, state)
    return Episode(state, legal, score, moves, time.time() - t0, samples)


def label_samples(ep: Episode):
    """Attach the training targets (visit policy, final score, legal) to each graph."""
    out = []
    for g, pi in ep.samples:
        g = g.clone()
        g["cand"].pi = torch.tensor(pi, dtype=torch.float32)
        g.z = torch.tensor([ep.score], dtype=torch.float32)
        g.legal = torch.tensor([1.0 if ep.legal else 0.0])
        out.append(g)
    return out


@dataclass
class Solution:
    submission: Submission
    delay_search: Optional[int]   # delay straight out of MCTS (None if it dead-ended)
    delay_final: int
    path: str                     # which result was kept: "mcts" or "negotiated (...)"
    seconds: float


def _negotiated_state(prob: Problem) -> State:
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


def solve(prob: Problem, mcts: MCTS, sims: int, cleanup_seconds: float = 30.0,
          verbose: bool = False) -> Solution:
    """MCTS (greedy) + cleanup, compared with the negotiated solution + cleanup;
    the lower-delay one is returned, so the result is never worse than the
    negotiated router's. ``delay_search`` is the search's own result (None if it
    failed), which is the number that shows whether training works."""
    t0 = time.time()
    ep = play(prob, mcts, sims, train=False, verbose=verbose)
    delay_search = ep.state.delay if ep.legal else None
    neg = _negotiated_state(prob)
    if ep.legal:
        state, path = ep.state, "mcts"
        if cleanup_seconds > 0:
            cleanup(prob, state, seconds=cleanup_seconds / 2, verbose=verbose)
            cleanup(prob, neg, seconds=cleanup_seconds / 2, verbose=verbose)
        if neg.delay < state.delay:
            state, path = neg, "negotiated (beat the search)"
    else:
        state, path = neg, "negotiated (search failed)"
        if cleanup_seconds > 0:
            cleanup(prob, state, seconds=cleanup_seconds, verbose=verbose)
    sub = state.to_submission(prob)
    res = check(prob.inst, sub)
    assert res.legal and res.total_delay == state.delay, res.reasons
    return Solution(sub, delay_search, state.delay, path, time.time() - t0)
