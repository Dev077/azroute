"""Cleanup pass: rip-up-and-reroute on a finished, legal board.

Runs after the search. It only ever accepts changes that lower total delay, so
it never makes a result worse.
"""
from __future__ import annotations

import time
from typing import List, Optional

from .problem import FREE, Candidate, Problem, State, route_spt


def _ideal(prob: Problem, nid: int) -> Candidate:
    """The net's best route if no other net were routed (pins are still obstacles)."""
    return route_spt(prob, prob.pin_owner, nid)


def _blockers(state: State, cand: Candidate) -> set:
    owner = state.owner
    return {owner[v] for v in cand.vids if owner[v] not in (FREE, cand.net)}


def _try_neighbourhood(prob: Problem, state: State, nets: List[int],
                       first: Optional[Candidate] = None) -> bool:
    """Rip up ``nets`` and reroute them (``first`` is laid before the rest).
    Keeps the result only if total delay drops; otherwise restores the old routes."""
    old = {nid: state.routes[nid] for nid in nets}
    before = state.delay
    for nid in nets:
        state.rip(prob, nid)
    ok = True
    if first is not None:
        state.lay(prob, first)
    for nid in sorted(nets, key=lambda i: prob.rank[i]):
        if nid not in state.unrouted:
            continue
        cand = route_spt(prob, state.owner, nid)
        if cand is None:
            ok = False
            break
        state.lay(prob, cand)
    if ok and state.delay < before:
        return True
    for nid in nets:
        if nid in state.routes:
            state.rip(prob, nid)
    for cand in old.values():
        state.lay(prob, cand)
    return False


def cleanup(prob: Problem, state: State, seconds: float = 30.0,
            max_blockers: int = 6, verbose: bool = False) -> int:
    """Improve a complete legal board in place. Returns the delay saved."""
    assert state.done
    start_delay = state.delay
    t_end = time.time() + seconds
    improved = True
    sweep = 0
    while improved and time.time() < t_end:
        improved = False
        sweep += 1
        # worst detours first
        nets = sorted(range(prob.n), key=lambda i: prob.lb[i] - state.routes[i].delay)
        for nid in nets:
            if time.time() >= t_end:
                break
            cur = state.routes[nid]
            if cur.delay == prob.lb[nid]:
                continue
            # 1) reroute this net alone
            if _try_neighbourhood(prob, state, [nid]):
                improved = True
                continue
            # 2) give it its ideal route and reroute the nets that were in the way
            ideal = _ideal(prob, nid)
            blockers = _blockers(state, ideal)
            if 0 < len(blockers) <= max_blockers:
                if _try_neighbourhood(prob, state, [nid] + sorted(blockers), first=ideal):
                    improved = True
        if verbose:
            print(f"    cleanup sweep {sweep}: delay {state.delay}")
    return start_delay - state.delay
