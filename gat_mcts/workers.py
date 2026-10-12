"""Units of work that run in parallel worker processes (or in-process with
``--workers 1``): one training episode, or solving one benchmark case.

Workers are started with the "spawn" method (safe with CUDA). The network is
handed over as a checkpoint path plus a tag; a worker reloads it only when the
tag changes.
"""
from __future__ import annotations

import io
import multiprocessing as mp
import os
import random
import time

import torch

from .data import benchmark_problem, pool_problem
from .mcts import MCTS
from .model import RouteGAT
from .selfplay import label_samples, play, play_guided, solve

_model = None
_model_tag = None


def init_worker(threads: int = 1) -> None:
    torch.set_num_threads(threads)


def _get_model(path: str, tag, device: str, compile_: bool = False):
    """The worker's network. Kept across tasks: new weights are loaded into the same
    module, so a compiled model is compiled once per worker, not every iteration."""
    global _model, _model_tag
    if _model is None or _model_tag != tag:
        ck = torch.load(path, map_location=device, weights_only=False)
        if _model is None or _model.config != ck["config"]:
            _model = RouteGAT(**ck["config"]).to(device)
        _model.load_state_dict(ck["state_dict"])
        _model.eval()
        _model_tag = tag
    if compile_ and not _model.compiled:
        _model.set_compiled(True)
    return _model


def pack_samples(samples) -> bytes:
    """Serialise an episode's samples into one blob for the trip back to the main process.
    Returned as tensors, torch's multiprocessing would hand each tensor over as its own
    shared-memory mapping, and a replay buffer full of those exhausts the process's
    mapping limit ("unable to mmap ... Cannot allocate memory")."""
    buf = io.BytesIO()
    torch.save(samples, buf)
    return buf.getvalue()


def unpack_samples(blob: bytes):
    return torch.load(io.BytesIO(blob), weights_only=False)  # our own workers' output


def run_episode(task: dict) -> dict:
    """task: kind ("warm" | "mcts"), data, entry, seed, frontier, rip_limit, plus
    eps (warm) or model_path, model_tag, device, sims, c_puct, temp_moves, compile (mcts)."""
    rng = random.Random(task["seed"])
    prob = pool_problem(task["data"], task["entry"], frontier=task["frontier"],
                        rip_limit=task["rip_limit"], seed=task["seed"])
    if task["kind"] == "warm":
        ep = play_guided(prob, task["eps"], rng)
    else:
        model = _get_model(task["model_path"], task["model_tag"], task["device"],
                           task.get("compile", False))
        mcts = MCTS(prob, model, task["device"], c_puct=task["c_puct"], rng=rng)
        ep = play(prob, mcts, task["sims"], train=True, temp_moves=task["temp_moves"])
    return {"samples": pack_samples(label_samples(ep)), "score": ep.score, "legal": ep.legal,
            "moves": ep.moves, "seconds": ep.seconds, "instance": task["entry"]["name"],
            "n": prob.n, "eps": task.get("eps")}


def solve_case(task: dict) -> dict:
    """task: suite_dir, case, cache_dir, model_path (None for rollout), model_tag,
    device, evaluator, sims, c_puct, frontier, rip_limit, cleanup_seconds, out (or None),
    compile (optional)."""
    t0 = time.time()
    prob, _neg_s = benchmark_problem(task["suite_dir"], task["case"], cache_dir=task.get("cache_dir"),
                                     frontier=task["frontier"], rip_limit=task["rip_limit"])
    model = None
    if task["evaluator"] == "net":
        model = _get_model(task["model_path"], task["model_tag"], task["device"],
                           task.get("compile", False))
    mcts = MCTS(prob, model, task["device"], c_puct=task["c_puct"],
                evaluator=task["evaluator"], rng=random.Random(0))
    sol = solve(prob, mcts, task["sims"], cleanup_seconds=task["cleanup_seconds"])
    if task.get("out"):
        sol.submission.save(os.path.join(task["out"], f"{prob.inst.name}.sol.json"))
    return {"case": prob.inst.name, "n": prob.n,
            "search": prob.score(sol.delay_search) if sol.delay_search else 0.0,
            "final": prob.score(sol.delay_final), "path": sol.path,
            "seconds": round(time.time() - t0, 2)}  # wall clock incl. the negotiated router


class Runner:
    """Maps tasks over a spawn pool, or in-process when workers == 1."""

    def __init__(self, workers: int):
        self.workers = max(1, workers)
        self.pool = None
        if self.workers > 1:
            self.pool = mp.get_context("spawn").Pool(self.workers, initializer=init_worker,
                                                     initargs=(1,))

    def imap(self, fn, tasks):
        if self.pool is None:
            for t in tasks:
                yield fn(t)
        else:
            yield from self.pool.imap_unordered(fn, tasks)

    def submit(self, fn, task):
        """Start ``fn(task)`` without waiting. Returns a handle with ready()/get();
        in-process it runs right away (so nothing overlaps with workers == 1)."""
        if self.pool is not None:
            return self.pool.apply_async(fn, (task,))
        return _Done(fn, task)

    def close(self):
        if self.pool is not None:
            self.pool.close()
            self.pool.join()
            self.pool = None


class _Done:
    """In-process stand-in for a pool AsyncResult."""

    def __init__(self, fn, task):
        try:
            self._value, self._exc = fn(task), None
        except Exception as exc:  # re-raised by get(), like a pool result
            self._value, self._exc = None, exc

    def ready(self):
        return True

    def get(self):
        if self._exc is not None:
            raise self._exc
        return self._value
