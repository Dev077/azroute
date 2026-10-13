"""Scores -> orders -> routings, and the gradient of the score-based loss.

The solver is "sort the nets by score, then route them in that order with the
negotiated router". It has no useful derivative (a tiny change in the scores
changes nothing, or swaps two nets), so the gradient is taken of its smoothed
version, the expected loss when the scores are perturbed by Gaussian noise
(Berthet et al., "Learning with Differentiable Perturbed Optimizers", 2020):

    d/dw E[ L(solve(w + sigma Z)) ] = E[ L(solve(w + sigma Z)) Z ] / sigma

estimated from ``samples`` noisy orders (antithetic pairs Z, -Z; the mean loss
subtracted as a baseline, which keeps the estimate unbiased and less noisy).

The loss is the challenge score turned into something to minimise:

    L = total_delay / baseline_total = 1 / score        (fail_loss if never legal)
"""
from __future__ import annotations

import multiprocessing as mp
import os
from typing import Dict, List, Optional

import torch

from m3d.model import Instance, Submission

from .router import route

_inst_cache: Dict[str, Instance] = {}


def order_from_scores(scores: torch.Tensor, net_ids: List[int]) -> List[int]:
    """Net ids, highest score first (ties by position)."""
    idx = sorted(range(len(net_ids)), key=lambda i: (-float(scores[i]), i))
    return [net_ids[i] for i in idx]


def noise(n: int, samples: int, gen: torch.Generator) -> torch.Tensor:
    """``samples`` standard-normal vectors, in antithetic pairs (Z, -Z)."""
    half = torch.randn(max(1, samples // 2), n, generator=gen)
    z = torch.cat([half, -half])
    if samples % 2:
        z = torch.cat([z, torch.randn(1, n, generator=gen)])
    return z[:samples]


def loss_of(delay: Optional[int], baseline: int, fail_loss: float) -> float:
    return delay / baseline if delay else fail_loss


def grad_estimate(losses: torch.Tensor, z: torch.Tensor, sigma: float) -> torch.Tensor:
    """d E[L] / d scores from losses (M,) of the noisy orders made with z (M, n)."""
    centred = losses - losses.mean()
    return (centred.unsqueeze(1) * z).mean(0) / sigma


def route_task(task: dict) -> dict:
    """task: path (instance file), order (net ids), max_iters, keep (return the submission)."""
    inst = _inst_cache.get(task["path"])
    if inst is None:
        inst = _inst_cache[task["path"]] = Instance.load(task["path"])
    sub, delay, secs = route(inst, task["order"], task["max_iters"])
    return {"delay": delay, "seconds": secs,
            "sub": sub.to_dict() if (sub is not None and task.get("keep")) else None}


class Router:
    """Routes tasks on a pool of spawned processes (in-process with workers=1)."""

    def __init__(self, workers: int):
        self.pool = mp.get_context("spawn").Pool(workers) if workers > 1 else None

    def map(self, tasks: List[dict]) -> List[dict]:
        if self.pool is None:
            return [route_task(t) for t in tasks]
        return self.pool.map(route_task, tasks, chunksize=1)

    def close(self):
        if self.pool is not None:
            self.pool.close()
            self.pool.join()
            self.pool = None


def candidate_orders(scores: torch.Tensor, net_ids: List[int], samples: int, sigma: float,
                     seed: int) -> List[List[int]]:
    """The greedy order (scores as they are) plus ``samples`` - 1 noisy ones."""
    gen = torch.Generator().manual_seed(seed)
    out = [order_from_scores(scores, net_ids)]
    if samples > 1:
        z = noise(len(net_ids), samples - 1, gen)
        out += [order_from_scores(scores + sigma * zi, net_ids) for zi in z]
    return out


def best_routing(router: Router, path: str, orders: List[List[int]], baseline: int,
                 max_iters: int) -> dict:
    """Route every order; return the best legal one (and the greedy order's score)."""
    res = router.map([{"path": path, "order": o, "max_iters": max_iters, "keep": True}
                      for o in orders])
    scores = [baseline / r["delay"] if r["delay"] else 0.0 for r in res]
    k = max(range(len(res)), key=lambda i: (scores[i], -i))
    return {"greedy": scores[0], "best": scores[k], "legal": sum(s > 0 for s in scores),
            "sub": Submission.from_dict(res[k]["sub"]) if res[k]["sub"] else None,
            "seconds": sum(r["seconds"] for r in res), "index": k}


def default_workers() -> int:
    return max(1, (os.cpu_count() or 2))
