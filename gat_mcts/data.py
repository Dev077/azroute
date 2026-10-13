"""Training-instance pool + loading benchmark cases as Problems.

A pool directory holds, per instance:
  <name>.json           the instance (m3d-instance)
  <name>.ref.sol.json   the negotiated router's legal solution
  <name>.hist.json      the negotiated router's per-vertex history costs (not used
                        by the kick search; kept so older pools stay valid)
and index.json listing {name, baseline_total, tier, side, n_nets}.
"""
from __future__ import annotations

import json
import os
import time
from typing import Dict, Optional, Tuple

from m3d.model import Instance, Submission

from . import challenge_root
from .problem import Problem, negotiate

TIER_DIRS = {"intro": "benchmarks", "hard": "benchmarks_hard", "scale": "benchmarks_scale",
             "stress": "benchmarks_stress", "congested": "benchmarks_congested",
             "designs": "benchmarks_designs"}


def tier_dir(tier: str) -> str:
    """Absolute path of a benchmark tier inside the challenge checkout."""
    return os.path.join(challenge_root(), TIER_DIRS[tier])


def save_negotiated(path_prefix: str, ref: Submission, hist: Dict[int, float]) -> None:
    ref.save(path_prefix + ".ref.sol.json")
    with open(path_prefix + ".hist.json", "w") as fh:
        json.dump({str(k): round(v, 4) for k, v in hist.items()}, fh)


def load_negotiated(path_prefix: str) -> Optional[Tuple[Submission, Dict[int, float]]]:
    ref_p, hist_p = path_prefix + ".ref.sol.json", path_prefix + ".hist.json"
    if not (os.path.exists(ref_p) and os.path.exists(hist_p)):
        return None
    with open(hist_p) as fh:
        hist = {int(k): float(v) for k, v in json.load(fh).items()}
    return Submission.load(ref_p), hist


def load_pool(pool_dir: str):
    with open(os.path.join(pool_dir, "index.json")) as fh:
        return json.load(fh)


def pool_problem(pool_dir: str, entry: Dict, **kw) -> Problem:
    prefix = os.path.join(pool_dir, entry["name"])
    inst = Instance.load(prefix + ".json")
    ref, _hist = load_negotiated(prefix)
    return Problem(inst, entry["baseline_total"], reference=ref, **kw)


def benchmark_problem(suite_dir: str, case: Dict, cache_dir: Optional[str] = None,
                      **kw) -> Tuple[Problem, float]:
    """Problem for a benchmark case. Runs the negotiated router (its solution is
    where the search starts) unless ``cache_dir`` already holds its result;
    returns the seconds spent on it so it can be counted in the reported runtime."""
    inst = Instance.load(os.path.join(suite_dir, case["instance_file"]))
    t0 = time.time()
    got = None
    prefix = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        prefix = os.path.join(cache_dir, f"{os.path.basename(os.path.normpath(suite_dir))}_{inst.name}")
        got = load_negotiated(prefix)
    if got is None:
        ref, hist = negotiate(inst)
        if prefix and ref is not None:
            save_negotiated(prefix, ref, hist)
    else:
        ref, hist = got
    secs = time.time() - t0
    return Problem(inst, case["baseline_total"], reference=ref, **kw), secs
