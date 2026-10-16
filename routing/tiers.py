"""The challenge tiers and their cases."""
from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple

from routing.paths import CHALLENGE_ROOT as ROOT

TIER_DIRS: Dict[str, str] = {
    "intro": "benchmarks",
    "hard": "benchmarks_hard",
    "scale": "benchmarks_scale",
    "stress": "benchmarks_stress",
    "congested": "benchmarks_congested",
    "designs": "benchmarks_designs",
}


def cases(tier: str) -> List[Tuple[str, str, int]]:
    """[(case name, instance path, baseline total)] for a tier."""
    d = os.path.join(ROOT, TIER_DIRS[tier])
    suite = json.load(open(os.path.join(d, "suite.json")))
    return [(c["name"], os.path.join(d, c["instance_file"]), c["baseline_total"]) for c in suite["cases"]]


def all_cases() -> List[Tuple[str, str, str, int]]:
    return [(t, n, p, b) for t in TIER_DIRS for (n, p, b) in cases(t)]
