"""Training pools and benchmark tiers.

A pool directory holds ``<name>.json`` (an m3d instance) per board and
``index.json`` listing ``{name, baseline_total, tier, side, n_nets}``. Pools
made by ``gat_order.gen_data`` and by ``gat_mcts.gen_data`` both work (only the
instance and its baseline total are used here).
"""
from __future__ import annotations

import json
import os
from typing import Dict, List

from m3d.model import Instance

from . import challenge_root

TIER_DIRS = {"intro": "benchmarks", "hard": "benchmarks_hard", "scale": "benchmarks_scale",
             "stress": "benchmarks_stress", "congested": "benchmarks_congested",
             "designs": "benchmarks_designs"}


def tier_dir(tier: str) -> str:
    """Absolute path of a benchmark tier inside the challenge checkout."""
    return os.path.join(challenge_root(), TIER_DIRS[tier])


def load_pool(pool_dir: str) -> List[Dict]:
    with open(os.path.join(pool_dir, "index.json")) as fh:
        return json.load(fh)


def pool_instance(pool_dir: str, entry: Dict) -> Instance:
    return Instance.load(os.path.join(pool_dir, entry["name"] + ".json"))
