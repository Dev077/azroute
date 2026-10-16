"""Where the challenge checkout (m3d toolkit + benchmark tiers) lives.

This repository needs a checkout of the challenge repo:
  1. $AZROUTE_CHALLENGE_ROOT if set;
  2. the parent directory, when this repository sits inside the challenge repo (local development);
  3. ./eda-3d-routing-challenge or /content/eda-3d-routing-challenge (Colab).
"""
from __future__ import annotations

import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _find() -> str:
    cands = [os.environ.get("AZROUTE_CHALLENGE_ROOT", ""), os.path.dirname(_REPO),
             os.path.join(os.getcwd(), "eda-3d-routing-challenge"), "/content/eda-3d-routing-challenge"]
    for c in cands:
        if c and os.path.isfile(os.path.join(c, "m3d", "checker.py")):
            return os.path.abspath(c)
    raise RuntimeError("challenge checkout not found: clone github.com/partcleda/eda-3d-routing-challenge "
                       "and set AZROUTE_CHALLENGE_ROOT to it")


CHALLENGE_ROOT = _find()
if CHALLENGE_ROOT not in sys.path:
    # appended, not prepended: the challenge repo has its own top-level packages (tests/, ...)
    # that must not shadow this repository's modules
    sys.path.append(CHALLENGE_ROOT)
