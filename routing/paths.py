"""Where the challenge checkout (m3d toolkit + benchmark tiers) lives.

azroute is its own repository; it needs a checkout of the challenge repo next to it:
  1. $AZROUTE_CHALLENGE_ROOT if set;
  2. the parent directory, when azroute sits inside the challenge repo (local development);
  3. ./eda-3d-routing-challenge or /content/eda-3d-routing-challenge (Colab).
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))


def _find() -> str:
    cands = [os.environ.get("AZROUTE_CHALLENGE_ROOT", ""), os.path.dirname(_HERE),
             os.path.join(os.getcwd(), "eda-3d-routing-challenge"), "/content/eda-3d-routing-challenge"]
    for c in cands:
        if c and os.path.isfile(os.path.join(c, "m3d", "checker.py")):
            return os.path.abspath(c)
    raise RuntimeError("challenge checkout not found: clone github.com/partcleda/eda-3d-routing-challenge "
                       "and set AZROUTE_CHALLENGE_ROOT to it")


CHALLENGE_ROOT = _find()
if CHALLENGE_ROOT not in sys.path:
    sys.path.insert(0, CHALLENGE_ROOT)
