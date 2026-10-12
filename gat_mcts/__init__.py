"""GAT policy/value network + MCTS router for the M3D routing challenge.

Needs the challenge toolkit (the ``m3d`` package and the ``benchmarks_*``
directories). If ``m3d`` is not importable already, it is looked for in
``$M3D_CHALLENGE_ROOT``, then in the current directory and its parents.
"""
import os
import sys

CHALLENGE_URL = "https://github.com/partcleda/eda-3d-routing-challenge"


def challenge_root() -> str:
    """Directory holding the challenge's ``m3d/`` package and benchmark tiers."""
    tried = []
    env = os.environ.get("M3D_CHALLENGE_ROOT")
    if env:
        tried.append(env)
    d = os.getcwd()
    while True:
        tried.append(d)
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    tried.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    for c in tried:
        if os.path.isfile(os.path.join(c, "m3d", "checker.py")):
            return os.path.abspath(c)
    raise RuntimeError(
        "Cannot find the challenge toolkit (m3d/). Clone " + CHALLENGE_URL +
        " and set the environment variable M3D_CHALLENGE_ROOT to that directory.")


def _ensure_m3d() -> None:
    try:
        import m3d  # noqa: F401
    except ImportError:
        sys.path.insert(0, challenge_root())


_ensure_m3d()
