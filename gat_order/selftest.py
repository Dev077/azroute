"""Pre-flight check: run every entry point at tiny scale before spending a
training budget. Prints PASS/FAIL per step and exits non-zero on the first failure.

    python -m gat_order.selftest [--workers 2]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time


def _step(name, fn):
    t0 = time.time()
    print(f"--- {name}", flush=True)
    try:
        fn()
    except Exception as exc:  # report and stop
        print(f"FAIL  {name}: {type(exc).__name__}: {exc}", flush=True)
        sys.exit(1)
    print(f"PASS  {name} ({time.time() - t0:.0f}s)", flush=True)


def _run(args, cwd):
    env = dict(os.environ)
    pkg_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env["PYTHONPATH"] = pkg_parent + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [sys.executable, "-m"] + args
    print("  $ " + " ".join(cmd), flush=True)
    p = subprocess.run(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    lines = [l for l in p.stdout.splitlines() if "Warning" not in l]
    for l in lines[-8:]:
        print("    " + l, flush=True)
    if p.returncode != 0:
        print("\n".join("    " + l for l in lines[-40:]))
        raise RuntimeError(f"exit code {p.returncode}")
    return p.stdout


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args(argv)
    tmp = tempfile.mkdtemp(prefix="gat_order_selftest_")
    cwd = os.getcwd()
    data, run, sub = (os.path.join(tmp, d) for d in ("data", "run", "sub"))
    w = str(args.workers)

    def env_check():
        import torch
        import torch_geometric
        from m3d.model import Instance
        from m3d.suite import load_manifest
        from . import challenge_root
        from .data import tier_dir
        from .graph import build_graph
        from .model import OrderGAT
        from .router import default_order, route
        print(f"  python {sys.version.split()[0]}, torch {torch.__version__}, "
              f"torch_geometric {torch_geometric.__version__}")
        print(f"  CUDA: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'not available (CPU only)'}")
        print(f"  challenge toolkit: {challenge_root()}")
        c = load_manifest(tier_dir("hard"))["cases"][0]
        inst = Instance.load(os.path.join(tier_dir("hard"), c["instance_file"]))
        g = build_graph(inst)
        m = OrderGAT()
        scores = m(g)
        assert scores.shape == (len(inst.nets),)
        scores.sum().backward()
        _sub, delay, secs = route(inst, default_order(inst))
        if delay != c["baseline_total"]:
            raise RuntimeError(f"default order gave {delay}, the baseline is {c['baseline_total']}")
        print(f"  {c['name']}: the default order reproduces the baseline ({secs:.0f}s)")

    _step("environment + network pass + router check", env_check)
    _step("training pool generation", lambda: _run(
        ["gat_order.gen_data", "--out", data, "--n", "3", "--min-side", "14", "--max-side", "16",
         "--workers", w], cwd))

    def train_fresh():
        _run(["gat_order.train", "--data", data, "--out", run, "--steps", "2", "--boards", "2",
              "--samples", "4", "--eval-every", "2", "--eval-cases", "1", "--eval-samples", "2",
              "--workers", w], cwd)
        for f in ("latest.pt", "best.pt", "best.json", "log.jsonl"):
            if not os.path.exists(os.path.join(run, f)):
                raise RuntimeError(f"{f} was not written")
        recs = [json.loads(l) for l in open(os.path.join(run, "log.jsonl"))]
        steps = [r for r in recs if r["kind"] == "step"]
        if len(steps) != 2 or not any(r["kind"] == "eval" for r in recs):
            raise RuntimeError("expected 2 steps and an evaluation in the log")
        if any(r["skipped"] or r["grad_norm"] != r["grad_norm"] for r in steps):
            raise RuntimeError("a step had a non-finite gradient")
        if all(r["failed"] == r["routings"] for r in steps):
            raise RuntimeError("every routing failed: nothing to learn from")
    _step("training: perturbed-order steps, evaluation", train_fresh)

    _step("training: resume", lambda: _run(
        ["gat_order.train", "--data", data, "--out", run, "--resume", os.path.join(run, "latest.pt"),
         "--steps", "3", "--boards", "1", "--samples", "2", "--eval-every", "0", "--workers", w], cwd))

    def evaluate():
        out = _run(["gat_order.evaluate", "--tier", "hard", "--ckpt", os.path.join(run, "latest.pt"),
                    "--cases", "case_01,case_02", "--samples", "2", "--workers", w, "--out", sub], cwd)
        if "legal 2/2" not in out:
            raise RuntimeError("evaluation did not produce 2 legal cases")
        for f in ("case_01.sol.json", "case_02.sol.json", "runtime.json", "meta.json"):
            if not os.path.exists(os.path.join(sub, f)):
                raise RuntimeError(f"{f} was not written")
    _step("evaluation: legal submission scored by the toolkit", evaluate)
    print(f"\nALL CHECKS PASSED (scratch files in {tmp})")


if __name__ == "__main__":
    main()
