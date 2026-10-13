"""Pre-flight check: run every entry point at tiny scale before spending a
training budget. Takes a few minutes; prints PASS/FAIL per step and exits
non-zero on the first failure.

    python -m gat_mcts.selftest [--workers 2]
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
    p = subprocess.run(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       text=True)
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
    tmp = tempfile.mkdtemp(prefix="gat_mcts_selftest_")
    cwd = os.getcwd()
    data, run, sub = (os.path.join(tmp, d) for d in ("data", "run", "sub"))
    w = str(args.workers)

    def env_check():
        import torch
        import torch_geometric
        from . import challenge_root
        from .data import tier_dir
        print(f"  python {sys.version.split()[0]}, torch {torch.__version__}, "
              f"torch_geometric {torch_geometric.__version__}")
        print(f"  CUDA: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'not available (CPU only)'}")
        print(f"  challenge toolkit: {challenge_root()}")
        if not os.path.isfile(os.path.join(tier_dir("hard"), "suite.json")):
            raise RuntimeError("benchmarks_hard/suite.json missing from the challenge checkout")
        # one forward + backward pass on the training device
        from m3d.suite import load_manifest
        from m3d.model import Instance, Submission
        from torch_geometric.data import Batch
        from .graph import build_graph
        from .model import RouteGAT
        from .problem import Board, Problem
        from .selfplay import start_state
        c = load_manifest(tier_dir("hard"))["cases"][0]
        inst = Instance.load(os.path.join(tier_dir("hard"), c["instance_file"]))
        ref = Submission.load(os.path.join(tier_dir("hard"), c["reference_file"]))
        prob = Problem(inst, c["baseline_total"], reference=ref)
        board = Board(prob, start_state(prob, 5.0), moves=4)
        acts = board.actions(board.start)
        g = build_graph(board, board.start, acts, 0, board.start.delay)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        m = RouteGAT().to(dev)
        logits, v, _ = m(Batch.from_data_list([g, g]).to(dev))
        (logits.sum() + v.sum()).backward()
        prior, gain = m.evaluate(g, dev)
        assert len(prior) == len(acts) and abs(sum(prior) - 1) < 1e-4

    _step("environment + one network pass", env_check)
    _step("training pool generation", lambda: _run(
        ["gat_mcts.gen_data", "--out", data, "--n", "3", "--min-side", "14", "--max-side", "16",
         "--workers", w], cwd))

    def train_fresh():
        _run(["gat_mcts.train", "--data", data, "--out", run, "--iters", "1", "--episodes", "2",
              "--moves", "3", "--sims", "2", "--start-seconds", "2", "--train-steps", "2",
              "--eval-every", "1", "--eval-cases", "1", "--eval-sims", "2",
              "--eval-cleanup", "1", "--full-max-nets", "0", "--window", "8", "--workers", w], cwd)
        for f in ("latest.pt", "iter_0000.pt", "best.pt", "best.json", "log.jsonl"):
            if not os.path.exists(os.path.join(run, f)):
                raise RuntimeError(f"{f} was not written")
        recs = [json.loads(l) for l in open(os.path.join(run, "log.jsonl"))]
        missing = {"eval_heuristic", "selfplay", "iter", "eval"} - {r["kind"] for r in recs}
        if missing:
            raise RuntimeError(f"log is missing {missing}")
        it = [r for r in recs if r["kind"] == "iter"][-1]
        for k in ("loss", "policy", "value", "grad_norm", "lr"):
            if not (isinstance(it.get(k), (int, float)) and it[k] == it[k] and abs(it[k]) != float("inf")):
                raise RuntimeError(f"iteration log has a bad {k}: {it.get(k)}")
        if not it.get("fresh"):
            raise RuntimeError("iteration log has no unseen-data loss")
        if not all(r.get("window") == 8 for r in recs if r["kind"] == "selfplay"):
            raise RuntimeError("--full-max-nets 0 --window 8: every episode should kick a window of 8 nets")
        if not all(r["score"] >= r["start"] - 1e-9 for r in recs if r["kind"] == "selfplay"):
            raise RuntimeError("an episode's best board is worse than the board it started from")
    _step("training: self-play, evaluation", train_fresh)

    def resume():
        _run(["gat_mcts.train", "--data", data, "--out", run, "--resume", os.path.join(run, "latest.pt"),
              "--iters", "2", "--episodes", "1", "--moves", "3", "--sims", "2", "--train-steps", "2",
              "--start-seconds", "2", "--eval-every", "0", "--workers", w], cwd)
        last = [json.loads(l) for l in open(os.path.join(run, "log.jsonl"))
                if '"selfplay"' in l][-1]
        if last.get("window") is not None:
            raise RuntimeError("small boards should kick the whole board by default")
    _step("training: resume (whole-board episodes)", resume)

    def evaluate():
        out = _run(["gat_mcts.evaluate", "--tier", "hard", "--ckpt", os.path.join(run, "latest.pt"),
                    "--cases", "case_01,case_02", "--sims", "2", "--moves", "2", "--rounds", "1",
                    "--cleanup-seconds", "1",
                    "--workers", w, "--out", sub], cwd)
        if "legal 2/2" not in out:
            raise RuntimeError("evaluation did not produce 2 legal cases")
        for f in ("case_01.sol.json", "case_02.sol.json", "runtime.json", "meta.json"):
            if not os.path.exists(os.path.join(sub, f)):
                raise RuntimeError(f"{f} was not written")
    _step("evaluation: legal submission scored by the toolkit", evaluate)
    print(f"\nALL CHECKS PASSED (scratch files in {tmp})")


if __name__ == "__main__":
    main()
