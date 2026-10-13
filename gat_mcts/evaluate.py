"""Route a benchmark tier with a trained checkpoint and score it like the leaderboard.

    python -m gat_mcts.evaluate --tier hard --ckpt runs/gat_mcts/latest.pt \
        --out submissions_local/hard/gat_mcts

Per case: negotiated router -> cleanup pass -> up to --rounds MCTS episodes of
--moves kicks, each from the best board so far -> cleanup pass. Writes
<case>.sol.json, runtime.json (wall-clock per case, everything included) and
meta.json, then scores the directory with the toolkit's scorer.

Use ``--evaluator heuristic`` (no checkpoint) to run the same search without a
network: a check of how far the search itself goes before any training.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os

import torch

from m3d.scorer import score_submission_set
from m3d.suite import load_manifest

from .data import TIER_DIRS, tier_dir
from .workers import Runner, solve_case


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tier", default="hard", choices=list(TIER_DIRS))
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--evaluator", default="net", choices=["net", "heuristic"])
    ap.add_argument("--out", required=True, help="submission directory to write")
    ap.add_argument("--cases", default=None, help="comma-separated case names (default: all)")
    ap.add_argument("--sims", type=int, default=32, help="MCTS simulations per kick")
    ap.add_argument("--moves", type=int, default=16, help="kicks per episode (as in training)")
    ap.add_argument("--rounds", type=int, default=4,
                    help="episodes per case, each from the best board so far (stops early "
                         "once a whole-board episode finds nothing better)")
    ap.add_argument("--settle", type=int, default=40)
    ap.add_argument("--window", type=int, default=32)
    ap.add_argument("--full-max-nets", type=int, default=110)
    ap.add_argument("--c-puct", type=float, default=1.5)
    ap.add_argument("--cleanup-seconds", type=float, default=60.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                    help="device for the network during search")
    ap.add_argument("--workers", type=int, default=1, help="cases routed in parallel")
    ap.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True,
                    help="torch.compile the network (--no-compile to turn off)")
    ap.add_argument("--author", default="Your Name or Handle")
    args = ap.parse_args(argv)
    if args.evaluator == "net" and not args.ckpt:
        ap.error("--ckpt is required with --evaluator net")
    if args.ckpt and not os.path.exists(args.ckpt):
        ap.error(f"checkpoint not found: {args.ckpt}")

    suite_dir = tier_dir(args.tier)
    manifest = load_manifest(suite_dir)
    cases = manifest["cases"]
    if args.cases:
        want = set(args.cases.split(","))
        cases = [c for c in cases if c["name"] in want]
        manifest = {**manifest, "cases": cases}
    os.makedirs(args.out, exist_ok=True)

    tasks = [{"suite_dir": suite_dir, "case": c, "cache_dir": None,
              "model_path": os.path.abspath(args.ckpt) if args.ckpt else None, "model_tag": 0,
              "device": args.device, "evaluator": args.evaluator, "sims": args.sims,
              "moves": args.moves, "rounds": args.rounds, "settle": args.settle,
              "window": args.window, "full_max_nets": args.full_max_nets,
              "c_puct": args.c_puct, "cleanup_seconds": args.cleanup_seconds, "out": args.out,
              "compile": args.compile} for c in cases]
    runtimes = {}
    runner = Runner(args.workers)
    try:
        for r in runner.imap(solve_case, tasks):
            runtimes[r["case"]] = r["seconds"]
            print(f"{r['case']}: {r['n']} nets  start {r['start']:.4f}  search {r['search']:.4f}  "
                  f"final {r['final']:.4f}  [{r['rounds']} round(s)]  {r['seconds']:.0f}s", flush=True)
    finally:
        runner.close()

    with open(os.path.join(args.out, "runtime.json"), "w") as fh:
        json.dump(dict(sorted(runtimes.items())), fh, indent=1)
    meta_p = os.path.join(args.out, "meta.json")
    if not os.path.exists(meta_p):
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
        with open(meta_p, "w") as fh:
            json.dump({
                "author": args.author,
                "url": "",
                "method": "negotiated router + cleanup, then GAT policy/value net + MCTS over "
                          "kick moves (re-route a net through others at a crossing penalty, "
                          "re-route what it displaced, local settle), cleanup",
                "date": datetime.date.today().isoformat(),
                "hardware": f"{os.cpu_count()} CPU threads, GPU: {gpu}",
                "threads": args.workers,
            }, fh, indent=1)

    s = score_submission_set(manifest, suite_dir, args.out, os.path.basename(args.out), runtimes)
    print(f"\n{args.tier}: aggregate {s.aggregate:.4f}  legal {s.n_legal}/{s.n_cases}  "
          f"total delay {s.total_delay}  runtime {s.total_runtime:.0f}s")


if __name__ == "__main__":
    main()
