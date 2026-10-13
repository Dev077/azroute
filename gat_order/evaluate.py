"""Route a benchmark tier with a trained ordering GAT and score it like the leaderboard.

    python -m gat_order.evaluate --tier hard --ckpt runs/order1/best.pt --out submissions_local/hard/order1

Per case: the GAT scores the nets; the negotiated router routes the board in
the GAT's order and in --samples - 1 noisy versions of it; the best legal
routing is submitted (if none is legal, the router's default order, which is
the baseline). Writes <case>.sol.json, runtime.json (wall clock per case) and
meta.json, then scores the directory with the toolkit's scorer.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import time

import torch

from m3d.model import Instance
from m3d.scorer import score_submission_set
from m3d.suite import load_manifest

from .data import TIER_DIRS, tier_dir
from .graph import build_graph
from .model import load_checkpoint
from .router import default_order, route
from .solver import Router, best_routing, candidate_orders


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tier", default="hard", choices=list(TIER_DIRS))
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True, help="submission directory to write")
    ap.add_argument("--cases", default=None, help="comma-separated case names (default: all)")
    ap.add_argument("--samples", type=int, default=16, help="greedy order + noisy ones per case")
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--max-iters", type=int, default=50)
    ap.add_argument("--workers", type=int, default=1, help="routings in parallel")
    ap.add_argument("--author", default="Your Name or Handle")
    args = ap.parse_args(argv)

    suite = tier_dir(args.tier)
    manifest = load_manifest(suite)
    cases = manifest["cases"]
    if args.cases:
        want = set(args.cases.split(","))
        cases = [c for c in cases if c["name"] in want]
        manifest = {**manifest, "cases": cases}
    os.makedirs(args.out, exist_ok=True)
    model, _ck = load_checkpoint(args.ckpt, "cpu")
    model.eval()

    runtimes = {}
    router = Router(args.workers)
    try:
        for c in cases:
            t0 = time.time()
            path = os.path.join(suite, c["instance_file"])
            inst = Instance.load(path)
            with torch.no_grad():
                w = model(build_graph(inst))
            orders = candidate_orders(w, [net.id for net in inst.nets], args.samples, args.sigma, seed=1)
            r = best_routing(router, path, orders, c["baseline_total"], args.max_iters)
            sub, note = r["sub"], f"order {r['index']} of {len(orders)}"
            if sub is None:  # nothing legal: the router's default order (the baseline)
                sub, _d, _s = route(inst, default_order(inst), args.max_iters)
                note = "default order (no GAT order was legal)"
            sub.save(os.path.join(args.out, f"{inst.name}.sol.json"))
            runtimes[inst.name] = round(time.time() - t0, 2)
            print(f"{c['name']}: {len(inst.nets)} nets  greedy {r['greedy']:.4f}  best {r['best']:.4f}  "
                  f"({r['legal']}/{len(orders)} legal, kept {note})  {runtimes[inst.name]:.0f}s", flush=True)
    finally:
        router.close()

    with open(os.path.join(args.out, "runtime.json"), "w") as fh:
        json.dump(dict(sorted(runtimes.items())), fh, indent=1)
    meta_p = os.path.join(args.out, "meta.json")
    if not os.path.exists(meta_p):
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
        with open(meta_p, "w") as fh:
            json.dump({
                "author": args.author,
                "url": "",
                "method": "GAT net ordering for the toolkit's negotiated router, trained on the "
                          "score (perturbed-optimizer gradient); best of the GAT's order and noisy "
                          "versions of it",
                "date": datetime.date.today().isoformat(),
                "hardware": f"{os.cpu_count()} CPU threads, GPU: {gpu}",
                "threads": args.workers,
            }, fh, indent=1)

    s = score_submission_set(manifest, suite, args.out, os.path.basename(args.out), runtimes)
    print(f"\n{args.tier}: aggregate {s.aggregate:.4f}  legal {s.n_legal}/{s.n_cases}  "
          f"total delay {s.total_delay}  runtime {s.total_runtime:.0f}s")


if __name__ == "__main__":
    main()
