"""Generate a pool of training instances with the challenge's own generator.

Instances are drawn like the benchmark tiers (mostly "hard"-style contended
cases, some "intro"-style sparse ones) but from a different master seed, so the
benchmarks are never trained on. Each gets the baseline total the challenge
would use (the tier's own baseline router) and the negotiated router's solution
and history costs (the ref / spt_hist candidates).

    python -m gat_mcts.gen_data --out gat_mcts_data/train --n 200 --workers 2
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from multiprocessing import Pool

from m3d.generator import GenConfig, generate_feasible

from .data import save_negotiated
from .problem import negotiate

MASTER_SEED = 4242  # benchmarks use 20260923


def make_config(i: int, rng: random.Random, min_side: int, max_side: int,
                p_sparse: float) -> GenConfig:
    side = rng.randint(min_side, max_side)
    seed = rng.randrange(1, 2 ** 31 - 1)
    if rng.random() < p_sparse:
        # intro-style: sparse, simple rip-up baseline
        return GenConfig(name=f"t{i:05d}", width=side, height=side, layers=6,
                         center_delay=1, layer_slope=1, via_delay=3,
                         n_nets=max(6, round(1.7 * side)), frac_cross=0.4, p_twopin=0.6,
                         max_fanout=5, frac_local=0.5, cell_min=2, cell_max=3,
                         pins_per_cell=3, cell_gap=1, seed=seed, master_seed=MASTER_SEED,
                         max_attempts=8, router="baseline")
    # hard-style: contended, negotiated baseline (same recipe as the hard tier)
    return GenConfig(name=f"t{i:05d}", width=side, height=side, layers=6,
                     center_delay=1, layer_slope=1, via_delay=3,
                     n_nets=round(rng.uniform(2.0, 2.6) * side), frac_cross=0.45,
                     p_twopin=0.6, max_fanout=6, frac_local=0.12, cell_min=2, cell_max=2,
                     pins_per_cell=2, cell_gap=0, seed=seed, master_seed=MASTER_SEED,
                     max_attempts=8, router="negotiated")


def _one(job):
    out, cfg = job
    t0 = time.time()
    try:
        res = generate_feasible(cfg)
    except RuntimeError as exc:
        return None, f"{cfg.name}: skipped ({exc})"
    inst = res.instance
    ref, hist = negotiate(inst)
    if ref is None:
        return None, f"{cfg.name}: skipped (negotiated router failed)"
    prefix = os.path.join(out, cfg.name)
    inst.save(prefix + ".json")
    save_negotiated(prefix, ref, hist)
    entry = {"name": cfg.name, "baseline_total": res.baseline_total,
             "style": "sparse" if cfg.router == "baseline" else "hard",
             "side": inst.width, "n_nets": len(inst.nets)}
    return entry, (f"{cfg.name}: {inst.width}x{inst.height}, {len(inst.nets)} nets, "
                   f"baseline {res.baseline_total}, {time.time() - t0:.1f}s")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default="gat_mcts_data/train")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--min-side", type=int, default=16)
    ap.add_argument("--max-side", type=int, default=30)
    ap.add_argument("--p-sparse", type=float, default=0.2)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    index_path = os.path.join(args.out, "index.json")
    index = []
    if os.path.exists(index_path):  # resumable: keep what is already there
        with open(index_path) as fh:
            index = json.load(fh)
    have = {e["name"] for e in index}
    rng = random.Random(MASTER_SEED + args.seed)
    cfgs = [make_config(i, rng, args.min_side, args.max_side, args.p_sparse)
            for i in range(args.n)]
    jobs = [(args.out, c) for c in cfgs if c.name not in have]
    print(f"generating {len(jobs)} instances into {args.out} with {args.workers} workers", flush=True)
    with Pool(args.workers) as pool:
        for entry, msg in pool.imap_unordered(_one, jobs):
            print("  " + msg, flush=True)
            if entry is not None:
                index.append(entry)
                index.sort(key=lambda e: e["name"])
                with open(index_path, "w") as fh:
                    json.dump(index, fh, indent=1)
    print(f"pool has {len(index)} instances")


if __name__ == "__main__":
    main()
