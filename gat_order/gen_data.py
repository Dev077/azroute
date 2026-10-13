"""Generate a training pool with the challenge's own generator (the hard tier's
recipe: contended boards, negotiated-router baseline), with a different master
seed from the benchmarks so they are never trained on.

    python -m gat_order.gen_data --out pool --n 300 --workers 2

A pool from ``gat_mcts.gen_data`` works as well (train with ``--tiers hard``).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from multiprocessing import Pool

from m3d.generator import GenConfig, generate_feasible

MASTER_SEED = 4243  # benchmarks use 20260923
HARD = dict(frac_cross=0.45, p_twopin=0.6, max_fanout=6, frac_local=0.12, cell_min=2,
            cell_max=2, pins_per_cell=2, cell_gap=0, router="negotiated", max_attempts=10)


def _one(job):
    out, cfg = job
    t0 = time.time()
    try:
        res = generate_feasible(cfg)
    except RuntimeError as exc:
        return None, f"{cfg.name}: skipped ({exc})"
    inst = res.instance
    inst.save(os.path.join(out, cfg.name + ".json"))
    entry = {"name": cfg.name, "baseline_total": res.baseline_total, "tier": "hard",
             "side": inst.width, "n_nets": len(inst.nets)}
    return entry, (f"{cfg.name}: {inst.width}x{inst.height}, {len(inst.nets)} nets, "
                   f"baseline {res.baseline_total}, {time.time() - t0:.1f}s")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--min-side", type=int, default=24)
    ap.add_argument("--max-side", type=int, default=40)
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
    jobs = []
    for i in range(args.n):
        side = rng.randint(args.min_side, args.max_side)
        cfg = GenConfig(name=f"o{i:05d}", width=side, height=side, layers=6, center_delay=1,
                        layer_slope=1, via_delay=3, n_nets=max(6, round(2.6 * side)),
                        seed=rng.randrange(1, 2 ** 31 - 1), master_seed=MASTER_SEED, **HARD)
        if cfg.name not in have:
            jobs.append((args.out, cfg))
    print(f"generating {len(jobs)} boards into {args.out} with {args.workers} workers", flush=True)
    with Pool(args.workers) as pool:
        for entry, msg in pool.imap_unordered(_one, jobs):
            print("  " + msg, flush=True)
            if entry is not None:
                index.append(entry)
                index.sort(key=lambda e: e["name"])
                with open(index_path, "w") as fh:
                    json.dump(index, fh, indent=1)
    print(f"pool has {len(index)} boards")


if __name__ == "__main__":
    main()
