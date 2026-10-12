"""Generate a pool of training instances with the challenge's own generator.

Each instance is drawn from one of the benchmark tiers' recipes (``--mix``):

  hard       24-40 a side, 2.6 nets per side, contended, negotiated baseline
  congested  64-116 a side, the hard recipe at large size, negotiated baseline
  scale      100-156 a side, 1.7 nets per side, sparse, simple rip-up baseline

with the tier's own generator settings (m3d/suite.py) but a different master
seed, so the benchmarks are never trained on. Each gets the baseline total the
challenge would use (the tier's own baseline router) and that baseline's solution
as the reference routes (the ref candidates, and the routes the rest of a large
board keeps in a windowed episode). For the negotiated tiers that is the
negotiated solution plus its history costs (the spt_hist candidates), from a
single negotiated run; for scale it is the simple router's solution, with no
history (running the negotiated router there would cost minutes per board).

    python -m gat_mcts.gen_data --out gat_mcts_data/train --n 200 --workers 2
    python -m gat_mcts.gen_data --out pool --n 2000 --mix hard=0.4,congested=0.1,scale=0.5
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import replace
from multiprocessing import Pool

from m3d import baseline as _baseline
from m3d.checker import check
from m3d.generator import GenConfig, _PlacementError, generate_candidate

from .data import save_negotiated
from .problem import negotiate

MASTER_SEED = 4242  # benchmarks use 20260923

_CONTENDED = dict(frac_cross=0.45, p_twopin=0.6, max_fanout=6, frac_local=0.12,
                  cell_min=2, cell_max=2, pins_per_cell=2, cell_gap=0, router="negotiated")
_SPARSE = dict(frac_cross=0.4, p_twopin=0.65, max_fanout=5, frac_local=0.5,
               cell_min=2, cell_max=3, pins_per_cell=3, cell_gap=1, router="baseline")
# tier -> (side range, nets per side, generator settings), as in m3d/suite.py
TIERS = {
    "hard": ((24, 40), 2.6, dict(_CONTENDED, max_attempts=10)),
    "congested": ((64, 116), 2.6, dict(_CONTENDED, max_attempts=6)),
    "scale": ((100, 156), 1.7, dict(_SPARSE, max_attempts=16)),
}


def parse_mix(text: str):
    """"hard=0.4,scale=0.6" -> (["hard", "scale"], [0.4, 0.6])."""
    names, weights = [], []
    for part in text.split(","):
        name, _, w = part.partition("=")
        name = name.strip()
        if name not in TIERS:
            raise SystemExit(f"--mix: unknown tier {name!r}; choose from {', '.join(TIERS)}")
        names.append(name)
        weights.append(float(w) if w else 1.0)
    if sum(weights) <= 0:
        raise SystemExit("--mix: weights must add up to more than 0")
    return names, weights


def make_config(i: int, rng: random.Random, tiers, weights, min_side=None, max_side=None):
    """Instance i's tier and generator settings. --min-side/--max-side override every
    tier's side range (for quick tests)."""
    tier = rng.choices(tiers, weights)[0]
    (lo, hi), per_side, settings = TIERS[tier]
    side = rng.randint(lo if min_side is None else min_side, hi if max_side is None else max_side)
    seed = rng.randrange(1, 2 ** 31 - 1)
    cfg = GenConfig(name=f"t{i:05d}", width=side, height=side, layers=6,
                    center_delay=1, layer_slope=1, via_delay=3,
                    n_nets=max(6, round(per_side * side)), seed=seed,
                    master_seed=MASTER_SEED, **settings)
    return tier, cfg


def generate(cfg: GenConfig):
    """m3d.generator.generate_feasible (same seeds and growth), but also returning
    the baseline's solution and (negotiated tiers) its history costs. Returns
    (instance, baseline total, reference solution, history), or raises RuntimeError."""
    grow = attempt = 0
    while True:
        seed = cfg.seed + attempt * 100003 + grow * 1_000_003
        c = replace(cfg, width=cfg.width + grow * 6, height=cfg.height + grow * 6)
        try:
            inst = generate_candidate(c, seed)
        except _PlacementError:
            inst = None
        if inst is not None:
            if cfg.router == "negotiated":
                sub, hist = negotiate(inst, order=cfg.baseline_order)
            else:
                (sub, _stats), hist = _baseline.route(inst, order=cfg.baseline_order), {}
            if sub is not None:
                res = check(inst, sub)
                if res.legal and res.total_delay is not None:
                    inst.seed = seed
                    return inst, res.total_delay, sub, hist
        attempt += 1
        if attempt >= cfg.max_attempts:
            attempt, grow = 0, grow + 1
        if grow > 6:
            raise RuntimeError("no feasible instance found after growth")


def _one(job):
    out, tier, cfg = job
    t0 = time.time()
    try:
        inst, baseline_total, ref, hist = generate(cfg)
    except RuntimeError as exc:
        return None, f"{cfg.name} ({tier}): skipped ({exc})"
    prefix = os.path.join(out, cfg.name)
    inst.save(prefix + ".json")
    save_negotiated(prefix, ref, hist)
    entry = {"name": cfg.name, "baseline_total": baseline_total, "tier": tier,
             "side": inst.width, "n_nets": len(inst.nets)}
    return entry, (f"{cfg.name} ({tier}): {inst.width}x{inst.height}, {len(inst.nets)} nets, "
                   f"baseline {baseline_total}, {time.time() - t0:.1f}s")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default="gat_mcts_data/train")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--mix", default="hard=0.4,congested=0.1,scale=0.5",
                    help="tier=weight,... share of the pool drawn from each tier's recipe")
    ap.add_argument("--min-side", type=int, default=None, help="override every tier's smallest side")
    ap.add_argument("--max-side", type=int, default=None, help="override every tier's largest side")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    tiers, weights = parse_mix(args.mix)

    os.makedirs(args.out, exist_ok=True)
    index_path = os.path.join(args.out, "index.json")
    index = []
    if os.path.exists(index_path):  # resumable: keep what is already there
        with open(index_path) as fh:
            index = json.load(fh)
    have = {e["name"] for e in index}
    rng = random.Random(MASTER_SEED + args.seed)
    jobs = [(args.out, *make_config(i, rng, tiers, weights, args.min_side, args.max_side))
            for i in range(args.n)]
    jobs = [j for j in jobs if j[2].name not in have]
    counts = {t: sum(1 for j in jobs if j[1] == t) for t in tiers}
    print(f"generating {len(jobs)} instances into {args.out} with {args.workers} workers "
          f"({', '.join(f'{t} {n}' for t, n in counts.items())})", flush=True)
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
