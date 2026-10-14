"""Compare acceptance rules for the improvement loop: from the same starting routings, run the
same network-free reroutes for the same time per case under each rule, keep the best routing
each one finds, and report the tier scores (geometric mean of baseline / delay).

    python -m azroute.accept_bench --tier hard --seconds 600 --from /path/to/run_work

`--from` is a run's work (or Drive) folder: its state/<tier>__<case>.npz routings are the starting
points. Without it, each case starts from the initial routing plus greedy rerouting until it stalls
(a much shallower plateau than a long run reaches).
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import random
import time

from .lns import Acceptance, Improver, LateAcceptance, RecordToRecord, greedy_policy, greedy_until_stall
from .problem import Problem
from .search import Search, completion_evaluator, drive
from .tiers import cases
from .worker import load_routing, save_routing


def make_rule(name: str, a) -> Acceptance:
    if name == "strict":
        return Acceptance()
    if name == "rrt":
        return RecordToRecord(a.tau, a.seconds)
    if name == "lahc":
        return LateAcceptance(a.lahc)
    raise ValueError(f"unknown rule {name!r} (strict, rrt, lahc)")


def prepare(job):
    """Write the starting routing of one case to <out>/state/ (once; reused by every rule)."""
    tier, name, path, a = job
    key = f"{tier}/{name}"
    if load_routing(a.out, key) is not None:
        return
    P = Problem.load(path)
    stored = load_routing(a.src, key) if a.src else None
    if stored is not None:
        P.restore(stored[1])
    else:
        big = P.W * P.H * P.L > 300_000
        if not P.initial_route(schedules=[P.SCHEDULES[0]] if big else None):
            raise RuntimeError(f"{key}: initial routing failed")
        greedy_until_stall(Improver(P, random.Random(0)), stall_tries=a.stall_tries, max_seconds=a.greedy_seconds)
    save_routing(a.out, key, P, 0)


def run_one(job):
    tier, name, path, base, rule, seed, a = job
    P = Problem.load(path)
    P.restore(load_routing(a.out, f"{tier}/{name}")[1])
    start = P.router.total_delay()
    rng = random.Random(seed)
    imp = Improver(P, random.Random(rng.randrange(2 ** 32)), accept=make_rule(rule, a))
    search = Search(imp, n_full=a.n_full, n_fast=a.n_fast, p_full=a.p_full, max_considered=a.considered,
                    record=False, rng=random.Random(rng.randrange(2 ** 32)))
    ev = completion_evaluator(imp)
    stats = {"reroutes": 0, "kept_better": 0, "kept_equal": 0, "kept_worse": 0, "blocked": 0}
    t0 = time.time()
    while time.time() - t0 < a.seconds:
        s = imp.pick_seed()
        if s is None:
            break
        imp.max_nets = rng.randint(a.k_min, a.k_max)
        before = imp.total
        hood = imp.neighbourhood(s)
        if a.move == "greedy":
            ok, old, new = imp.reroute(hood, greedy_policy)
        else:
            ok, old, new, _ = drive(search.reroute(hood), ev)
        stats["reroutes"] += 1
        if new < 0:
            stats["blocked"] += 1
        elif ok:
            d = imp.total - before
            stats["kept_better" if d < 0 else "kept_equal" if d == 0 else "kept_worse"] += 1
    final_seen = imp.total
    imp.restore_best()
    res = P.verify()
    if not res.legal or res.total_delay != P.router.total_delay() or res.total_delay != imp.best_total:
        raise RuntimeError(f"{tier}/{name} {rule}: best routing does not verify ({res.reasons[:3]})")
    return {"tier": tier, "case": name, "rule": rule, "seed": seed, "baseline": base, "start": start,
            "best": imp.best_total, "final_seen": final_seen, "seconds": round(time.time() - t0, 1), **stats}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tier", default="hard")
    ap.add_argument("--cases", default="", help="comma-separated case indices (default: all)")
    ap.add_argument("--rules", default="strict,rrt,lahc")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--seconds", type=float, default=600, help="rerouting time per case and rule")
    ap.add_argument("--from", dest="src", default="", help="run work/Drive folder with state/*.npz to start from")
    ap.add_argument("--out", default="accept_bench_out")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--move", choices=["search", "greedy"], default="search",
                    help="search = the run's network-free search; greedy = DELAY mode, random order")
    ap.add_argument("--tau", type=float, default=0.003, help="rrt: allowed excess over the best total at the start")
    ap.add_argument("--lahc", type=int, default=200, help="lahc: history length in reroutes")
    ap.add_argument("--k-min", type=int, default=3)
    ap.add_argument("--k-max", type=int, default=8)
    ap.add_argument("--n-full", type=int, default=32)
    ap.add_argument("--n-fast", type=int, default=8)
    ap.add_argument("--p-full", type=float, default=0.25)
    ap.add_argument("--considered", type=int, default=16)
    ap.add_argument("--stall-tries", type=int, default=300, help="without --from: greedy stall length")
    ap.add_argument("--greedy-seconds", type=float, default=300)
    a = ap.parse_args(argv)
    os.makedirs(os.path.join(a.out, "state"), exist_ok=True)

    cs = cases(a.tier)
    idx = [int(i) for i in a.cases.split(",")] if a.cases else range(len(cs))
    chosen = [cs[i] for i in idx]
    rules = a.rules.split(",")
    for r in rules:
        make_rule(r, a)
    ctx = mp.get_context("spawn")
    with ctx.Pool(a.jobs) as pool:
        pool.map(prepare, [(a.tier, n, p, a) for n, p, _ in chosen])
        jobs = [(a.tier, n, p, b, r, s, a) for n, p, b in chosen for s in range(a.seeds) for r in rules]
        rows = []
        with open(os.path.join(a.out, "results.jsonl"), "a") as f:
            for row in pool.imap_unordered(run_one, jobs):
                rows.append(row)
                f.write(json.dumps(row) + "\n")
                f.flush()
                print(f"{row['case']:>12s} {row['rule']:6s} seed {row['seed']}: {row['start']} -> {row['best']} "
                      f"({100 * (row['start'] - row['best']) / row['start']:.2f}%)  reroutes={row['reroutes']} "
                      f"kept better/equal/worse={row['kept_better']}/{row['kept_equal']}/{row['kept_worse']}",
                      flush=True)

    def gm(rs, key):
        return math.exp(sum(math.log(r["baseline"] / r[key]) for r in rs) / len(rs))

    print(f"\n{a.tier}: {len(chosen)} cases, {a.seconds:.0f} s per case, move={a.move}")
    print(f"{'rule':8s} {'start':>8s} {'best':>8s} {'gain':>7s} {'reroutes':>9s}")
    for r in rules:
        rs = [x for x in rows if x["rule"] == r]
        s0, s1 = gm(rs, "start"), gm(rs, "best")
        print(f"{r:8s} {s0:8.4f} {s1:8.4f} {100 * (s1 / s0 - 1):6.2f}% {sum(x['reroutes'] for x in rs) / len(rs):9.0f}")


if __name__ == "__main__":
    main()
