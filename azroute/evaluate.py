"""One-shot test of a trained network: route every case from scratch, no training, no polish.

    python -m azroute.evaluate --model /content/drive/MyDrive/azroute_run/model.pt \
        --work /content/eval_network --out /content/drive/MyDrive/azroute_eval/network --minutes 10

Each case starts from its initial routing (negotiated congestion) and gets --minutes of worker
time of greedy rerouting plus search-guided reroutes, with the network guiding the search and
strict acceptance (a reroute is kept only if it does not increase the delay). With --arm control
the same visits run network-free, so the two arms show what the network adds for the same
time. The work directory must be new: stored routings would make the test start from them.

Reuses the run's worker and inference processes (worker.py, inference.py) unchanged; only the
dispatcher differs: fixed time per case instead of a deadline, no learner, no polish workers.
Ends with the same checked export as a run (export.py): .sol.json, runtime.json, summary.md.
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import queue
import shutil
import time


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="", help="model.pt from a training run (required for --arm network)")
    ap.add_argument("--arm", default="network", choices=["network", "control"],
                    help="network: the network guides the search; control: the same visits network-free")
    ap.add_argument("--work", required=True, help="new, empty directory for this test's state")
    ap.add_argument("--out", required=True, help="where the checked export goes")
    ap.add_argument("--minutes", type=float, default=10.0, help="worker time per case")
    ap.add_argument("--max-stall", type=int, default=8, help="a case ends after this many visits without gain")
    ap.add_argument("--hours", type=float, default=12.0, help="wall-time cap for the whole test")
    ap.add_argument("--tiers", default="all")
    ap.add_argument("--cases", default="", help="comma list of tier/case keys (overrides --tiers)")
    ap.add_argument("--workers", type=int, default=0, help="0 = auto, at most one per case")
    # search settings: the run's defaults, so the network is tested the way it was trained
    ap.add_argument("--k-max", type=int, default=8)
    ap.add_argument("--k-max-stalled", type=int, default=16)
    ap.add_argument("--n-full", type=int, default=128)
    ap.add_argument("--n-fast", type=int, default=8)
    ap.add_argument("--p-full", type=float, default=0.25)
    ap.add_argument("--gumbel-scale", type=float, default=1.0)
    ap.add_argument("--considered", type=int, default=32)
    ap.add_argument("--considered-fast", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args(argv)

    import torch
    from .export import export
    from .model import load_model
    from .run import auto_workers
    from .tiers import TIER_DIRS, all_cases

    work = os.path.abspath(a.work)
    if os.path.isdir(os.path.join(work, "state")) and os.listdir(os.path.join(work, "state")):
        raise SystemExit(f"{work} already holds routings; use a new --work directory for a one-shot test")
    for d in ("state", "gen_state", "records"):
        os.makedirs(os.path.join(work, d), exist_ok=True)
    for f in ("STOP", "net_ready", "model.pt"):
        if os.path.exists(os.path.join(work, f)):
            os.remove(os.path.join(work, f))

    device = torch.device("cuda" if torch.cuda.is_available() and not a.cpu else "cpu")
    if a.arm == "network":
        if not a.model or not os.path.exists(a.model):
            raise SystemExit("--arm network needs --model pointing at a model.pt")
        _, ck = load_model(a.model, device)        # fails here, not in the workers, if the file is bad
        shutil.copy2(a.model, os.path.join(work, "model.pt"))
        open(os.path.join(work, "net_ready"), "w").close()
        log(f"model {a.model} ({ck.get('steps', '?')} training steps)")

    base = {f"{t}/{n}": b for (t, n, _p, b) in all_cases()}
    if a.cases:
        cases = [k.strip() for k in a.cases.split(",") if k.strip()]
        unknown = [k for k in cases if k not in base]
        if unknown:
            raise SystemExit(f"unknown cases: {unknown}")
    else:
        tiers = list(TIER_DIRS) if a.tiers == "all" else a.tiers.split(",")
        cases = [k for k in base if k.split("/")[0] in tiers]
    gpu_gb = torch.cuda.get_device_properties(0).total_memory / 2 ** 30 if device.type == "cuda" else 0.0
    W = max(1, min(len(cases), auto_workers(a.workers, gpu_gb)))
    budget = a.minutes * 60

    # the run's worker settings, with every visit network-guided (or none, for control) and no polish
    cfg = {"work": work, "seed": a.seed, "stop_flag": os.path.join(work, "STOP"), "cpu": a.cpu,
           "concurrency": 1, "cache_cases": 2, "stall_tries": 200, "greedy_seconds": 15.0,
           "min_excess": 0.003, "max_search": 64, "n_full": a.n_full, "n_fast": a.n_fast, "p_full": a.p_full,
           "gumbel_scale": a.gumbel_scale, "considered": a.considered, "considered_fast": a.considered_fast,
           "k_min": 3, "k_max": a.k_max, "control_share": 0.0, "explore_fast": True, "gen_cases": {},
           "polish_workers": 0, "polish_seconds": 0.0, "polish_tau": 0.0, "polish_search": {}}
    json.dump(dict(vars(a), workers=W, device=str(device), cases=len(cases)),
              open(os.path.join(work, "eval_config.json"), "w"), indent=1)
    log(f"arm={a.arm} device={device} workers={W} cases={len(cases)} minutes/case={a.minutes}")

    stats = {"_meta": {"learner_seconds": 0.0}}
    for k in cases:
        t, n = k.split("/")
        stats[k] = {"tier": t, "case": n, "baseline": base[k], "seconds": 0.0, "visits": 0, "records": 0,
                    "stall": 0}
    done: set = set()       # cases out of budget, stalled, or failed
    busy: set = set()

    from .inference import inference_main
    from .worker import worker_main
    ctx = mp.get_context("spawn")
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(v, "1")
    req_q = ctx.Queue()
    resp_qs = [ctx.Queue() for _ in range(W)]
    infer_q = ctx.Queue()
    infer_resp_qs = [ctx.Queue() for _ in range(W)]
    infer_proc = ctx.Process(target=inference_main, args=(cfg, infer_q, infer_resp_qs, req_q), daemon=True)
    infer_proc.start()
    procs = [ctx.Process(target=worker_main, args=(i, cfg, req_q, resp_qs[i], infer_q, infer_resp_qs[i]),
                         daemon=True) for i in range(W)]
    for p in procs:
        p.start()
    held = {i: None for i in range(W)}
    visits_f = open(os.path.join(work, "visits.jsonl"), "a")
    t_start = time.time()
    deadline = t_start + a.hours * 3600
    stopping = False
    exited: set = set()     # workers that exited or died

    def next_case():
        """The free case that has had the least worker time: every case moves forward evenly."""
        free = [k for k in cases if k not in busy and k not in done]
        return min(free, key=lambda k: stats[k]["seconds"]) if free else None

    def summary() -> str:
        parts = []
        for t in TIER_DIRS:
            rs = [s for k, s in stats.items() if not k.startswith("_") and s["tier"] == t and s.get("best")]
            if rs:
                gm = math.exp(sum(math.log(s["baseline"] / s["best"]) for s in rs) / len(rs))
                parts.append(f"{t}={gm:.4f}({len(rs)})")
        return " ".join(parts)

    def save():
        json.dump(stats, open(os.path.join(work, "case_stats.json.tmp"), "w"), indent=1)
        os.replace(os.path.join(work, "case_stats.json.tmp"), os.path.join(work, "case_stats.json"))

    def handle(msg):
        kind = msg[0]
        if kind == "get":
            wid = msg[1]
            k = None if stopping else next_case()
            if k is None:
                resp_qs[wid].put(None)
                return
            busy.add(k)
            held[wid] = k
            st = stats[k]["stall"]
            resp_qs[wid].put((k, {"k_max": min(a.k_max_stalled, a.k_max + 2 * st) if st else a.k_max,
                                  "polish": False}))
        elif kind == "done":
            k, st = msg[1], msg[2]
            busy.discard(k)
            for w, h in held.items():
                if h == k:
                    held[w] = None
            s = stats[k]
            s["seconds"] += st.get("seconds", 0.0)
            s["visits"] += 1
            if st.get("failed"):
                done.add(k)
                log(f"{k}: failed")
            elif "delay" in st:
                s["best"] = st["delay"]
                s["reroutes"] = s.get("reroutes", 0) + st.get("reroutes", 0)
                s["better"] = s.get("better", 0) + st.get("better", 0)
                s["stall"] = 0 if st["delay"] < st["start"] else s["stall"] + 1
                if s["seconds"] >= budget or s["stall"] >= a.max_stall or not st.get("reroutes"):
                    done.add(k)     # out of time, stalled, or already within min_excess of the lower bound
            visits_f.write(json.dumps({"t": time.time(), "case": k, **st}) + "\n")
            visits_f.flush()
        elif kind == "init":
            stats[msg[1]].update(msg[2])
            stats[msg[1]]["best"] = msg[2]["initial_delay"]
        elif kind == "log":
            log(f"{msg[1] if isinstance(msg[1], str) else 'worker ' + str(msg[1])}: {msg[2]}")
        elif kind == "error":
            log(f"{msg[1]} crashed:\n{msg[2]}")
        elif kind == "exit":
            exited.add(msg[1])

    last_log = time.time()
    while True:
        try:
            while True:
                handle(req_q.get(timeout=0.2))
        except queue.Empty:
            pass
        now = time.time()
        # a crashed worker's case is released and not retried
        for wid, p in enumerate(procs):
            if p is not None and not p.is_alive():
                k = held[wid]
                if k is not None:
                    busy.discard(k)
                    done.add(k)
                    log(f"{k}: worker {wid} died (exit code {p.exitcode}); case ends here")
                procs[wid], held[wid] = None, None
                exited.add(wid)
        for f in os.listdir(os.path.join(work, "records")):     # training records are not needed here
            try:
                os.remove(os.path.join(work, "records", f))
            except OSError:
                pass
        if not stopping and (len(done) >= len(cases) or now >= deadline or os.path.exists(cfg["stop_flag"])):
            log("all cases finished" if len(done) >= len(cases) else
                ("wall-time cap reached" if now >= deadline else "STOP file found"))
            stopping = True
            deadline = min(deadline, now)
            open(cfg["stop_flag"], "w").close()
        if stopping and (len(exited) >= W or now > deadline + 1800):
            break
        if now - last_log > 120:
            last_log = now
            save()
            log(f"{len(done)}/{len(cases)} cases finished, busy={len(busy)} | {summary()}")
    infer_q.put(None)
    for p in procs:
        if p is not None:
            p.join(timeout=60)
    save()
    log(f"finished in {(time.time() - t_start) / 60:.1f} min | {summary()}")
    export(work, a.out)


if __name__ == "__main__":
    main()
