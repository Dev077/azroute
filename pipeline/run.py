"""Train and solve: workers improve the challenge cases while the learner trains the network.

    python -m pipeline.run --hours 23.5 --work /content/work --drive /content/drive/MyDrive/azroute

Resumable: on start it restores <drive> into <work> (case routings, replay logs, model,
learner checkpoint, stats); every --sync-minutes it copies <work> back to <drive>. At the
deadline it stops the workers, syncs, and exports checked submission files.
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import queue
import random
import shutil
import time

import numpy as np

SYNC_ITEMS = ["state", "model.pt", "learner.pt", "case_stats.json", "metrics.jsonl", "visits.jsonl",
              "run_config.json", "run.log"]


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def sync(src: str, dst: str) -> None:
    """Copy run state from src to dst: files missing in dst or newer in src (never older over newer)."""
    if not dst:
        return
    os.makedirs(dst, exist_ok=True)
    for item in SYNC_ITEMS:
        s, d = os.path.join(src, item), os.path.join(dst, item)
        if os.path.isdir(s):
            os.makedirs(d, exist_ok=True)
            for f in os.listdir(s):
                if f.endswith(".tmp.npz") or f.endswith(".tmp"):
                    continue
                fs, fd = os.path.join(s, f), os.path.join(d, f)
                st = os.stat(fs)
                if not os.path.exists(fd) or st.st_mtime > os.stat(fd).st_mtime + 1:
                    shutil.copy2(fs, fd)
        elif os.path.exists(s) and (not os.path.exists(d) or os.stat(s).st_mtime > os.stat(d).st_mtime + 1):
            shutil.copy2(s, d + ".tmp")
            os.replace(d + ".tmp", d)


def auto_workers(requested: int, gpu_gb: float) -> int:
    """One process per spare CPU core, bounded by host RAM (~3 GB each) and GPU memory (~1 GB each,
    after reserving 45 % of the GPU for the learner)."""
    if requested > 0:
        return requested
    cpus = os.cpu_count() or 2
    try:
        import psutil
        ram_gb = psutil.virtual_memory().available / 2 ** 30
    except Exception:
        ram_gb = 12.0
    gpu_cap = int(gpu_gb * 0.55) if gpu_gb else 64
    return int(max(1, min(cpus - 4, 64, (ram_gb - 16) // 3, gpu_cap)))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="/content/work")
    ap.add_argument("--drive", default="", help="Drive folder for checkpoints/resume ('' = none)")
    ap.add_argument("--hours", type=float, default=None,
                    help="total wall time including export (default 23.5; on resume, the original deadline is kept)")
    ap.add_argument("--export-minutes", type=float, default=20.0, help="reserved at the end for stop + export")
    ap.add_argument("--tiers", default="all")
    ap.add_argument("--workers", type=int, default=0, help="0 = auto")
    ap.add_argument("--concurrency", type=int, default=0, help="cases per worker; 0 = auto")
    ap.add_argument("--batch", type=int, default=0, help="training batch; 0 = auto from GPU memory")
    ap.add_argument("--buffer", type=int, default=0, help="replay capacity; 0 = auto from RAM")
    ap.add_argument("--replay-ratio", type=float, default=16.0, help="max samples trained per record collected")
    ap.add_argument("--prio", type=float, default=0.5,
                    help="share of each batch drawn in proportion to the record's decision spread")
    ap.add_argument("--control-share", type=float, default=0.1,
                    help="share of visits that stay network-free after the handoff (for comparison)")
    ap.add_argument("--gumbel-scale", type=float, default=1.0,
                    help="exploration noise on the decision scores (higher = tries less likely decisions more)")
    ap.add_argument("--considered", type=int, default=32, help="candidate decisions sampled per full search")
    ap.add_argument("--considered-fast", type=int, default=16, help="candidate decisions sampled per fast search")
    ap.add_argument("--no-explore-fast", dest="explore_fast", action="store_false",
                    help="no Gumbel noise on fast (unrecorded) searches")
    ap.add_argument("--k-max", type=int, default=8, help="max nets per neighbourhood on a case that improves")
    ap.add_argument("--k-max-stalled", type=int, default=16,
                    help="max nets per neighbourhood once a case stops improving (+2 per visit without gain)")
    ap.add_argument("--polish-workers", type=int, default=-1,
                    help="workers that only polish scored cases (network-free, record-to-record acceptance, "
                         "no training records); -1 = half the workers, 0 = none")
    ap.add_argument("--polish-minutes", type=float, default=30.0, help="length of one polish visit")
    ap.add_argument("--polish-tau", type=float, default=0.003,
                    help="polish: a worse reroute is kept while the case total stays within this fraction of "
                         "the best total of the visit; shrinks linearly to 0 over the visit")
    ap.add_argument("--min-records", type=int, default=2000)
    ap.add_argument("--warmup-steps", type=int, default=1000, help="train steps before workers switch to the network")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--n-full", type=int, default=128, help="evaluations per recorded (full) decision")
    ap.add_argument("--n-fast", type=int, default=8)
    ap.add_argument("--p-full", type=float, default=0.25)
    ap.add_argument("--sync-minutes", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cpu", action="store_true", help="force CPU (local testing)")
    ap.add_argument("--smoke", action="store_true", help="tiny model, 2 hard cases, minutes-long run")
    a = ap.parse_args(argv)

    import torch
    from network.learner import ReplayBuffer, Trainer
    from network.model import RouteNet, load_model, save_model
    from routing.tiers import TIER_DIRS, all_cases

    if a.smoke:
        a.min_records, a.warmup_steps, a.n_full, a.n_fast = 50, 30, 16, 4
        a.tiers = a.tiers if a.tiers != "all" else "hard"

    work = os.path.abspath(a.work)
    for d in ("state", "records"):
        os.makedirs(os.path.join(work, d), exist_ok=True)
    if a.drive and os.path.isdir(a.drive):
        log(f"resuming from {a.drive}")
        sync(a.drive, work)
    for f in ("STOP", "net_ready"):
        if os.path.exists(os.path.join(work, f)):
            os.remove(os.path.join(work, f))
    prev_cfg_path = os.path.join(work, "run_config.json")
    prev_cfg = {}
    if a.hours is None and os.path.exists(prev_cfg_path):
        prev_cfg = json.load(open(prev_cfg_path))
    prev_deadline = prev_cfg.get("deadline_epoch")
    if prev_deadline and prev_deadline > time.time():
        deadline = prev_deadline
        start_epoch = prev_cfg["start_epoch"]      # keeps the LR schedule's position
        log(f"keeping the original deadline: {(deadline - time.time()) / 3600:.2f} h left")
    else:
        start_epoch = time.time()
        deadline = start_epoch + (a.hours or 23.5) * 3600 - a.export_minutes * 60

    tiers = list(TIER_DIRS) if a.tiers == "all" else a.tiers.split(",")
    cases = [f"{t}/{n}" for (t, n, _p, _b) in all_cases() if t in tiers]
    if a.smoke:
        cases = cases[:2]
    base = {f"{t}/{n}": b for (t, n, _p, b) in all_cases()}

    device = torch.device("cuda" if torch.cuda.is_available() and not a.cpu else "cpu")
    gpu_gb = torch.cuda.get_device_properties(0).total_memory / 2 ** 30 if device.type == "cuda" else 0.0
    W = auto_workers(a.workers, gpu_gb) if not a.smoke else 2
    C = a.concurrency or max(1, min(2, len(cases) // W))
    PW = min(W - 1, W // 2 if a.polish_workers < 0 else a.polish_workers) if W > 1 else 0
    if a.batch:
        batch = a.batch
    elif device.type == "cuda":
        # training activations are ~35-70 MB per sample for the default network
        batch = 512 if gpu_gb >= 80 else (256 if gpu_gb >= 40 else 128)
    else:
        batch = 32
    try:
        import psutil
        avail = psutil.virtual_memory().available
    except Exception:
        avail = 12 * 2 ** 30
    cap = a.buffer or int(min(1_000_000, max(20_000, 0.2 * avail / 30_000)))   # ~30 KB per record
    if a.smoke:
        cap = 5_000

    cfg = {"work": work, "seed": a.seed, "stop_flag": os.path.join(work, "STOP"), "cpu": a.cpu,
           "concurrency": C, "cache_cases": max(2, C + 1), "stall_tries": 200, "greedy_seconds": 15.0,
           "min_excess": 0.003, "max_search": 64 if not a.smoke else 3, "n_full": a.n_full,
           "n_fast": a.n_fast, "p_full": a.p_full, "gumbel_scale": a.gumbel_scale, "considered": a.considered,
           "considered_fast": a.considered_fast, "k_min": 3, "k_max": a.k_max,
           "control_share": a.control_share, "explore_fast": a.explore_fast,
           # polish visits: network-free search, 3-8 nets
           "polish_workers": PW, "polish_seconds": a.polish_minutes * 60, "polish_tau": a.polish_tau,
           "polish_search": dict(n_full=32, n_fast=8, p_full=0.25, considered=16, k_min=3, k_max=8)}
    run_cfg = dict(vars(a), deadline_epoch=deadline, start_epoch=start_epoch, workers=W, concurrency=C,
                   polish_workers=PW, batch=batch, buffer=cap, device=str(device),
                   cases=len(cases), gpu=torch.cuda.get_device_name(0) if device.type == "cuda" else None,
                   cpus=os.cpu_count())
    json.dump(run_cfg, open(os.path.join(work, "run_config.json"), "w"), indent=1)
    log(f"device={device} gpu={run_cfg['gpu']} cpus={os.cpu_count()} workers={W} (polish {PW}) concurrency={C} "
        f"batch={batch} buffer={cap} cases={len(cases)}")

    # -- learner -------------------------------------------------------------------
    model_path = os.path.join(work, "model.pt")
    learner_path = os.path.join(work, "learner.pt")
    layers = 6
    if os.path.exists(model_path):
        model, ck = load_model(model_path, device)
        log(f"loaded model ({ck.get('steps', 0)} steps)")
    else:
        model = RouteNet.preset("tiny" if a.smoke else "default", layers).to(device)
    trainer = Trainer(model, device, lr=a.lr, warmup=200 if a.smoke else 1000)
    if os.path.exists(learner_path):
        try:
            trainer.load_state_dict(torch.load(learner_path, map_location=device, weights_only=False))
        except Exception as e:
            log(f"learner checkpoint not loaded: {e}")
    buf = ReplayBuffer(cap, layers)
    rng = np.random.default_rng(a.seed)
    records_in = 0
    net_ready = trainer.steps >= a.warmup_steps
    steps0 = trainer.steps      # the replay buffer is not checkpointed: the replay ratio counts this session only

    ready_flag = os.path.join(work, "net_ready")

    def publish():
        save_model(model, model_path, {"net_ready": net_ready, "steps": trainer.steps})
        if net_ready and not os.path.exists(ready_flag):
            open(ready_flag, "w").close()      # workers switch to the inference process

    if net_ready:
        publish()

    stats_path = os.path.join(work, "case_stats.json")
    stats = json.load(open(stats_path)) if os.path.exists(stats_path) else {}
    meta = stats.setdefault("_meta", {"learner_seconds": 0.0})
    for k in cases:
        t, n = k.split("/")
        stats.setdefault(k, {"tier": t, "case": n, "baseline": base[k], "seconds": 0.0, "visits": 0,
                             "records": 0})

    # -- workers -------------------------------------------------------------------
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(v, "1")          # one core per worker; inherited by spawned workers
    ctx = mp.get_context("spawn")
    req_q = ctx.Queue()
    resp_qs = [ctx.Queue() for _ in range(W)]
    from pipeline.worker import worker_main
    infer_q = ctx.Queue()
    infer_resp_qs = [ctx.Queue() for _ in range(W)]
    from network.inference import inference_main
    infer_proc = ctx.Process(target=inference_main, args=(cfg, infer_q, infer_resp_qs, req_q), daemon=True)
    infer_proc.start()
    def spawn(i):
        p = ctx.Process(target=worker_main, args=(i, cfg, req_q, resp_qs[i], infer_q, infer_resp_qs[i]),
                        daemon=True)
        p.start()
        return p

    procs = [spawn(i) for i in range(W)]

    order: list = []
    busy: set = set()
    held = {i: set() for i in range(W)}     # cases each worker is on, released if it crashes
    bad: set = set()                        # cases skipped for the rest of the run
    crashes: dict = {}                      # case -> worker crashes while on it
    restarts = 0
    exited = 0
    metrics_f = open(os.path.join(work, "metrics.jsonl"), "a")
    visits_f = open(os.path.join(work, "visits.jsonl"), "a")
    last_sync = last_log = last_reap = time.time()
    train_time = 0.0
    learner_seconds0 = meta["learner_seconds"]
    stopping = False
    polish_log: list = []                   # relative gain of each polish visit since the last status line

    def next_case():
        nonlocal order
        for _ in range(2):
            for i, k in enumerate(order):
                if k not in busy and k not in bad:
                    order.pop(i)
                    return k
            order = order + random.Random(int(rng.integers(1 << 30))).sample(cases, len(cases))
        return None

    def next_polish():
        """The free scored case with the most expected gain: never-polished cases first, then by
        the recent relative gain per polish minute, growing with the time since the last polish."""
        now = time.time()
        best_k, best_p = None, -1.0
        for k in cases:
            if k in busy or k in bad:
                continue
            s = stats[k]
            if s.get("polish_rate") is None:
                p = float("inf")
            else:
                p = max(s["polish_rate"], 1e-7) * (1 + (now - s.get("polish_t", now)) / 3600)
            if p > best_p:
                best_k, best_p = k, p
        return best_k

    def handle(msg):
        nonlocal exited
        kind = msg[0]
        if kind == "get":
            wid = msg[1]
            polish = wid < PW
            k = None if stopping else (next_polish() if polish else next_case())
            if k is None:
                resp_qs[wid].put(None)
            else:
                busy.add(k)
                held[wid].add(k)
                stall = stats.get(k, {}).get("stall", 0)
                resp_qs[wid].put((k, {"k_max": min(a.k_max_stalled, a.k_max + 2 * stall) if stall else a.k_max,
                                      "polish": polish}))
        elif kind == "done":
            k, st = msg[1], msg[2]
            busy.discard(k)
            for h in held.values():
                h.discard(k)
            s = stats[k]
            if st.get("polish") and "delay" in st:
                gain = (st["start"] - st["delay"]) / max(1, st["start"])
                rate = gain / max(1.0, st.get("seconds", 0.0) / 60)
                s["polish_rate"] = rate if s.get("polish_rate") is None else 0.5 * s["polish_rate"] + 0.5 * rate
                s["polish_t"] = time.time()
                s["polish_visits"] = s.get("polish_visits", 0) + 1
                polish_log.append(gain)
            elif "delay" in st and "start" in st:
                s["stall"] = 0 if st["delay"] < st["start"] else s.get("stall", 0) + 1
            s["seconds"] += st.get("seconds", 0.0)
            s["visits"] += 1
            s["records"] += st.get("records", 0)
            if "delay" in st:
                s["best"] = st["delay"]
            visits_f.write(json.dumps({"t": time.time(), "case": k, **st}) + "\n")
            visits_f.flush()
        elif kind == "init":
            stats[msg[1]].update(msg[2])
            stats[msg[1]]["best"] = msg[2]["initial_delay"]
        elif kind == "log":
            log(f"{msg[1] if isinstance(msg[1], str) else 'worker ' + str(msg[1])}: {msg[2]}")
        elif kind == "error":
            if msg[1] == "inference":
                log(f"inference process crashed (workers fall back to network-free search):\n{msg[2]}")
            else:
                log(f"worker {msg[1]} crashed:\n{msg[2]}")    # reap() restarts it once it has exited
        elif kind == "exit":
            exited += 1

    def reap():
        """Restart workers that died (crash, or killed e.g. for memory), releasing their cases."""
        nonlocal restarts, exited
        for wid, p in enumerate(procs):
            if p is None or p.is_alive():
                continue
            for k in held[wid]:
                busy.discard(k)
                crashes[k] = crashes.get(k, 0) + 1
                if crashes[k] >= 2:
                    bad.add(k)
                    log(f"{k}: was being worked on when a worker died twice; skipped for the rest of the run")
            held[wid] = set()
            for q in (resp_qs[wid], infer_resp_qs[wid]):    # drop replies meant for the dead worker
                try:
                    while True:
                        q.get_nowait()
                except queue.Empty:
                    pass
            if restarts < 3 * W:
                restarts += 1
                procs[wid] = spawn(wid)
                log(f"worker {wid} died (exit code {p.exitcode}); restarted ({restarts} restarts so far)")
            else:
                procs[wid] = None
                exited += 1
                log(f"worker {wid} died (exit code {p.exitcode}); restart limit reached, not restarted")

    def ingest():
        nonlocal records_in
        rdir = os.path.join(work, "records")
        for f in sorted(os.listdir(rdir)):
            if not f.endswith(".npz") or f.endswith(".tmp.npz"):
                continue
            p = os.path.join(rdir, f)
            try:
                z = dict(np.load(p))
            except Exception:
                continue
            records_in += buf.add(z)
            os.remove(p)

    def summary() -> str:
        parts = []
        for t in tiers:
            rs = [s for k, s in stats.items() if not k.startswith("_") and s["tier"] == t and s.get("best")]
            if rs:
                gm = math.exp(sum(math.log(s["baseline"] / s["best"]) for s in rs) / len(rs))
                parts.append(f"{t}={gm:.4f}({len(rs)})")
        return " ".join(parts)

    def save_all():
        meta["learner_seconds"] = learner_seconds0 + train_time
        json.dump(stats, open(stats_path + ".tmp", "w"), indent=1)
        os.replace(stats_path + ".tmp", stats_path)
        publish()
        torch.save(trainer.state_dict(), learner_path + ".tmp")
        os.replace(learner_path + ".tmp", learner_path)

    t_start = start_epoch
    while True:
        now = time.time()
        if not stopping and (now >= deadline or os.path.exists(cfg["stop_flag"])):
            log("deadline reached: stopping workers" if now >= deadline else "STOP file found: stopping workers")
            stopping = True
            open(cfg["stop_flag"], "w").close()
            deadline = min(deadline, now)
        try:
            while True:
                handle(req_q.get_nowait())
        except queue.Empty:
            pass
        if stopping and (exited >= W or time.time() > deadline + a.export_minutes * 30):
            break
        ingest()
        if not stopping and now - last_reap > 10:
            last_reap = now
            reap()
        can_train = buf.n >= a.min_records and (trainer.steps - steps0) * batch < a.replay_ratio * records_in
        if can_train and not stopping:
            t0 = time.time()
            progress = (now - t_start) / max(1.0, deadline - t_start)
            m = trainer.step(buf.sample(batch, rng, a.prio), progress, int(rng.integers(16)))
            train_time += time.time() - t0
            if trainer.steps % 50 == 0:
                metrics_f.write(json.dumps({"t": now, "step": trainer.steps, "records": records_in,
                                            "buffer": buf.n, **m}) + "\n")
                metrics_f.flush()
            if not net_ready and trainer.steps >= a.warmup_steps:
                net_ready = True
                publish()
                log(f"network published to workers at step {trainer.steps}")
            elif net_ready and trainer.steps % 250 == 0:
                publish()
        else:
            time.sleep(0.05)
        if now - last_log > 300:
            last_log = now
            pol = (f" polish={len(polish_log)} visits, {sum(g > 0 for g in polish_log)} improved, "
                   f"mean {100 * sum(polish_log) / len(polish_log):.3f}%" if polish_log else "")
            polish_log.clear()
            log(f"steps={trainer.steps} records={records_in} buffer={buf.n} busy={len(busy)} "
                f"train_s={train_time:.0f}{pol} | {summary()}")
        if now - last_sync > a.sync_minutes * 60:
            last_sync = now
            save_all()
            sync(work, a.drive)
            log("synced to drive")
    for p in procs:
        if p is not None:
            p.join(timeout=60)
    infer_q.put(None)
    infer_proc.join(timeout=30)
    save_all()
    sync(work, a.drive)
    log(f"stopped. {summary()}")
    from pipeline.export import export
    out = os.path.join(a.drive or work, "export")
    export(work, out)
    log(f"exported submissions to {out}")


if __name__ == "__main__":
    main()
