"""Worker process: improves cases and produces training records.

A worker asks the dispatcher (run.py) for a case, loads its current routing (or builds
the initial routing on first visit), runs fast greedy rerouting until it stalls, then a
number of search-guided reroutes proportional to the case's remaining excess. It works on
several cases at once so their network calls can be batched on the GPU. Until the learner
publishes a trained network, the search uses the network-free completion evaluator; after
that, each visit uses the network except a `control_share` of visits, which stay
network-free so the two can be compared over the same hours.

Generated (training-only) cases, keys "gen/<name>", start from their reference solution
instead of an initial routing, skip the greedy phase (its reroutes leave no training
records) and keep their state under <work>/gen_state/, which is not synced to Drive.

Per challenge case it keeps, under <work>/state/:  <key>.npz (current routing + version) and
<key>.replay.jsonl (every accepted reroute, for exact replay from the initial routing).
Training records go to <work>/records/ as .npz shards.
"""
from __future__ import annotations

import json
import math
import os
import queue
import random
import time
import traceback
from collections import OrderedDict
from typing import Dict, List, Optional

import numpy as np

from .features import decision_index, encode
from .lns import Improver, greedy_until_stall
from .problem import Problem
from .search import Leaf, Search, completion_evaluator, score_total
from .tiers import all_cases

ALL = {f"{t}/{n}": (t, n, p, b) for (t, n, p, b) in all_cases()}


def is_gen(key: str) -> bool:
    return key.startswith("gen/")


def key_file(work: str, key: str, suffix: str) -> str:
    return os.path.join(work, "gen_state" if is_gen(key) else "state", key.replace("/", "__") + suffix)


def save_routing(work: str, key: str, P: Problem, version: int) -> None:
    snap = P.snapshot()
    offs = np.cumsum([0] + [len(e) for e in snap]).astype(np.int64)
    edges = np.array([ab for e in snap for ab in e], dtype=np.int32).reshape(-1, 2)
    path = key_file(work, key, ".npz")
    tmp = path + ".tmp.npz"
    np.savez(tmp, version=version, delay=P.router.total_delay(), offsets=offs, edges=edges)
    os.replace(tmp, path)


def load_routing(work: str, key: str):
    path = key_file(work, key, ".npz")
    if not os.path.exists(path):
        return None
    z = np.load(path)
    offs, edges = z["offsets"], z["edges"]
    snap = [[(int(a), int(b)) for a, b in edges[offs[i]:offs[i + 1]]] for i in range(len(offs) - 1)]
    return int(z["version"]), snap


def stored_version(work: str, key: str) -> int:
    path = key_file(work, key, ".npz")
    if not os.path.exists(path):
        return -1
    return int(np.load(path)["version"])


class NetEvaluator:
    """Batched network evaluation of reroute states; reloads weights when the learner publishes."""

    def __init__(self, path: str, device):
        self.path, self.device = path, device
        self.model, self.mtime, self.ready = None, 0.0, False

    def refresh(self) -> None:
        import torch
        from .model import load_model
        try:
            mt = os.path.getmtime(self.path)
        except OSError:
            return
        if mt == self.mtime:
            return
        try:
            m, ck = load_model(self.path, self.device)
        except Exception:
            return               # being written; try again next time
        m.eval()
        self.model, self.mtime, self.ready = m, mt, bool(ck.get("net_ready", False))

    def __call__(self, reqs) -> List:
        """reqs: live states (encoded now) or Leafs (encoded earlier)."""
        import torch
        from .model import collate
        feats = [r.feats if isinstance(r, Leaf) else encode(r.problem, r) for r in reqs]
        states = [r.st if isinstance(r, Leaf) else r for r in reqs]
        b = collate(feats, self.device)
        with torch.no_grad(), torch.autocast(self.device.type, dtype=torch.bfloat16,
                                             enabled=self.device.type == "cuda"):
            out = self.model(b["maps"], b["nodes"], b["rois"], b["glob"], b["legal"], b["ld"], b["pairs"])
        logits = out["logits"].float().cpu().numpy()
        y = (out["y"].float() * self.model.y_sigma + self.model.y_mu).cpu().numpy()
        fb = torch.sigmoid(out["blocked"].float()).cpu().numpy()
        price = (torch.sigmoid(out["future"].float()) * 255).round().to(torch.uint8).cpu().numpy()
        return to_results(states, logits, y, fb, price)


def to_results(states, logits, y, fb, price) -> List:
    """Network outputs -> (logits over each state's legal decisions, q, price canvas) per state."""
    res = []
    for i, st in enumerate(states):
        lg = logits[i][decision_index(st)]
        est = st.rerouted_delay + st.ideal_pending * math.exp(float(np.clip(y[i], -5, 5)))
        q = (1 - fb[i]) * score_total(st, est) - fb[i]
        res.append((lg, float(q), price[i]))
    return res


class RemoteEvaluator:
    """Evaluates states in the shared inference process (inference.py)."""

    def __init__(self, wid: int, infer_q, resp_q, ready_flag: str):
        self.wid, self.q, self.resp_q, self.flag = wid, infer_q, resp_q, ready_flag
        self.rid = 0
        self.ready = False

    def refresh(self) -> None:
        if not self.ready:
            self.ready = os.path.exists(self.flag)

    def __call__(self, leaves: List[Leaf]) -> Optional[List]:
        """Results, or None if the inference process does not answer (this worker then stops
        using the network)."""
        states = [lf.st for lf in leaves]
        self.rid += 1
        self.q.put(("eval", self.wid, self.rid, [lf.feats for lf in leaves]))
        while True:
            try:
                rid, logits, y, fb, price = self.resp_q.get(timeout=120)
            except queue.Empty:
                self.ready, self.flag = False, "/nonexistent"
                return None
            if rid == self.rid:
                break
        if logits is None:
            return None
        return to_results(states, logits, y, fb, price)


def neutral_results(leaves: List[Leaf]) -> List:
    """Stand-in when the network cannot answer: uniform prior, ideal completion of the pending nets."""
    return [(np.zeros(len(lf.st.legal_decisions())), score_total(lf.st, lf.st.rerouted_delay + lf.st.ideal_pending), None) for lf in leaves]


class Slot:
    """One case being worked on: greedy phase done, now driving search-guided reroutes."""

    def __init__(self, key, P, imp, search, n_search, cfg, t0, start_delay, version, use_net=False):
        self.key, self.P, self.imp, self.search = key, P, imp, search
        self.use_net = use_net
        self.n_left, self.cfg = n_search, cfg
        self.t0, self.start_delay, self.version = t0, start_delay, version
        self.records: List[Dict] = []
        self.stats = {"reroutes": 0, "better": 0, "equal": 0, "worse": 0, "blocked": 0}
        self.gen = None
        self.pending_req = None

    def next_request(self, reply=None):
        """Advance until the search needs evaluations (returns the request list) or the visit ends (None)."""
        while True:
            if self.gen is None:
                if self.n_left <= 0 or self.cfg.get("stop_flag") and os.path.exists(self.cfg["stop_flag"]):
                    return None
                seed = self.imp.pick_seed()
                if seed is None:
                    return None
                self.imp.max_nets = self.imp.rng.randint(self.cfg["k_min"], self.cfg["k_max"])
                self.gen = self.search.reroute(self.imp.neighbourhood(seed))
                self.n_left -= 1
                reply = None
            try:
                return self.gen.send(reply) if reply is not None else next(self.gen)
            except StopIteration as stop:
                ok, old, new, recs = stop.value
                self.records += recs
                self.stats["reroutes"] += 1
                self.stats["blocked" if new < 0 else ("better" if new < old else ("equal" if new == old else "worse"))] += 1
                self.gen, reply = None, None


def worker_main(wid: int, cfg: Dict, req_q, resp_q, infer_q, infer_resp_q) -> None:
    try:
        _worker(wid, cfg, req_q, resp_q, infer_q, infer_resp_q)
    except Exception:
        req_q.put(("error", wid, traceback.format_exc()))
        raise


def _worker(wid: int, cfg: Dict, req_q, resp_q, infer_q, infer_resp_q) -> None:
    import torch
    torch.set_num_threads(1)
    device = torch.device("cuda" if torch.cuda.is_available() and not cfg.get("cpu") else "cpu")
    work = cfg["work"]
    rng = random.Random(cfg["seed"] * 1000 + wid)
    net = RemoteEvaluator(wid, infer_q, infer_resp_q, os.path.join(work, "net_ready"))
    cache: "OrderedDict[str, Problem]" = OrderedDict()
    slots: List[Slot] = []
    shard = 0
    stop_flag = cfg["stop_flag"]
    parent = os.getppid()

    def orphaned() -> bool:
        """The run process died (killed or crashed): this worker must not keep writing."""
        return os.getppid() != parent

    def acquire() -> Optional[Slot]:
        req_q.put(("get", wid))
        while True:
            try:
                msg = resp_q.get(timeout=10)
                break
            except queue.Empty:
                if orphaned():
                    raise SystemExit(0)
        if msg is None:
            return None
        key, hint = msg
        t0 = time.time()
        gen = is_gen(key)
        if gen:
            path, ref = cfg["gen_cases"][key]
            base = None
        else:
            tier, name, path, base = ALL[key]
        P = cache.get(key)
        ver = stored_version(work, key)
        if P is None or getattr(P, "_version", None) != ver:
            if gen:
                # a generated file that does not load is skipped for the rest of the run
                try:
                    if P is None:
                        P = Problem.load(path)
                    if load_routing(work, key) is None:
                        P.restore_submission(ref)
                except Exception as e:
                    req_q.put(("log", wid, f"{key}: skipped, could not load ({type(e).__name__}: {e})"))
                    req_q.put(("done", key, {"seconds": time.time() - t0, "failed": True}))
                    cache.pop(key, None)
                    return acquire()
            if P is None:
                P = Problem.load(path)
            stored = load_routing(work, key)
            if stored is None:
                if gen:
                    base = P.router.total_delay()
                else:
                    big = P.W * P.H * P.L > 300_000
                    ok = P.initial_route(schedules=[P.SCHEDULES[0]] if big else None,
                                         orders=None)
                    if not ok:
                        req_q.put(("log", wid, f"{key}: initial routing failed"))
                        req_q.put(("done", key, {"seconds": time.time() - t0, "failed": True}))
                        return acquire()
                ver = 0
                save_routing(work, key, P, ver)
                init = {"initial_delay": P.router.total_delay(), "lower_bound": P.lower_bound, "baseline": base}
                req_q.put(("init", key, init))
            else:
                ver, snap = stored
                P.restore(snap)
            P._version = ver
        cache[key] = P
        cache.move_to_end(key)
        while len(cache) > cfg["cache_cases"]:
            cache.popitem(last=False)
        imp = Improver(P, random.Random(rng.randrange(2 ** 32)))
        start = P.router.total_delay()
        if not gen:
            greedy_until_stall(imp, stall_tries=cfg["stall_tries"], max_seconds=cfg["greedy_seconds"])
        frac = (P.router.total_delay() - P.lower_bound) / max(1, P.lower_bound)
        n_search = 0 if frac < cfg["min_excess"] else int(min(cfg["max_search"], max(4, round(P.N * frac))))
        net.refresh()
        use_net = net.ready and rng.random() >= cfg["control_share"]
        search = Search(imp, n_full=cfg["n_full"], n_fast=cfg["n_fast"], p_full=cfg["p_full"],
                        gumbel_scale=cfg["gumbel_scale"], max_considered=cfg.get("considered", 16),
                        max_considered_fast=cfg.get("considered_fast"),
                        rng=random.Random(rng.randrange(2 ** 32)),
                        defer=use_net, explore_fast=cfg["explore_fast"])
        scfg = dict(cfg, stop_flag=stop_flag, k_max=hint.get("k_max", cfg["k_max"]))
        s = Slot(key, P, imp, search, n_search, scfg, t0, start, ver, use_net)
        s.pending_req = s.next_request()
        return s

    def release(s: Slot) -> None:
        nonlocal shard
        P, key = s.P, s.key
        delay = P.router.total_delay()
        if s.imp.log:
            ver = s.version + 1
            save_routing(work, key, P, ver)
            P._version = ver
            if not is_gen(key):          # generated cases are never exported: no replay log
                with open(key_file(work, key, ".replay.jsonl"), "a") as f:
                    for e in s.imp.log:
                        f.write(json.dumps(e) + "\n")
            s.imp.log.clear()
        if s.records:
            out = {k: np.stack([r[k] for r in s.records]) for k in s.records[0]}
            path = os.path.join(work, "records", f"w{wid:02d}_{shard:06d}.npz")
            np.savez(path + ".tmp.npz", **out)
            os.replace(path + ".tmp.npz", path)
            shard += 1
        stats = dict(s.stats, seconds=time.time() - s.t0, start=s.start_delay, delay=delay,
                     records=len(s.records), lower_bound=P.lower_bound, network=s.use_net,
                     k_max=s.cfg["k_max"])
        req_q.put(("done", key, stats))

    completion = {}

    def evaluate(slots: List[Slot]) -> List[List]:
        """Answer every slot's pending requests: one inference call for all network slots."""
        out: List[Optional[List]] = [None] * len(slots)
        net_i = [i for i, s in enumerate(slots) if s.use_net]
        if net_i:
            leaves = [lf for i in net_i for lf in slots[i].pending_req]
            res = net(leaves) if net.ready else None
            if res is None:
                res = neutral_results(leaves)
            j = 0
            for i in net_i:
                k = len(slots[i].pending_req)
                out[i] = res[j:j + k]
                j += k
        for i, s in enumerate(slots):
            if not s.use_net:
                ev = completion.setdefault(id(s.imp), completion_evaluator(s.imp))
                out[i] = [ev(st) for st in s.pending_req]
        return out

    while not os.path.exists(stop_flag):
        if orphaned():
            return
        net.refresh()
        while len(slots) < cfg["concurrency"] and not os.path.exists(stop_flag):
            s = acquire()
            if s is None:
                break
            if s.pending_req is None:
                release(s)
            else:
                slots.append(s)
        if not slots:
            time.sleep(1.0)
            continue
        keep = []
        for s, r in zip(slots, evaluate(slots)):
            s.pending_req = s.next_request(r)
            if s.pending_req is None:
                completion.pop(id(s.imp), None)
                release(s)
            else:
                keep.append(s)
        slots = keep
    # stop requested: finish the reroutes in flight, start no new ones
    for s in slots:
        s.n_left = 0
    while slots:
        keep = []
        for s, r in zip(slots, evaluate(slots)):
            s.pending_req = s.next_request(r)
            if s.pending_req is None:
                release(s)
            else:
                keep.append(s)
        slots = keep
    req_q.put(("exit", wid))
