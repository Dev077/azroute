"""Inference process: the only process that runs the network for the search.

Every worker sends its encoded states here and waits for the results. Whatever requests
are pending are evaluated together in one batch, so the GPU sees one process with large
batches instead of one small-batch process per worker (many processes sharing a GPU spend
most of their time switching between each other). Weights are reloaded whenever the
learner publishes a new model.pt.

Protocol (multiprocessing queues):
  request   ("eval", worker_id, request_id, [encoded features, ...])  -> infer_q
  response  (request_id, logits (n, N*M), y (n,), p_blocked (n,), price (n, L, 64, 64) uint8)
            -> infer_resp_qs[worker_id]
  shutdown  None -> infer_q
"""
from __future__ import annotations

import os
import queue
import time
import traceback
from typing import Dict


MAX_BATCH = 768          # workers send a whole halving round per request
GATHER_SECONDS = 0.002      # after the first request, wait this long for more to join the batch
RELOAD_SECONDS = 2.0


def inference_main(cfg: Dict, infer_q, infer_resp_qs, log_q) -> None:
    try:
        _serve(cfg, infer_q, infer_resp_qs, log_q)
    except Exception:
        log_q.put(("error", "inference", traceback.format_exc()))
        raise


def _serve(cfg: Dict, infer_q, infer_resp_qs, log_q) -> None:
    import torch
    from network.model import collate, load_model
    torch.set_num_threads(2)
    device = torch.device("cuda" if torch.cuda.is_available() and not cfg.get("cpu") else "cpu")
    path = os.path.join(cfg["work"], "model.pt")
    parent = os.getppid()
    model, mtime, last_check = None, 0.0, 0.0
    served = batches = 0
    t_report = time.time()

    def reload() -> None:
        nonlocal model, mtime
        try:
            mt = os.path.getmtime(path)
        except OSError:
            return
        if mt == mtime:
            return
        try:
            m, _ = load_model(path, device)
        except Exception:
            return               # being written; next check picks it up
        model, mtime = m.eval(), mt

    while True:
        try:
            first = infer_q.get(timeout=1.0)
        except queue.Empty:
            if os.getppid() != parent:
                return
            continue
        if first is None:
            return
        reqs = [first]
        deadline = time.time() + GATHER_SECONDS
        n_states = len(first[3])
        while n_states < MAX_BATCH:
            try:
                r = infer_q.get(timeout=max(0.0, deadline - time.time()))
            except queue.Empty:
                break
            if r is None:
                infer_q.put(None)      # handle the shutdown after this batch
                break
            reqs.append(r)
            n_states += len(r[3])
        now = time.time()
        if model is None or now - last_check > RELOAD_SECONDS:
            reload()
            last_check = now
        if model is None:              # no published network yet: should not happen, workers check first
            time.sleep(0.5)
            reload()
            if model is None:
                for _, wid, rid, feats in reqs:
                    infer_resp_qs[wid].put((rid, None, None, None, None))
                continue
        feats = [f for r in reqs for f in r[3]]
        b = collate(feats, device)
        with torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16,
                                                    enabled=device.type == "cuda"):
            out = model(b["maps"], b["nodes"], b["rois"], b["glob"], b["legal"], b["ld"], b["pairs"])
            logits = out["logits"].float().cpu().numpy()
            y = (out["y"].float() * model.y_sigma + model.y_mu).cpu().numpy()
            fb = torch.sigmoid(out["blocked"].float()).cpu().numpy()
            price = (torch.sigmoid(out["future"].float()) * 255).round().to(torch.uint8).cpu().numpy()
        i = 0
        for _, wid, rid, fs in reqs:
            k = len(fs)
            infer_resp_qs[wid].put((rid, logits[i:i + k], y[i:i + k], fb[i:i + k], price[i:i + k]))
            i += k
        served += len(feats)
        batches += 1
        if now - t_report > 300:
            log_q.put(("log", "inference", f"{served / (now - t_report):.0f} evals/s, "
                                           f"mean batch {served / max(1, batches):.1f}"))
            served = batches = 0
            t_report = now
