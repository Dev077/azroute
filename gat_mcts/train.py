"""Train the GAT policy/value network: a warm start, then AlphaZero-style self-play.

    python -m gat_mcts.train --data gat_mcts_data/train --out runs/gat_mcts --max-hours 4

1. Warm start (skipped on --resume): episodes without search that follow the
   negotiated route for each net but deviate at random with probability eps. The
   policy learns to imitate the legal teacher; the value head learns, from the
   real challenge score of each perturbed episode, which deviations fail (0) and
   which beat the baseline (> 1).
2. Self-play: every iteration plays --episodes MCTS episodes on random pool
   instances (spread over --workers processes), adds (state graph, MCTS visit
   distribution, final score) to the replay buffer, and trains on it.

Losses: policy cross-entropy to the visit distribution, MSE of the value head
to the episode's challenge score (baseline_total / total_delay, 0 if it failed),
and BCE of the legal head. Stops at --iters or when --max-hours would be
exceeded, always leaving a checkpoint (latest.pt) and a JSON-lines log in --out.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import tempfile
import time
from collections import deque

import torch
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.utils import softmax as segment_softmax

from m3d.suite import load_manifest

from .data import TIER_DIRS, load_pool, tier_dir
from .model import RouteGAT, load_checkpoint, save_checkpoint
from .workers import Runner, run_episode, solve_case


def train_steps(model, opt, buffer, steps, batch_size, device, value_w=1.0, legal_w=0.5):
    model.train()
    tot = {"loss": 0.0, "policy": 0.0, "value": 0.0, "legal": 0.0}
    for _ in range(steps):
        batch = Batch.from_data_list(random.sample(buffer, min(batch_size, len(buffer)))).to(device)
        logits, value, legal_logit, cb = model(batch)
        logp = torch.log(segment_softmax(logits, cb).clamp_min(1e-12))
        B = batch.g.size(0)
        l_pol = -(batch["cand"].pi * logp).sum() / B
        l_val = F.mse_loss(value, batch.z)
        l_leg = F.binary_cross_entropy_with_logits(legal_logit, batch.legal)
        loss = l_pol + value_w * l_val + legal_w * l_leg
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        for k, v in (("loss", loss), ("policy", l_pol), ("value", l_val), ("legal", l_leg)):
            tot[k] += float(v)
    return {k: round(v / max(1, steps), 5) for k, v in tot.items()}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", default="gat_mcts_data/train", help="pool from gat_mcts.gen_data")
    ap.add_argument("--out", default="runs/gat_mcts", help="checkpoints + log (e.g. on Google Drive)")
    ap.add_argument("--resume", default=None, help="checkpoint to resume from (skips warm start)")
    ap.add_argument("--max-hours", type=float, default=0.0,
                    help="training budget in hours, summed over every session of this --out "
                         "(resumes included); stops before an iteration would exceed it (0 = none)")
    # model (defaults: ~3.0M parameters; hidden must be divisible by heads)
    ap.add_argument("--hidden", type=int, default=152)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--layers", type=int, default=6)
    # environment / search
    ap.add_argument("--frontier", type=int, default=4, help="nets MCTS may choose between")
    ap.add_argument("--rip-limit", type=int, default=8)
    ap.add_argument("--sims", type=int, default=16, help="MCTS simulations per move")
    ap.add_argument("--c-puct", type=float, default=1.5)
    ap.add_argument("--temp-moves", type=float, default=0.3,
                    help="fraction of an episode that samples moves (then greedy)")
    ap.add_argument("--max-nets", type=int, default=80, help="skip pool instances larger than this")
    # warm start
    ap.add_argument("--warm-episodes", type=int, default=150)
    ap.add_argument("--warm-eps", type=float, default=0.08,
                    help="max random-deviation rate in warm-start episodes")
    ap.add_argument("--warm-steps", type=int, default=1500)
    # self-play
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--episodes", type=int, default=6, help="self-play episodes per iteration")
    ap.add_argument("--train-steps", type=int, default=150,
                    help="gradient steps per iteration (capped at ~4 passes over the buffer)")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--buffer", type=int, default=15000, help="replay buffer size (~150 KB/sample)")
    # evaluation on a real tier
    ap.add_argument("--eval-tier", default="hard", choices=list(TIER_DIRS))
    ap.add_argument("--eval-every", type=int, default=10, help="iterations (0 = never)")
    ap.add_argument("--eval-cases", type=int, default=2)
    ap.add_argument("--eval-sims", type=int, default=16)
    ap.add_argument("--eval-cleanup", type=float, default=10.0, help="cleanup seconds per case")
    # hardware
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                    help="training device")
    ap.add_argument("--selfplay-device", default=None,
                    help="device for the network during search (default: same as --device)")
    ap.add_argument("--workers", type=int, default=1, help="parallel self-play processes")
    ap.add_argument("--threads", type=int, default=2, help="torch CPU threads in the main process")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    args.selfplay_device = args.selfplay_device or args.device

    t_start = time.time()
    budget = args.max_hours * 3600 if args.max_hours > 0 else float("inf")
    budget_path = os.path.join(args.out, "budget.json")
    prev_s = 0.0  # training seconds spent by earlier sessions of this run
    if args.resume and os.path.exists(budget_path):
        with open(budget_path) as fh:
            prev_s = json.load(fh)["seconds"]

    def spent():
        return prev_s + time.time() - t_start

    def record_spent():
        with open(budget_path, "w") as fh:
            json.dump({"seconds": round(spent(), 1)}, fh)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    os.makedirs(args.out, exist_ok=True)
    log_path = os.path.join(args.out, "log.jsonl")
    scratch = tempfile.mkdtemp(prefix="gat_mcts_")
    actor_path = os.path.join(scratch, "actor.pt")  # weights handed to the workers

    def log(msg, **rec):
        print(msg, flush=True)
        if rec:
            with open(log_path, "a") as fh:
                fh.write(json.dumps({"time": time.time(), **rec}) + "\n")

    pool = [e for e in load_pool(args.data) if e["n_nets"] <= args.max_nets]
    if not pool:
        raise SystemExit(f"no instances with <= {args.max_nets} nets in {args.data}")
    log(f"pool: {len(pool)} instances from {args.data}; train device {args.device}, "
        f"self-play device {args.selfplay_device}, {args.workers} worker(s)")

    def task(kind, **kw):
        return {"kind": kind, "data": args.data, "entry": random.choice(pool),
                "seed": random.randrange(1 << 30), "frontier": args.frontier,
                "rip_limit": args.rip_limit, **kw}

    start_iter = 0
    if args.resume:
        model, ck = load_checkpoint(args.resume, args.device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
        if ck.get("optimizer"):
            opt.load_state_dict(ck["optimizer"])
        start_iter = ck["extra"].get("iter", -1) + 1
        log(f"resumed {args.resume} at iteration {start_iter} (replay buffer starts empty)")
    else:
        model = RouteGAT(args.hidden, args.heads, args.layers).to(args.device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    log(f"model: {sum(p.numel() for p in model.parameters()):,} parameters")
    buffer = deque(maxlen=args.buffer)
    runner = Runner(args.workers)
    try:
        # ---- 1. warm start ----------------------------------------------------
        if not args.resume and args.warm_episodes > 0:
            log(f"warm start: {args.warm_episodes} guided episodes")
            t0 = time.time()
            tasks = [task("warm", eps=random.uniform(0.0, args.warm_eps))
                     for _ in range(args.warm_episodes)]
            for k, r in enumerate(runner.imap(run_episode, tasks)):
                buffer.extend(r["samples"])
                log(f"  warm {k + 1}/{args.warm_episodes} {r['instance']} eps {r['eps']:.3f}: "
                    f"score {r['score']:.4f} legal {r['legal']} ({r['seconds']:.1f}s, buffer {len(buffer)})",
                    kind="warm", instance=r["instance"], eps=r["eps"], score=r["score"],
                    legal=r["legal"], seconds=r["seconds"])
            stats = train_steps(model, opt, list(buffer), args.warm_steps, args.batch, args.device)
            log(f"warm start done in {time.time() - t0:.0f}s: {stats}", kind="warm_train", **stats)
            save_checkpoint(os.path.join(args.out, "warm.pt"), model, opt, {"iter": -1})
            save_checkpoint(os.path.join(args.out, "latest.pt"), model, opt, {"iter": -1})
            record_spent()

        # ---- 2. self-play -----------------------------------------------------
        last_iter_s = 0.0
        for it in range(start_iter, args.iters):
            if spent() + 1.2 * last_iter_s > budget:
                log(f"stopping: --max-hours {args.max_hours} reached "
                    f"({spent() / 3600:.2f} h used, next iteration ~{last_iter_s / 60:.0f} min)")
                break
            t0 = time.time()
            save_checkpoint(actor_path, model)
            tasks = [task("mcts", model_path=actor_path, model_tag=it, device=args.selfplay_device,
                          sims=args.sims, c_puct=args.c_puct, temp_moves=args.temp_moves)
                     for _ in range(args.episodes)]
            scores = []
            for k, r in enumerate(runner.imap(run_episode, tasks)):
                buffer.extend(r["samples"])
                scores.append(r["score"])
                log(f"  iter {it} ep {k}: {r['instance']} ({r['n']} nets) score {r['score']:.4f} "
                    f"legal {r['legal']} moves {r['moves']} ({r['seconds']:.0f}s)",
                    kind="selfplay", iter=it, instance=r["instance"], score=r["score"],
                    legal=r["legal"], moves=r["moves"], seconds=r["seconds"])
            steps = min(args.train_steps, max(1, 4 * len(buffer) // args.batch))
            stats = train_steps(model, opt, list(buffer), steps, args.batch, args.device)
            mean = sum(scores) / len(scores)
            last_iter_s = time.time() - t0
            log(f"iter {it}: mean self-play score {mean:.4f}, buffer {len(buffer)}, {steps} steps, "
                f"{stats}, {last_iter_s:.0f}s (total {spent() / 3600:.2f} h)",
                kind="iter", iter=it, mean_score=mean, buffer=len(buffer), steps=steps,
                seconds=last_iter_s, **stats)
            save_checkpoint(os.path.join(args.out, "latest.pt"), model, opt, {"iter": it})
            record_spent()

            if args.eval_every and (it + 1) % args.eval_every == 0:
                t1 = time.time()
                save_checkpoint(os.path.join(args.out, f"iter_{it:04d}.pt"), model, opt, {"iter": it})
                save_checkpoint(actor_path, model)
                suite_dir = tier_dir(args.eval_tier)
                cases = load_manifest(suite_dir)["cases"][:args.eval_cases]
                etasks = [{"suite_dir": suite_dir, "case": c,
                           "cache_dir": os.path.join(args.out, "negotiated_cache"),
                           "model_path": actor_path, "model_tag": f"eval{it}",
                           "device": args.selfplay_device, "evaluator": "net",
                           "sims": args.eval_sims, "c_puct": args.c_puct,
                           "frontier": args.frontier, "rip_limit": args.rip_limit,
                           "cleanup_seconds": args.eval_cleanup, "out": None} for c in cases]
                res = sorted(runner.imap(solve_case, etasks), key=lambda r: r["case"])
                for r in res:
                    log(f"    eval {r['case']}: search {r['search']:.4f}  final {r['final']:.4f}  "
                        f"({r['path']}, {r['seconds']:.0f}s)")
                log(f"  eval {args.eval_tier}: mean search {sum(r['search'] for r in res) / len(res):.4f} "
                    f"final {sum(r['final'] for r in res) / len(res):.4f}",
                    kind="eval", iter=it, cases=res)
                last_iter_s += time.time() - t1
                record_spent()
    finally:
        runner.close()
    log(f"training finished: {spent() / 3600:.2f} h in total, "
        f"checkpoint {os.path.join(args.out, 'latest.pt')}")


if __name__ == "__main__":
    main()
