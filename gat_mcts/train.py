"""Train the GAT policy/value network: a warm start, then AlphaZero-style self-play.

    python -m gat_mcts.train --data gat_mcts_data/train --out runs/gat_mcts --max-hours 4

1. Warm start (skipped on --resume): episodes without search that follow the
   negotiated route for each net but deviate at random with probability eps. The
   policy learns to imitate the legal teacher; the value head learns, from the
   real challenge score of each perturbed episode, which deviations fail (0) and
   which beat the baseline (> 1). A held-out set of episodes gives early stopping.
2. Self-play: every iteration plays --episodes MCTS episodes on random pool
   instances (spread over --workers processes), measures the loss on those new,
   unseen states, adds them to the replay buffer, and trains on it. With
   --workers > 1 the GPU trains *while* the workers play (not after), so neither
   waits for the other; the number of gradient steps is the same either way.

Losses: policy cross-entropy to the visit distribution, MSE of the value head
to the episode's challenge score (baseline_total / total_delay, 0 if it failed),
and BCE of the legal head.

Training stability:
  * exploding gradients: gradient-norm clipping; any step with a non-finite loss
    or gradient is skipped; if most of an iteration's steps are bad, the last
    good checkpoint is restored
  * vanishing gradients: residual + LayerNorm layers (model.py), a numerically
    stable log-softmax for the policy, learning-rate warmup
  * overfitting: dropout, decoupled weight decay (not on biases / norms), a cap
    on passes over the replay buffer per iteration, early stopping on held-out
    episodes in the warm start, a loss on unseen episodes every iteration, and
    best.pt (best evaluation score) kept separately from latest.pt
  * smooth end of training: cosine learning-rate decay over --max-hours (constant
    learning rate after warmup when there is no hour limit)

Stops at --iters, when --max-hours would be exceeded, or (with --plateau-evals)
when the evaluation score hasn't improved for that many evaluations, always
leaving a checkpoint (latest.pt) and a JSON-lines log in --out. Each iteration
logs GPU utilisation (sampled from nvidia-smi) and samples generated per hour.
"""
from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
import os
import random
import shutil
import subprocess
import tempfile
import threading
import time
from collections import deque

import torch
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.utils import scatter

from m3d.suite import load_manifest

from .data import TIER_DIRS, load_pool, tier_dir
from .model import RouteGAT, load_checkpoint, save_checkpoint
from .workers import Runner, run_episode, solve_case

VALUE_W, LEGAL_W = 1.0, 0.5   # loss = policy + VALUE_W * value + LEGAL_W * legal


def segment_log_softmax(logits, seg):
    """log-softmax within each graph's candidates (stable: no log of a clamped prob,
    so tiny probabilities still get gradient)."""
    n = int(seg.max()) + 1 if seg.numel() else 0
    m = scatter(logits.detach(), seg, dim=0, dim_size=n, reduce="max")
    z = logits - m[seg]
    lse = torch.log(scatter(z.exp(), seg, dim=0, dim_size=n, reduce="sum"))
    return z - lse[seg]


def batch_losses(model, samples, device):
    batch = Batch.from_data_list(samples).to(device)
    logits, value, legal_logit, cb = model(batch)
    B = batch.g.size(0)
    l_pol = -(batch["cand"].pi * segment_log_softmax(logits, cb)).sum() / B
    l_val = F.mse_loss(value, batch.z)
    l_leg = F.binary_cross_entropy_with_logits(legal_logit, batch.legal)
    return l_pol + VALUE_W * l_val + LEGAL_W * l_leg, l_pol, l_val, l_leg


@torch.no_grad()
def eval_losses(model, samples, device, batch_size=64):
    """Mean losses on ``samples`` without training (dropout off)."""
    model.eval()
    tot = {"loss": 0.0, "policy": 0.0, "value": 0.0, "legal": 0.0}
    n = 0
    for i in range(0, len(samples), batch_size):
        chunk = samples[i:i + batch_size]
        parts = batch_losses(model, chunk, device)
        for k, v in zip(tot, parts):
            tot[k] += float(v) * len(chunk)
        n += len(chunk)
    return {k: round(v / max(1, n), 5) for k, v in tot.items()}


def make_optimizer(model, lr, wd):
    """AdamW; weight decay on weight matrices only (not biases or LayerNorm)."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        (no_decay if p.ndim < 2 or "norm" in name else decay).append(p)
    return torch.optim.AdamW([{"params": decay, "weight_decay": wd},
                              {"params": no_decay, "weight_decay": 0.0}], lr=lr)


class Trainer:
    def __init__(self, model, opt, args, frac_fn, step=0, log=print):
        self.model, self.opt, self.args = model, opt, args
        self.frac_fn = frac_fn      # fraction of the time budget used (for the LR decay)
        self.step = step            # global gradient step, kept across resumes
        self.log = log

    def lr(self):
        a = self.args
        warm = min(1.0, (self.step + 1) / a.warmup_steps) if a.warmup_steps > 0 else 1.0
        frac = min(1.0, self.frac_fn())
        decay = a.min_lr_frac + (1 - a.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * frac))
        return a.lr * warm * decay

    def train(self, buffer, steps):
        a, model, opt = self.args, self.model, self.opt
        model.train()
        tot = {"loss": 0.0, "policy": 0.0, "value": 0.0, "legal": 0.0}
        norms, skipped, done = [], 0, 0
        for _ in range(steps):
            lr = self.lr()
            for g in opt.param_groups:
                g["lr"] = lr
            parts = batch_losses(model, random.sample(buffer, min(a.batch, len(buffer))), a.device)
            if not torch.isfinite(parts[0]):
                skipped += 1
                continue
            opt.zero_grad(set_to_none=True)
            parts[0].backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)
            if not torch.isfinite(gn):
                opt.zero_grad(set_to_none=True)
                skipped += 1
                continue
            opt.step()
            self.step += 1
            done += 1
            norms.append(float(gn))
            for k, v in zip(tot, parts):
                tot[k] += float(v)
        stats = {k: round(v / max(1, done), 5) for k, v in tot.items()}
        stats.update(grad_norm=round(sum(norms) / max(1, len(norms)), 4),
                     grad_norm_max=round(max(norms, default=0.0), 4),
                     skipped=skipped, lr=round(self.lr(), 7))
        return stats


class GPUMonitor:
    """Samples GPU utilisation and memory from nvidia-smi every few seconds in a
    background thread; ``take()`` returns the averages since the last call."""

    def __init__(self, every: float = 5.0):
        self.every = every
        self.samples = []
        self.lock = threading.Lock()
        self.ok = torch.cuda.is_available() and shutil.which("nvidia-smi") is not None
        if self.ok:
            threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while True:
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=10).stdout
                util, mem = out.strip().splitlines()[0].split(",")
                with self.lock:
                    self.samples.append((float(util), float(mem)))
            except Exception:
                pass
            time.sleep(self.every)

    def take(self):
        with self.lock:
            s, self.samples = self.samples, []
        if not s:
            return None
        return {"gpu_util": round(sum(u for u, _ in s) / len(s), 1),
                "gpu_mem_mb": round(max(m for _, m in s))}


def merge_stats(parts):
    """Combine Trainer.train() stats from several chunks (weighted by steps)."""
    parts = [(n, st) for n, st in parts if n > 0]
    if not parts:
        return {"loss": 0.0, "policy": 0.0, "value": 0.0, "legal": 0.0, "grad_norm": 0.0,
                "grad_norm_max": 0.0, "skipped": 0, "lr": 0.0}
    tot = sum(n for n, _ in parts)
    out = {k: round(sum(n * st[k] for n, st in parts) / tot, 5)
           for k in ("loss", "policy", "value", "legal", "grad_norm")}
    out["grad_norm_max"] = max(st["grad_norm_max"] for _, st in parts)
    out["skipped"] = sum(st["skipped"] for _, st in parts)
    out["lr"] = parts[-1][1]["lr"]
    return out


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
    ap.add_argument("--dropout", type=float, default=0.1)
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
    ap.add_argument("--warm-steps", type=int, default=1500, help="maximum; early stopping may end sooner")
    ap.add_argument("--val-frac", type=float, default=0.1, help="warm-start episodes held out")
    ap.add_argument("--patience", type=int, default=3,
                    help="warm start stops after this many checks (every 100 steps) without improvement")
    # self-play
    ap.add_argument("--iters", type=int, default=1000, help="self-play iterations (0 = no limit)")
    ap.add_argument("--episodes", type=int, default=6, help="self-play episodes per iteration")
    ap.add_argument("--train-steps", type=int, default=150,
                    help="gradient steps per iteration (capped at --max-passes over the buffer)")
    ap.add_argument("--max-passes", type=float, default=4.0,
                    help="cap: samples seen per iteration <= this x buffer size")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--buffer", type=int, default=15000, help="replay buffer size (~150 KB/sample)")
    # optimisation
    ap.add_argument("--lr", type=float, default=3e-4, help="peak learning rate")
    ap.add_argument("--warmup-steps", type=int, default=1000)
    ap.add_argument("--min-lr-frac", type=float, default=0.1,
                    help="learning rate decays (cosine, over --max-hours) to this fraction of --lr; "
                         "without --max-hours it stays at --lr")
    ap.add_argument("--wd", type=float, default=0.01, help="AdamW weight decay (weights only)")
    ap.add_argument("--clip", type=float, default=1.0, help="gradient-norm clip")
    # evaluation on a real tier
    ap.add_argument("--eval-tier", default="hard", choices=list(TIER_DIRS))
    ap.add_argument("--eval-every", type=int, default=10, help="iterations (0 = never)")
    ap.add_argument("--eval-cases", type=int, default=2)
    ap.add_argument("--eval-sims", type=int, default=16)
    ap.add_argument("--eval-cleanup", type=float, default=10.0, help="cleanup seconds per case")
    ap.add_argument("--plateau-evals", type=int, default=0,
                    help="stop when the eval search score hasn't improved for this many "
                         "evaluations (0 = never)")
    # hardware
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                    help="training device")
    ap.add_argument("--selfplay-device", default=None,
                    help="device for the network during search (default: same as --device)")
    ap.add_argument("--workers", type=int, default=1, help="parallel self-play processes")
    ap.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True,
                    help="torch.compile the network for training and search (--no-compile to turn off)")
    ap.add_argument("--threads", type=int, default=2, help="torch CPU threads in the main process")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    args.selfplay_device = args.selfplay_device or args.device

    t_start = time.time()
    budget = args.max_hours * 3600 if args.max_hours > 0 else float("inf")
    os.makedirs(args.out, exist_ok=True)
    budget_path = os.path.join(args.out, "budget.json")
    best_path = os.path.join(args.out, "best.json")
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

    start_iter, step = 0, 0
    if args.resume:
        model, ck = load_checkpoint(args.resume, args.device)
        opt = make_optimizer(model, args.lr, args.wd)
        try:
            if ck.get("optimizer"):
                opt.load_state_dict(ck["optimizer"])
        except ValueError:
            log("note: optimizer state in the checkpoint doesn't match; starting it fresh")
        start_iter = ck["extra"].get("iter", -1) + 1
        step = ck["extra"].get("step", 0)
        log(f"resumed {args.resume} at iteration {start_iter} (replay buffer starts empty)")
    else:
        model = RouteGAT(args.hidden, args.heads, args.layers, args.dropout).to(args.device)
        opt = make_optimizer(model, args.lr, args.wd)
    if args.compile:
        model.set_compiled(True)
    log(f"model: {sum(p.numel() for p in model.parameters()):,} parameters, config {model.config}, "
        f"torch.compile {'on' if args.compile else 'off'}")
    trainer = Trainer(model, opt, args, lambda: spent() / budget, step, log=log)
    best_search, best_iter, evals_since_best = -1.0, None, 0
    if args.resume and os.path.exists(best_path):
        with open(best_path) as fh:
            b = json.load(fh)
        best_search, best_iter = b["search"], b.get("iter")
        evals_since_best = b.get("evals_since_best", 0)
    gpu = GPUMonitor()

    def save_actor():
        """Weights for the workers, written atomically (a worker never reads half a file)."""
        tmp = actor_path + ".tmp"
        save_checkpoint(tmp, model)
        os.replace(tmp, actor_path)

    def checkpoint(path, it):
        save_checkpoint(path, model, opt, {"iter": it, "step": trainer.step})

    buffer = deque(maxlen=args.buffer)
    runner = Runner(args.workers)
    try:
        # ---- 1. warm start ----------------------------------------------------
        if not args.resume and args.warm_episodes > 0:
            log(f"warm start: {args.warm_episodes} guided episodes")
            t0 = time.time()
            tasks = [task("warm", eps=random.uniform(0.0, args.warm_eps))
                     for _ in range(args.warm_episodes)]
            episodes = []
            for k, r in enumerate(runner.imap(run_episode, tasks)):
                episodes.append(r["samples"])
                log(f"  warm {k + 1}/{args.warm_episodes} {r['instance']} eps {r['eps']:.3f}: "
                    f"score {r['score']:.4f} legal {r['legal']} ({r['seconds']:.1f}s)",
                    kind="warm", instance=r["instance"], eps=r["eps"], score=r["score"],
                    legal=r["legal"], seconds=r["seconds"])
            # hold out whole episodes (states within an episode are correlated)
            n_val = max(1, round(args.val_frac * len(episodes))) if len(episodes) >= 2 else 0
            val = [s for ep in episodes[:n_val] for s in ep]
            for ep in episodes[n_val:]:
                buffer.extend(ep)
            best_val, best_state, bad, done = math.inf, None, 0, 0
            train_list = list(buffer)
            while done < args.warm_steps:
                n = min(100, args.warm_steps - done)
                stats = trainer.train(train_list, n)
                done += n
                if not val:
                    continue
                v = eval_losses(model, val, args.device)
                log(f"  warm step {done}: train {stats['loss']:.4f}  held-out {v['loss']:.4f}  "
                    f"(grad norm {stats['grad_norm']:.3f}, lr {stats['lr']:.2e})",
                    kind="warm_val", step=done, train=stats, val=v)
                if v["loss"] < best_val - 1e-4:
                    best_val, best_state, bad = v["loss"], copy.deepcopy(model.state_dict()), 0
                else:
                    bad += 1
                    if bad >= args.patience:
                        log(f"  warm start: held-out loss stopped improving, stopping at step {done}")
                        break
            if best_state is not None:
                model.load_state_dict(best_state)
            buffer.extend(val)  # held-out states join the replay buffer afterwards
            log(f"warm start done in {time.time() - t0:.0f}s, best held-out loss {best_val:.4f}",
                kind="warm_train", best_val=best_val, steps=done)
            checkpoint(os.path.join(args.out, "warm.pt"), -1)
            checkpoint(os.path.join(args.out, "latest.pt"), -1)
            record_spent()

        # ---- 2. self-play -----------------------------------------------------
        last_iter_s = 0.0
        for it in (range(start_iter, args.iters) if args.iters > 0 else itertools.count(start_iter)):
            if spent() + 1.2 * last_iter_s > budget:
                log(f"stopping: --max-hours {args.max_hours} reached "
                    f"({spent() / 3600:.2f} h used, next iteration ~{last_iter_s / 60:.0f} min)")
                break
            t0 = time.time()
            save_actor()
            tasks = [task("mcts", model_path=actor_path, model_tag=it, device=args.selfplay_device,
                          sims=args.sims, c_puct=args.c_puct, temp_moves=args.temp_moves,
                          compile=args.compile)
                     for _ in range(args.episodes)]
            pending = [runner.submit(run_episode, t) for t in tasks]
            scores, n_new, failed = [], 0, 0
            fresh_sum, fresh_n = {}, 0
            chunks, done_steps = [], 0

            def step_cap():
                return min(args.train_steps, max(1, int(args.max_passes * len(buffer) / args.batch)))

            while pending:
                ready = [h for h in pending if h.ready()]
                for h in ready:
                    pending.remove(h)
                    try:
                        r = h.get()
                    except Exception as exc:  # one broken episode shouldn't end the run
                        failed += 1
                        log(f"  iter {it}: an episode crashed and was skipped ({type(exc).__name__}: {exc})")
                        continue
                    if r["samples"]:
                        # loss on states the network has never trained on: the overfitting check
                        f = eval_losses(model, r["samples"], args.device)
                        for k, v in f.items():
                            fresh_sum[k] = fresh_sum.get(k, 0.0) + v * len(r["samples"])
                        fresh_n += len(r["samples"])
                    buffer.extend(r["samples"])
                    n_new += len(r["samples"])
                    scores.append(r["score"])
                    log(f"  iter {it} ep {len(scores) - 1}: {r['instance']} ({r['n']} nets) "
                        f"score {r['score']:.4f} legal {r['legal']} moves {r['moves']} ({r['seconds']:.0f}s)",
                        kind="selfplay", iter=it, instance=r["instance"], score=r["score"],
                        legal=r["legal"], moves=r["moves"], seconds=r["seconds"])
                if pending and not ready:
                    # workers are busy: train on the GPU meanwhile instead of idling
                    n = min(10, step_cap() - done_steps)
                    if buffer and n > 0:
                        chunks.append((n, trainer.train(list(buffer), n)))
                        done_steps += n
                    else:
                        time.sleep(0.2)
            if failed > len(tasks) // 2:
                raise RuntimeError(f"{failed}/{len(tasks)} self-play episodes crashed in iteration {it}")
            if not scores:
                continue
            fresh = {k: round(v / fresh_n, 5) for k, v in fresh_sum.items()} if fresh_n else None
            overlapped = done_steps  # gradient steps taken while the workers were playing
            steps = max(done_steps, step_cap())
            if steps > done_steps:
                chunks.append((steps - done_steps, trainer.train(list(buffer), steps - done_steps)))
            stats = merge_stats(chunks)
            if stats["skipped"] > steps // 2:
                log(f"  WARNING: {stats['skipped']}/{steps} steps had non-finite loss/gradients; "
                    f"restoring the last good checkpoint")
                good, ck = load_checkpoint(os.path.join(args.out, "latest.pt"), args.device)
                model.load_state_dict(good.state_dict())
                try:
                    opt.load_state_dict(ck["optimizer"])
                except (ValueError, TypeError):
                    pass
            mean = sum(scores) / len(scores)
            last_iter_s = time.time() - t0
            per_hour = round(n_new * 3600 / last_iter_s)
            g = gpu.take() or {}
            gap = f", unseen-data loss {fresh['loss']:.4f}" if fresh else ""
            gtxt = f", GPU {g['gpu_util']:.0f}% / {g['gpu_mem_mb'] / 1024:.1f} GB" if g else ""
            log(f"iter {it}: mean self-play score {mean:.4f}, train loss {stats['loss']:.4f}{gap}, "
                f"grad norm {stats['grad_norm']:.3f}, lr {stats['lr']:.2e}, buffer {len(buffer)}, "
                f"{steps} steps ({overlapped} during self-play), {per_hour} samples/h{gtxt}, {last_iter_s:.0f}s "
                f"(total {spent() / 3600:.2f} h)",
                kind="iter", iter=it, mean_score=mean, buffer=len(buffer), steps=steps,
                seconds=last_iter_s, fresh=fresh, samples=n_new, samples_per_hour=per_hour,
                steps_during_selfplay=overlapped,
                **g, **stats)
            checkpoint(os.path.join(args.out, "latest.pt"), it)
            record_spent()

            if args.eval_every and (it + 1) % args.eval_every == 0:
                t1 = time.time()
                checkpoint(os.path.join(args.out, f"iter_{it:04d}.pt"), it)
                save_actor()
                suite_dir = tier_dir(args.eval_tier)
                cases = load_manifest(suite_dir)["cases"][:args.eval_cases]
                etasks = [{"suite_dir": suite_dir, "case": c,
                           "cache_dir": os.path.join(args.out, "negotiated_cache"),
                           "model_path": actor_path, "model_tag": f"eval{it}",
                           "device": args.selfplay_device, "evaluator": "net",
                           "sims": args.eval_sims, "c_puct": args.c_puct,
                           "frontier": args.frontier, "rip_limit": args.rip_limit,
                           "cleanup_seconds": args.eval_cleanup, "out": None,
                           "compile": args.compile} for c in cases]
                res = sorted(runner.imap(solve_case, etasks), key=lambda r: r["case"])
                for r in res:
                    log(f"    eval {r['case']}: search {r['search']:.4f}  final {r['final']:.4f}  "
                        f"({r['path']}, {r['seconds']:.0f}s)")
                mean_search = sum(r["search"] for r in res) / len(res)
                log(f"  eval {args.eval_tier}: mean search {mean_search:.4f} "
                    f"final {sum(r['final'] for r in res) / len(res):.4f}",
                    kind="eval", iter=it, cases=res)
                if mean_search > best_search:
                    best_search, best_iter, evals_since_best = mean_search, it, 0
                    checkpoint(os.path.join(args.out, "best.pt"), it)
                    log(f"  new best search score {mean_search:.4f}: saved best.pt")
                else:
                    evals_since_best += 1
                with open(best_path, "w") as fh:
                    json.dump({"iter": best_iter, "search": best_search,
                               "evals_since_best": evals_since_best}, fh)
                last_iter_s += time.time() - t1
                record_spent()
                if args.plateau_evals and evals_since_best >= args.plateau_evals:
                    log(f"stopping: no eval improvement in {evals_since_best} evaluations "
                        f"(best search score {best_search:.4f})")
                    break
    finally:
        runner.close()
    log(f"training finished: {spent() / 3600:.2f} h in total, "
        f"checkpoint {os.path.join(args.out, 'latest.pt')}")


if __name__ == "__main__":
    main()
