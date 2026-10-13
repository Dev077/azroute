"""Train the net-ordering GAT on the challenge score.

    python -m gat_order.train --data pool --out runs/order1 --workers 2

Each step takes --boards pool boards. For each, the GAT scores every net, and
--samples noisy copies of those scores (noise --sigma) give that many orders.
The negotiated router routes the board in each order, and each routing gets the
score-based loss L = total_delay / baseline_total = 1 / score (--fail-loss if it
never becomes legal). The gradient of the expected loss with respect to the
scores comes from how the losses moved with the noise (solver.py), and is
passed back through the GAT: one optimizer step per step.

Every --eval-every steps, the first --eval-cases cases of the hard tier are
routed with the GAT's order (greedy) and --eval-samples - 1 noisy ones; the
best legal one is the eval score (the default order scores exactly 1.0 there).
Stops at --steps, or when the eval score hasn't improved for --plateau-evals
evaluations. Checkpoints (latest.pt, best.pt) and a JSON-lines log go to --out.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import time

import torch
from torch_geometric.data import Batch

from m3d.model import Instance
from m3d.suite import load_manifest

from .data import load_pool, tier_dir
from .graph import build_graph
from .model import OrderGAT, load_checkpoint, save_checkpoint
from .solver import (Router, best_routing, candidate_orders, grad_estimate, loss_of, noise,
                     order_from_scores)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", required=True, help="pool directory (index.json + <name>.json)")
    ap.add_argument("--out", required=True, help="checkpoints + log (e.g. on Google Drive)")
    ap.add_argument("--resume", default=None, help="checkpoint to resume from")
    ap.add_argument("--tiers", default="hard", help="pool tiers to train on (comma-separated)")
    ap.add_argument("--max-nets", type=int, default=110, help="skip pool boards larger than this")
    # model
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--layers", type=int, default=4)
    # perturbed-optimizer training
    ap.add_argument("--steps", type=int, default=0, help="optimizer steps (0 = no limit)")
    ap.add_argument("--boards", type=int, default=4, help="boards per step")
    ap.add_argument("--samples", type=int, default=8, help="noisy orders routed per board")
    ap.add_argument("--sigma", type=float, default=1.0, help="noise on the scores")
    ap.add_argument("--fail-loss", type=float, default=1.5,
                    help="loss of a routing that never became legal (a legal one is ~1)")
    ap.add_argument("--max-iters", type=int, default=50, help="negotiation passes per routing")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--clip", type=float, default=1.0)
    # evaluation
    ap.add_argument("--eval-every", type=int, default=10, help="steps (0 = never)")
    ap.add_argument("--eval-cases", type=int, default=2)
    ap.add_argument("--eval-samples", type=int, default=8, help="greedy order + noisy ones")
    ap.add_argument("--plateau-evals", type=int, default=0,
                    help="stop when the eval score hasn't improved for this many evaluations (0 = never)")
    # hardware
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--workers", type=int, default=1, help="parallel routing processes")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    log_path = os.path.join(args.out, "log.jsonl")
    best_path = os.path.join(args.out, "best.json")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)

    def log(msg, **rec):
        print(msg, flush=True)
        if rec:
            with open(log_path, "a") as fh:
                fh.write(json.dumps({"time": time.time(), **rec}) + "\n")

    tiers = set(args.tiers.split(","))
    pool = [e for e in load_pool(args.data)
            if e["n_nets"] <= args.max_nets and e.get("tier", "hard") in tiers]
    if not pool:
        raise SystemExit(f"no {args.tiers} boards with <= {args.max_nets} nets in {args.data}")
    log(f"pool: {len(pool)} boards from {args.data} ({args.tiers}); device {args.device}, "
        f"{args.workers} routing worker(s)")

    start, best_eval, evals_since_best = 0, -1.0, 0
    if args.resume:
        model, ck = load_checkpoint(args.resume, args.device)
        start = ck["extra"].get("step", -1) + 1
        if os.path.exists(best_path):
            with open(best_path) as fh:
                b = json.load(fh)
            best_eval, evals_since_best = b["eval"], b.get("evals_since_best", 0)
    else:
        model, ck = OrderGAT(args.hidden, args.heads, args.layers).to(args.device), None
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    if ck and ck.get("optimizer"):
        try:
            opt.load_state_dict(ck["optimizer"])
        except ValueError:
            log("note: optimizer state in the checkpoint doesn't match; starting it fresh")
    log(f"model: {sum(p.numel() for p in model.parameters()):,} parameters, config {model.config}"
        + (f"; resumed at step {start}" if args.resume else ""))

    graphs = {}  # board name -> (instance path, instance, graph)

    def board(name, path):
        if name not in graphs:
            inst = Instance.load(path)
            graphs[name] = (path, inst, build_graph(inst))
        return graphs[name]

    def checkpoint(path, step):
        save_checkpoint(path, model, opt, {"step": step})

    suite = tier_dir("hard")
    eval_cases = load_manifest(suite)["cases"][:args.eval_cases]
    router = Router(args.workers)
    try:
        for step in (range(start, args.steps) if args.steps else itertools.count(start)):
            t0 = time.time()
            entries = random.sample(pool, min(args.boards, len(pool)))
            boards = [board(e["name"], os.path.join(args.data, e["name"] + ".json")) for e in entries]
            model.train()
            batch = Batch.from_data_list([g for _p, _i, g in boards]).to(args.device)
            w_all = model(batch)
            ws = [w_all[batch["net"].ptr[b]:batch["net"].ptr[b + 1]] for b in range(len(boards))]
            tasks, zs = [], []
            for (path, inst, _g), w in zip(boards, ws):
                z = noise(len(inst.nets), args.samples, gen)
                zs.append(z)
                ids = [net.id for net in inst.nets]
                wd = w.detach().cpu()
                tasks += [{"path": path, "order": order_from_scores(wd + args.sigma * zi, ids),
                           "max_iters": args.max_iters} for zi in z]
            res = router.map(tasks)
            surrogate, scores_all, fails, per_board = 0.0, [], 0, []
            for b, (e, w, z) in enumerate(zip(entries, ws, zs)):
                r = res[b * args.samples:(b + 1) * args.samples]
                losses = torch.tensor([loss_of(x["delay"], e["baseline_total"], args.fail_loss) for x in r])
                g = grad_estimate(losses, z, args.sigma).to(args.device)
                surrogate = surrogate + (w * g).sum() / len(boards)
                sc = [e["baseline_total"] / x["delay"] for x in r if x["delay"]]
                fails += len(r) - len(sc)
                scores_all += sc
                per_board.append({"board": e["name"], "n": e["n_nets"], "mean": sum(sc) / len(sc) if sc else 0.0,
                                  "best": max(sc, default=0.0), "failed": len(r) - len(sc)})
            opt.zero_grad(set_to_none=True)
            surrogate.backward()
            gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip))
            skipped = not torch.isfinite(torch.tensor(gn))
            if not skipped:
                opt.step()
            opt.zero_grad(set_to_none=True)
            spread = float(torch.cat([w.detach() for w in ws]).std())
            mean = sum(scores_all) / len(scores_all) if scores_all else 0.0
            for pb in per_board:
                log(f"  step {step} {pb['board']} ({pb['n']} nets): mean {pb['mean']:.4f} "
                    f"best {pb['best']:.4f}, {pb['failed']}/{args.samples} failed", kind="board", step=step, **pb)
            log(f"step {step}: mean score {mean:.4f} (baseline order = 1.0), best "
                f"{max(scores_all, default=0.0):.4f}, {fails}/{len(res)} routings failed, "
                f"grad norm {gn:.4f}{' (skipped)' if skipped else ''}, score spread {spread:.3f} "
                f"(vs noise {args.sigma}), {time.time() - t0:.0f}s",
                kind="step", step=step, mean_score=mean, best_score=max(scores_all, default=0.0),
                failed=fails, routings=len(res), grad_norm=gn, skipped=bool(skipped), spread=spread,
                seconds=time.time() - t0)
            checkpoint(os.path.join(args.out, "latest.pt"), step)

            if args.eval_every and (step + 1) % args.eval_every == 0:
                model.eval()
                out = []
                for c in eval_cases:
                    path = os.path.join(suite, c["instance_file"])
                    _p, inst, g = board("eval:" + c["name"], path)
                    with torch.no_grad():
                        w = model(g.to(args.device)).cpu()
                    orders = candidate_orders(w, [net.id for net in inst.nets], args.eval_samples,
                                              args.sigma, seed=1)
                    r = best_routing(router, path, orders, c["baseline_total"], args.max_iters)
                    out.append({"case": c["name"], "greedy": r["greedy"], "best": r["best"],
                                "legal": r["legal"], "of": len(orders)})
                    log(f"    eval {c['name']}: greedy order {r['greedy']:.4f}  best of {len(orders)} "
                        f"{r['best']:.4f}  ({r['legal']}/{len(orders)} legal)")
                score = sum(o["best"] for o in out) / len(out)
                log(f"  eval hard: mean best {score:.4f}, mean greedy "
                    f"{sum(o['greedy'] for o in out) / len(out):.4f} (default order = 1.0)",
                    kind="eval", step=step, cases=out, score=score)
                checkpoint(os.path.join(args.out, f"step_{step:05d}.pt"), step)
                if score > best_eval:
                    best_eval, evals_since_best = score, 0
                    checkpoint(os.path.join(args.out, "best.pt"), step)
                    log(f"  new best eval {score:.4f}: saved best.pt")
                else:
                    evals_since_best += 1
                with open(best_path, "w") as fh:
                    json.dump({"step": step, "eval": best_eval, "evals_since_best": evals_since_best}, fh)
                if args.plateau_evals and evals_since_best >= args.plateau_evals:
                    log(f"stopping: no eval improvement in {evals_since_best} evaluations (best {best_eval:.4f})")
                    break
    finally:
        router.close()
    log(f"training finished; checkpoint {os.path.join(args.out, 'latest.pt')}")


if __name__ == "__main__":
    main()
