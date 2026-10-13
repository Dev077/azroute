# gat_mcts: GAT policy/value network + MCTS over kicks, improving a routing

A learned router for the [M3D routing challenge](https://github.com/partcleda/eda-3d-routing-challenge).
It starts from a legal routing (the toolkit's negotiated router, then a cleanup
pass) and improves it. Each move is a *kick*: re-route one net through other
nets' wires at a crossing penalty, re-route the nets it displaced, then let the
nets around the freed space take shorter routes. MCTS looks ahead over kicks
(every board it tries is legal and scored exactly); a GAT tells it which kicks
to try first (policy) and how much better the board can still get (value), and
is trained AlphaZero-style on the search's results. The best board an episode
reaches is kept.

```
negotiated router ──► cleanup ──► start board (a local optimum of single-net rerouting)

board ──► graph (nets, kicks, regions) ──► GATv2 (6 layers, ~3.3M parameters)
                                            ├─► policy: logit per kick (net x strength)
                                            └─► value: % gain still reachable
MCTS (PUCT) picks a kick ──► re-route net, displaced nets, settle ──► next board (always legal)
...after --moves kicks: best board seen ──► cleanup ──► submission
```

## Train on Colab

1. Push this repo (with `gat_mcts/` in it) to GitHub.
2. Open `gat_mcts/colab.ipynb` in Colab (File → Open notebook → GitHub, or upload it).
3. Runtime → Change runtime type → **T4 GPU**.
4. In the **Settings** cell set `GAT_REPO_URL` (and `GAT_REPO_SUBDIR` if
   `gat_mcts/` isn't at the repo root). For a private repo, add a GitHub token
   as a Colab secret named `GITHUB_TOKEN` (key icon in the left bar).
5. Runtime → **Run all**, and approve the Google Drive pop-up.

The notebook then:
- clones the challenge toolkit (pinned to the tested commit) and your repo
- runs a few-minute self-test of every step
- generates the training data once
- trains until the evaluation score plateaus (`PLATEAU_EVALS`; no hour limit)
- scores all 9 hard-tier cases into a leaderboard-format submission
- **disconnects the runtime**

Any error also disconnects the runtime (with `AUTO_SHUTDOWN = True`), so a
failure doesn't keep using compute units. Everything is saved in Drive under
`MyDrive/gat_mcts_work/`. If Colab drops, Run all again: the data is reused and
training resumes. There is no hour limit, so a long run can use many units;
interrupt the training cell to stop sooner (`latest.pt` is kept).

## Requirements

* Python 3.9+, `torch`, `torch_geometric` (see `requirements.txt`).
* The challenge toolkit: the `m3d/` package and `benchmarks_*/` folders from the
  challenge repo. `gat_mcts` finds it via the `M3D_CHALLENGE_ROOT` environment
  variable, else in the current directory or its parents.

## Running outside Colab

```bash
git clone https://github.com/partcleda/eda-3d-routing-challenge /path/to/challenge
export M3D_CHALLENGE_ROOT=/path/to/challenge
pip install -r gat_mcts/requirements.txt

python -m gat_mcts.selftest --workers 2                     # a few minutes' sanity check
python -m gat_mcts.gen_data --out data --n 150 --workers 4  # training pool (resumable; --mix picks the tiers)
python -m gat_mcts.train --data data --out runs/r1 --max-hours 3 --workers 4
python -m gat_mcts.train --data data --out runs/r1 --max-hours 6 --workers 4 --resume runs/r1/latest.pt
python -m gat_mcts.evaluate --tier hard --ckpt runs/r1/latest.pt --workers 4 --out submissions/hard/r1

# no network: the same search with heuristic priors (the baseline the network is compared with)
python -m gat_mcts.evaluate --tier hard --evaluator heuristic --out submissions/hard/heuristic
```

## The pieces

| file | what it does |
|---|---|
| `problem.py` | per-instance tables, the routing `State`, shortest-path routing (Dijkstra), exact delay, kick moves, `settle`, `Board` |
| `graph.py` | board + kicks on offer → `HeteroData` (node types `net`, `cand`, `region`; 11 edge types) |
| `model.py` | `RouteGAT`: `HeteroConv` of `GATv2Conv` per edge type, residual + LayerNorm, policy and value heads |
| `mcts.py` | AlphaZero-style PUCT search over kicks, with the network or without it (best-along-path returns, min-max normalised Q, Dirichlet noise, subtree reuse, failed kicks masked) |
| `selfplay.py` | start board, MCTS episodes (self-play and greedy), `solve()` for evaluation |
| `cleanup.py` | rip-up-and-reroute pass on a finished board (start board and final pass) |
| `workers.py` | parallel self-play episodes and case solving (spawned processes, CUDA-safe) |
| `gen_data.py` | training pool from the hard / congested / scale tiers' generator recipes (`--mix`; different seed from the benchmarks) |
| `train.py` | AlphaZero self-play training loop (no warm start), time budget, checkpoints, periodic evaluation |
| `evaluate.py` | route a tier with a checkpoint, write a leaderboard-format submission, score it |
| `selftest.py` | runs every entry point at tiny scale |
| `colab.ipynb` | the whole flow on Colab |

**Scoring.** Every board in the search is a complete, legal routing, scored
exactly (`baseline_total / total_delay`); each episode's best board is checked
with the toolkit's own checker and scorer (`m3d.scorer.score_case`).

**Targets.** Each state the search played from is labelled with the search's
visit counts (policy target) and with how much better, in percent, the best
board from that point on was (value target). The value head's estimate is
what the search uses at a leaf.

**Moves.** A kick is (net, strength). The net is re-routed by Dijkstra with
other nets' wires allowed at a cost of 3, 6 or 12 per vertex (`KICKS`; pins are
never crossed); the nets it crosses are ripped up and re-routed (largest
first); then `settle` retries the nets whose search window overlaps the freed
space and could get shorter through it (worst detour first, at most
`--settle` reroutes, each searched in a box around the net's pins and current
route). A kick that leaves a crossed net with no route, or changes nothing, is
masked the first time the search tries it. Kicks can make the board worse;
the episode keeps the best board it reaches.

**Graph.** Each net node knows its size, bounding box, lower-bound delay, its
current detour, whether it has changed this episode and how many kicks would
push it aside. Each kick node knows its strength, the net's detour, and which
nets the net's ideal route crosses (also as `hits` edges to those nets). Each
region (a tile of one layer) knows its layer cost, occupancy and pin density.
The graph-level features include the moves left, the gain so far and how far
the current board is from the episode's best.
Defaults: width 152, 4 heads, 6 layers ≈ 3.3M parameters (`--hidden`,
`--heads`, `--layers`; width must be divisible by heads). Checkpoints store
their own size; checkpoints from earlier designs don't load (different inputs).

## Design decisions (measured on `benchmarks_hard`)

1. **Improve a routing instead of building one.** The first design routed nets
   one at a time, choosing among candidate routes that included the negotiated
   router's. Trained for 120 iterations, its search reproduced the negotiated
   routing exactly (score 1.0000) at every evaluation: every gain came from the
   cleanup pass. Building net by net, each route is chosen before the nets
   after it exist, so only the negotiated route is safe.
2. **Kicks get past cleanup.** Cleanup (single-net rerouting) stops at a local
   optimum: 1.0275 on case_01, 1.0450 on case_02. From there, about one kick in
   ten followed by the local settle beats it (on case_01 the best single kick
   gains 3.0%). Without any network, MCTS over kicks reached 1.1031 on case_01
   (+7.4% over cleanup) in 8 kicks.
3. **AlphaZero, no warm start.** The network starts untrained; the search
   explores from the first iteration (Dirichlet noise at the root, moves
   sampled from the visit counts for the first 30% of each episode) and, since
   every board it tries is scored exactly, already improves on cleanup while
   the network is random. The network learns the search's visit counts and
   the gains its episodes reached, and in turn makes the search pick better
   kicks. Training starts with a one-off evaluation of the same search without
   a network (the `eval_heuristic` log line): the bar the network has to beat.
   (Vanilla policy gradient was tried first: an untrained policy sampling kicks
   gained +0.07% on average, against +7.4% for the search alone.)
4. **All tiers, windows on large boards.** Boards with more than
   `--full-max-nets 110` nets (congested, scale) kick only `--window 32` nets
   around a random one, and their start's cleanup covers just those and the
   nets around them. A kick costs 0.01–0.13 s on hard-tier boards and about
   1.3 s on a 126×126 scale board.
5. **Evaluation never loses to the baseline.** `solve()` starts from negotiated
   + cleanup and only moves to a better board. The log and `evaluate.py` print
   `start` (negotiated + cleanup), `search` (the best board the search found,
   the number that shows whether training works) and `final` (after one more
   cleanup pass, what gets submitted).

## Training stability

| risk | what guards against it |
|---|---|
| exploding gradients | gradient-norm clipping (`--clip 1.0`); steps with a non-finite loss or gradient are skipped; if most of an iteration's steps are bad, the last good checkpoint is restored |
| vanishing gradients | residual + LayerNorm in every layer; a numerically stable log-softmax for the policy loss; learning-rate warmup (`--warmup-steps 200`) |
| overfitting | dropout 0.1 (attention and messages); AdamW weight decay 0.01 on weights (not biases/norms); at most `--max-passes 4` passes over the buffer per iteration; loss on each iteration's new, unseen episodes logged before training on them; `best.pt` kept apart from `latest.pt` |
| a rough end of training | cosine learning-rate decay to 10% over `--max-hours` (constant after warmup without it, as in the notebook) |

The Colab report plots train loss against the unseen-episode loss (a growing
gap means overfitting) and the gradient norm before clipping.

## GPU use and how long to train

Self-play (kicks: shortest-path searches; graph building) is CPU work in
Python; the GPU runs the network during search and the training steps. With
`--workers > 1` the GPU trains on the replay buffer *while* the workers play
the next episodes (the iteration log shows `N steps (M during self-play)`), so
neither waits for the other. On a 2-core runtime the CPU is the limit and the
GPU is often idle; more CPU cores (more workers) is what raises throughput.
Each iteration logs GPU utilisation (from `nvidia-smi`) and samples generated
per hour.

torch.compile is on by default (`--no-compile` on `train`/`evaluate` turns it
off). It compiles the network's tensor core (`RouteGAT._core`, with dynamic
shapes) in the training process and once per self-play / evaluation worker;
workers load new weights into the same compiled module, so they don't
recompile every iteration. Compiling takes a few minutes at the start of a
session.

How long: the notebook has no hour limit; `PLATEAU_EVALS = 5` stops training
once the evaluation score (every 10 iterations) hasn't improved for 5
evaluations. Raise it to be more patient, lower it to stop sooner. After a run,
look at the eval `search` scores in the report: if they plateaued, more training
won't help without changes (more simulations, more moves per episode, a
different model size).

## Reading the training log (`log.jsonl` / cell output)

* `eval_heuristic` line (once, at the start of a fresh run): the eval done by
  the search without a network. The network's eval `search` should end up
  above it.
* `selfplay` lines: each episode's start (after cleanup), best board and gain
  in %.
* `iter` lines: `mean_score` is the average best-board score of the self-play
  episodes and `mean_gain` their average gain over the cleaned start, in %;
  `value` is the value-head error (in squared % points), and the unseen-data
  loss is measured on each iteration's new episodes before training on them.
  Each line also shows the seconds per iteration.
* `eval` lines (every 10 iterations, 2 hard cases): `start` is negotiated +
  cleanup, `search` the best board the search reached. `search` should rise
  over training; `final` is what gets submitted.

## Status

Tested locally (CPU) from a copy outside the challenge repo: the self-test, the
notebook flow (with Colab's Drive/secret/shutdown calls simulated), legal
hard-tier submissions accepted by `m3d.scorer`, resume and the hour budget
(`--max-hours`, still available on the command line).
Not tested on an actual Colab GPU. The notebook's self-test step is there to
catch environment differences before training starts. It hasn't been trained
for real yet. Reference points on `hard/case_01`: negotiated = 1.000, after
cleanup 1.0275, MCTS over kicks without a network 1.1031 (16 simulations, 8
kicks); no legal routing can exceed 1.668 (per-net lower bound).
