# gat_mcts: GAT policy/value network + MCTS router

A learned router for the [M3D routing challenge](https://github.com/partcleda/eda-3d-routing-challenge).
The routing state becomes a graph, a GAT scores candidate routes (policy) and
predicts the final score (value), MCTS uses both to pick the next move, and the
chosen route is laid down exactly. A cleanup pass runs on the finished board.

```
state ──► graph (nets, candidate routes, regions) ──► GATv2 (6 layers, ~3.0M parameters)
                                                       ├─► policy: logit per candidate
                                                       └─► value: final score, P(legal)
MCTS (PUCT) picks (net, candidate) ──► lay route ──► repair if a net got cut off ──► next state
...all nets routed ──► cleanup (rip-up & reroute, only keeps improvements) ──► submission
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
- runs a 3–5 minute self-test of every step
- generates the training data once
- trains for `TRAIN_HOURS`
- scores all 9 hard-tier cases into a leaderboard-format submission
- **disconnects the runtime**

Any error also disconnects the runtime (with `AUTO_SHUTDOWN = True`), so a
failure doesn't keep using compute units. Everything is saved in Drive under
`MyDrive/gat_mcts_work/`. If Colab drops, Run all again: the data is reused and
training resumes. `TRAIN_HOURS` is the total across sessions, so a resume never
goes over budget; raise it to train longer.

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

python -m gat_mcts.selftest --workers 2                     # ~3-5 min sanity check
python -m gat_mcts.gen_data --out data --n 150 --workers 4  # training pool (resumable)
python -m gat_mcts.train --data data --out runs/r1 --max-hours 3 --workers 4
python -m gat_mcts.train --data data --out runs/r1 --max-hours 6 --workers 4 --resume runs/r1/latest.pt
python -m gat_mcts.evaluate --tier hard --ckpt runs/r1/latest.pt --workers 4 --out submissions/hard/r1

# no network: the same search with leaves scored by finishing the board
python -m gat_mcts.evaluate --tier hard --evaluator rollout --sims 16 --out submissions/hard/rollout
```

## The pieces

| file | what it does |
|---|---|
| `problem.py` | per-instance tables, the routing `State`, candidate generation (Dijkstra), exact delay, rip-up repair |
| `graph.py` | state → `HeteroData` (node types `net`, `cand`, `region`; 10 edge types) |
| `model.py` | `RouteGAT`: `HeteroConv` of `GATv2Conv` per edge type, residual + LayerNorm, policy and value heads |
| `mcts.py` | AlphaZero-style PUCT search (min-max normalised Q, Dirichlet noise, subtree reuse) |
| `selfplay.py` | warm-start episodes, MCTS self-play episodes, `solve()` for evaluation |
| `cleanup.py` | rip-up-and-reroute pass on a finished board |
| `workers.py` | parallel episodes and case solving (spawned processes, CUDA-safe) |
| `gen_data.py` | training pool from the challenge's own generator (different seed from the benchmarks) |
| `train.py` | warm start + self-play training loop, time budget, checkpoints, periodic evaluation |
| `evaluate.py` | route a tier with a checkpoint, write a leaderboard-format submission, score it |
| `selftest.py` | runs every entry point at tiny scale |
| `colab.ipynb` | the whole flow on Colab |

**Scoring.** Training targets use the toolkit's own scorer
(`m3d.scorer.score_case`): the per-case score is `baseline_total / total_delay`,
and an illegal (failed) routing scores 0. The value head predicts that number.

**Graph.** Each net node knows its size, bounding box, lower-bound delay and
whether it's routed. Each candidate knows its detour over the net's lower
bound, wirelength, vias, which layers it uses, and how much it overlaps other
candidates and hugs other nets' pins. Each region (a tile of one layer) knows
its layer cost, occupancy, pin density and future demand. Conflict edges link
candidates of different nets that share a vertex. Information spreads one hop
per GAT layer. A net is about 4 hops from a net it competes with
(net → candidate → region → candidate → net), so 6 layers cover that with a
hop to spare. Defaults: width 152, 4 heads, 6 layers ≈ 3.0M parameters
(`--hidden`, `--heads`, `--layers`; width must be divisible by heads).
Checkpoints store their own size, so older checkpoints still load.

## Design decisions (measured on `benchmarks_hard`)

1. **Candidates include the negotiated router's route (`ref`).** Routing nets
   one at a time with only shortest-path-style candidates never finished a
   hard case; every variant cascaded into hundreds of rip-ups. So each net is
   also offered its route from the toolkit's negotiated router (run by this
   pipeline and counted in runtime), plus a shortest-path tree that avoids what
   the negotiator found contended (`spt_hist`). Choosing `ref` everywhere
   reproduces the baseline exactly (score 1.0), and the search learns which
   deviations beat it.
2. **Rip-up repair inside the environment.** When a move cuts a net off from
   its pins, that net is routed through other nets' wires at a penalty, and
   those nets go back in the queue. Episodes only fail (score 0) if one net
   needs repair more than `--rip-limit` times.
3. **Warm start.** An untrained network failed every episode (score 0), which
   gives it nothing to learn from. So training starts with episodes that follow
   `ref` with random deviations, labelled with their real outcome.
4. **Small frontier.** MCTS chooses among the next `--frontier` nets (4,
   largest first), with up to 7 candidates each.
5. **Evaluation keeps the better result.** `solve()` compares MCTS + cleanup
   with the negotiated solution + cleanup and submits the lower delay. The log
   and `evaluate.py` also print the search's own score (`search`), which is the
   number that shows whether training is working.

## Reading the training log (`log.jsonl` / cell output)

* `warm` lines: `score` mostly 0.9–1.0 and `legal` true.
* `iter` lines: `mean_score` is the average self-play challenge score on
  training instances (above 1.0 = the search beats the baseline before
  cleanup); `value` is the value-head error. Each line also shows the seconds
  per iteration.
* `eval` lines (every 10 iterations, 2 hard cases): `search` is the score
  straight out of MCTS. It should rise over training; `final` is what gets submitted.

## Status

Tested locally (CPU) from a copy outside the challenge repo: the self-test, the
notebook flow (with Colab's Drive/secret/shutdown calls simulated), legal
hard-tier submissions accepted by `m3d.scorer`, resume and the hour budget.
Not tested on an actual Colab GPU. The notebook's self-test step is there to
catch environment differences before training starts. It hasn't been trained
for real yet, so there are no meaningful scores. Reference points on
`hard/case_01`: following `ref` everywhere = 1.000 (1.0275 after cleanup); no
legal routing can exceed 1.668 (per-net lower bound).
