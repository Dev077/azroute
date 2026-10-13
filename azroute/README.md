# azroute

A search-and-learning router for the [M3D routing challenge](https://github.com/partcleda/eda-3d-routing-challenge):
the AlphaZero learning loop adapted to rip-up-and-reroute.

## How it works

1. **Initial routing.** Negotiated congestion (PathFinder-style) in C++ with delay-aware trees,
   from a few net orders and pressure schedules; the best legal result is kept.
2. **Improvement loop.** Repeatedly pick a neighbourhood (a badly detoured net plus the nets
   blocking its ideal tree), rip it up, and reroute it inside a bounded region one routing
   decision at a time. Each decision is a (net, mode) pair; the modes are `KEEP` (its previous
   route), `DELAY`, `SPARE_MID`, `LEARNED_PRICE`, `OUTER_TRUNK`, `FAR_SINK_FIRST` and `COMPACT`.
   A reroute is kept only if the neighbourhood's delay does not increase, so the routing is
   legal at every point.
3. **Search.** Gumbel root selection with sequential halving over the routing decisions.
   A partial reroute is scored by the share of the neighbourhood's room for improvement it is
   expected to recover.
4. **Network (~10 M parameters).** A multi-scale residual CNN over the region (occupancy, pins,
   previous routes, layer delays) so every cell sees the whole region, and net interaction
   layers: each net reads the grid with attention biased to its own box, and nets exchange
   messages conditioned on pair vectors that start from exact conflict counts (how much of
   net i's ideal tree net j occupies now, would occupy if kept, or blocks at its pins).
   Outputs: decision distribution, remaining excess over the ideal delay, chance of being
   blocked, future cell occupancy (which drives `LEARNED_PRICE`), and per-net excess.
   See `docs/ARCHITECTURE.md`.
5. **Training.** The network learns the search's improved decision distribution and the
   outcomes the reroutes actually reached, from records produced while improving the 45
   challenge cases. Better network, stronger search, better records.

Every accepted reroute is logged, so each final routing can be replayed exactly from the
initial routing (`lns.replay`).

## Layout

| file | role |
|---|---|
| `core/router_core.cpp` | C++ core (pybind11): occupancy, A* tree building, rip-up/undo, negotiated congestion |
| `problem.py`, `modes.py`, `tiers.py`, `paths.py` | a case loaded into the core; routing modes; the tiers |
| `lns.py` | neighbourhoods, the improvement loop, replay |
| `search.py` | search over routing decisions; training records |
| `features.py`, `model.py` | network inputs (incl. pair conflict features); the network |
| `docs/` | architecture diagram, loss and every setting (`ARCHITECTURE.md`); what to expect from a run (`RUN_EXPECTATIONS.md`) |
| `learner.py` | replay buffer, symmetry augmentation, loss |
| `worker.py`, `inference.py`, `run.py` | worker processes; the single process that runs the network for all workers (batched); the orchestrator (dispatch, training, Drive sync, deadline) |
| `export.py` | checked submission files, `runtime.json`, summary |
| `colab_train.ipynb` | the Colab notebook |

## Run on Colab

Open `colab_train.ipynb` from this repository in Colab, choose a **G4** GPU runtime, set
`REPO_URL` to this repository, and run the cells. The notebook clones the challenge repo for
the `m3d` toolkit and the benchmarks, builds the core, runs the tests, starts the run in the
background, and has cells to monitor, stop early, and export. State is checkpointed to Drive
every 30 minutes; after a disconnect, rerun cells 1-5 and it resumes with its original deadline.

Outputs (in `DRIVE_DIR/export/`): `submissions/<tier>/<case>.sol.json`, `submissions/<tier>/runtime.json`,
and `summary.md`. Before submitting to the leaderboard, add `meta.json` per tier as described in
the challenge's CONTRIBUTING.md (hardware, threads, a link to this repo).

## Run locally

```bash
git clone https://github.com/partcleda/eda-3d-routing-challenge
export AZROUTE_CHALLENGE_ROOT=$PWD/eda-3d-routing-challenge
pip install -r azroute/requirements.txt
python azroute/setup.py                       # builds azroute/_core*.so
python -m unittest azroute.tests.test_core
python -m azroute.run --smoke --cpu --hours 0.1 --work /tmp/az --drive ''   # about 6 minutes
```

`python -m azroute.run --help` lists every setting (workers, batch, search budgets, network size, ...).

## Notes

* **Runtime accounting.** `runtime.json` charges each case the worker time spent on it plus a share
  of the learner's GPU time proportional to the training records it produced.
* **Reproducibility.** Replay logs reproduce a routing exactly on the same build of the core;
  the initial routing is deterministic for a given build.
