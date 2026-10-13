# gat_order: a GAT picks the net order, Dijkstra connects

The simplest learned router in this repo. Two parts:

* **GAT** (`model.py`, ~0.56M parameters): reads the netlist as a graph (nets,
  die tiles, which nets' bounding boxes overlap) and gives every net a priority
  score. Nets are routed highest score first.
* **Dijkstra** (`router.py`): the toolkit's negotiated router
  (`m3d.negotiated`, PathFinder-style Dijkstra with congestion costs) connects
  the nets in that order. Only the order is changed; the toolkit is untouched.

```
netlist ──► graph ──► GAT ──► score per net ──► sort ──► negotiated router (Dijkstra) ──► routing ──► score
```

Why the order: the router's default order (largest bounding box first) is how
the hard tier's baseline was made, so it scores exactly 1.0. Other orders do
better or worse (measured on hard case_01: 0.957 to 1.074 over 8 orders; on
case_02 3 of 6 random orders never became legal). A plain order-then-Dijkstra
router without negotiation failed every hard case whatever the order, which
is why the negotiated router does the connecting.

## The loss, modelled on the score

The challenge scores a case as `baseline_total / total_delay` (higher is
better; illegal = 0). The loss is its inverse, something to minimise:

    L = total_delay / baseline_total = 1 / score        (--fail-loss, 1.5, if never legal)

"Sort, then route" has no useful derivative, so the gradient is taken of its
smoothed version: the expected loss when the GAT's scores `w` get Gaussian
noise (perturbed optimizers, Berthet et al. 2020):

    d/dw E[ L(route(sort(w + sigma Z))) ]  =  E[ L(...) Z ] / sigma

Each step routes `--samples` noisy orders per board (antithetic pairs Z, −Z;
the mean loss subtracted as a baseline) and passes that gradient back through
the GAT. Early on the GAT's scores are tiny next to the noise (`spread` in the
log), so the orders are nearly random: that is the exploration. As the GAT
learns, its scores spread out and the orders follow them.

## Run

```bash
export M3D_CHALLENGE_ROOT=/path/to/challenge
pip install -r gat_order/requirements.txt

python -m gat_order.selftest --workers 4                     # a few minutes' sanity check
python -m gat_order.gen_data --out pool --n 300 --workers 4    # or reuse a gat_mcts pool
python -m gat_order.train --data pool --out runs/order1 --workers 4 --plateau-evals 5
python -m gat_order.evaluate --tier hard --ckpt runs/order1/best.pt --workers 4 --out submissions_local/hard/order1
```

On Colab: `gat_order/colab.ipynb` (same flow as the gat_mcts notebook; it uses
the hard boards of your existing `gat_mcts_work/data_tiers` pool if it's in
Drive). A GPU isn't needed; CPU cores are what make it faster.

## The pieces

| file | what it does |
|---|---|
| `router.py` | the negotiated router with a given order; `default_order` (the baseline's) |
| `graph.py` | netlist → `HeteroData` (nets, die tiles; covers / overlaps / adjacency edges) |
| `model.py` | `OrderGAT`: GATv2 layers (residual + LayerNorm) → one score per net |
| `solver.py` | scores → orders, noise, the loss, the gradient estimate, parallel routing |
| `train.py` | training loop, checkpoints (`latest.pt`, `best.pt`), periodic evaluation, plateau stop |
| `evaluate.py` | route a tier with a checkpoint (GAT order + noisy ones, best legal kept), score it |
| `gen_data.py` | pool of hard-recipe boards (negotiated baseline), different seed from the benchmarks |
| `data.py` | pool and tier loading (pools from `gat_mcts.gen_data` work too) |
| `selftest.py` | every entry point at tiny scale |
| `colab.ipynb` | the whole flow on Colab |

## Reading the log

* `step` lines: the mean and best score of that step's routings (default order
  = 1.0), how many never became legal, the gradient norm, and `spread`: how
  far apart the GAT's scores are, next to the noise `sigma` (spread much
  smaller than sigma = orders still mostly random).
* `eval` lines (every 10 steps, hard case_01 and case_02): `greedy` is the
  GAT's own order, `best` the best of it and `--eval-samples` − 1 noisy ones.
  Above 1.0 beats the baseline. `best.pt` follows the mean `best`.

## Cost

Each routing takes ~8–16 s on hard case_01/case_02 sizes and ~45 s on the
largest hard case, so a step (`--boards 4` × `--samples 8` = 32 routings)
takes a few minutes on a few cores. Defaults train on hard-tier boards only
(`--tiers hard`, `--max-nets 110`): the negotiated router takes minutes per
board on congested and scale sizes.

## Status

Tested locally (CPU): the self-test (the default order reproduces the hard
baseline exactly; training steps; resume; a legal 2-case submission scored by
the toolkit). Not trained for real yet.
