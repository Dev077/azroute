# azroute: network, loss and settings

Current defaults, read off the code (`network/model.py`, `network/features.py`,
`network/learner.py`, `search/search.py`, `routing/lns.py`, `routing/modes.py`,
`routing/problem.py`, `pipeline/run.py`). Command-line flags of `python -m pipeline.run`
are given where a setting has one.

## Network: 10.33 M parameters, 11.9 GFLOPs per sample

| part | what it is | parameters |
|---|---|---:|
| 64 x 64 stage | stem conv 3x3 46 -> 96 + 2 residual blocks (96 ch) | 0.37 M |
| 32 x 32 stage | stride-2 conv + 3 residual blocks (160 ch) with region-wide context | 1.68 M |
| 16 x 16 stage | stride-2 conv + 3 residual blocks (224 ch) with region-wide context | 3.34 M |
| decoder | upsample + skip + 1 block at 32 x 32 (160 ch) and at 64 x 64 (96 ch) | 0.73 M |
| grid tokens | 32 x 32 decoder features, Linear 160 -> 256 + learned position (1024 tokens) | 0.30 M |
| net interaction | 3 layers x 1.12 M: grid read 0.26 M, pair update 0.05 M, net exchange 0.28 M, feed-forward 0.53 M | 3.36 M |
| initial states | nets: [net 24, reroute 6, box-pooled 96] -> 256 -> 256; pairs: [geometry 6, conflicts 3 + 3] -> 64 -> 64 | 0.10 M |
| future head | conv 3x3 96 -> 96, ReLU, conv 1x1 -> 6 | 0.08 M |
| trunk + heads | trunk 742 -> 256 -> 256; decision 256 -> 256 -> 7 per net; net excess 256 -> 128 -> 1; value; blocked | 0.36 M |

A `tiny` preset of the same network is used for smoke runs (`--smoke`).

### Grid backbone

A residual CNN that goes 64 x 64 -> 32 x 32 -> 16 x 16 and back up with skip connections.
The 16 x 16 stage gives every cell a view of the whole region (the old full-resolution-only
CNN saw about +-20 cells); the blocks at 32 x 32 and 16 x 16 also add region-wide context
(masked mean and max over the region -> per-channel bias). The full-resolution path keeps
single-cell detail for the future-occupancy head and the box pooling.

### Net interaction layers

One 256-wide state per net and one 64-wide state per ordered pair (i, j). Pair states start
from box geometry and exact conflict features computed by the router (below), in both
directions. Each of the 3 layers (8 heads, pre-LayerNorm residuals):

1. **Grid read.** Each net attends over the 1024 tokens of the 32 x 32 decoder features
   (2 x 2 cells per token), with a learned per-head penalty for tokens outside its box
   (expanded by one token) and cells outside the region masked.
2. **Pair update.** z_ij += f(h_i) + g(h_j) + W . overlap_ij, where overlap_ij is the attention
   mass nets i and j put on the same grid tokens (per head); then z += MLP(LayerNorm(z)).
3. **Net exchange.** Attention among nets with a per-head bias from z_ij; the message net j
   sends net i is v_j + W z_ij, so it depends on how the two nets conflict.
4. **Feed-forward** 256 -> 1024 -> 256.

### Inputs

| input | shape | contents |
|---|---|---|
| region canvas | 6 x 64 x 64, uint8 bits | bit 0 fixed-net occupancy, 1 rerouted-net occupancy, 2 fixed-net pins, 3 pending-net pins, 4 previous routes of pending nets, 5 inside region, (6 = training target: used later). Regions larger than 64 cells are max-pooled by an integer factor. |
| input planes (built on GPU) | 46 x 64 x 64 | 36 bit planes, 7 delay planes (6 layers + via, x region mask), 1 demand plane (pending boxes x density), 2 coordinates |
| net features | 16 x 24 | pending / rerouted / seed flags, log fanout, cross-die, driver and sinks on top die, box size and centre, ideal share, log(old / ideal), log(new / ideal), mode one-hot (7), box demand density, valid flag |
| net boxes | 16 x 4 | pin bounding box on the canvas |
| pair conflicts | 16 x 16 x 3 | fractions of net i's ideal tree (minimum-delay tree in the region, other nets ignored) that net j: occupies now; lies on j's previous route while j is pending (what KEEP on j brings back); sits in the via columns above j's pins |
| reroute features | 6 | log(old / ideal), slack, progress, canvas scale, #pending, #nets |
| legal decisions | 16 x 7 | pending nets x 7 modes |

### Outputs and how the search uses them

| output | shape | used for |
|---|---|---|
| decision logits | 112 (16 nets x 7 modes, illegal = -inf) | prior over routing decisions |
| remaining excess y | 1 (standardised) | estimated delay of the pending nets: LB_pending x exp(y) |
| blocked | 1 logit | probability the reroute ends blocked |
| future occupancy | 6 x 64 x 64 logits | price canvas for the `LEARNED_PRICE` mode |
| net excess | 16 | auxiliary training target only |

Score of a partial reroute in the search:

```
q = (1 - P_blocked) * clip((old - (rerouted + LB_pending * exp(y))) / slack, -1, 1) - P_blocked
slack = max(old - ideal, 0.02 * ideal)
```

## Loss

```
loss = KL(pi' || p)                       decision distribution; pi' = the search's improved distribution
     + 1.0  * (y_std - y_hat)^2            remaining excess, standardised; completed reroutes only
     + 0.5  * BCE(blocked)
     + 0.5  * BCE(future occupancy)        per region cell and layer; completed reroutes only
     + 0.25 * (net excess - target)^2      per net; completed reroutes only
     + AdamW weight decay 1e-4
```

| target | definition |
|---|---|
| pi' | softmax(logits + sigma(completed Q)), sigma(q) = (50 + max visits) x 0.1 x q, Q min-max rescaled per node |
| y | log(actual delay of the nets routed from this decision on / their ideal delay) |
| y_std | (y - mu) / sigma, with mu and sigma running averages (momentum 0.999, sigma floor 0.05) |
| blocked | 1 if the reroute ended blocked |
| future | cell used by the final route of a net routed after this decision |
| net excess | log(final net delay / ideal) |

## Hyperparameters

### Training

| setting | value | flag |
|---|---|---|
| optimiser | AdamW, weight decay 1e-4, gradient clip 1.0 | |
| learning rate | 3e-4; linear warm-up over 1000 steps; x 0.3 after 70 % of the run, x 0.1 after 90 % | `--lr` |
| batch | 512 on GPUs with >= 80 GB (G4), 256 with >= 40 GB, else 128 | `--batch` |
| precision | bfloat16 autocast on GPU | |
| replay buffer | min(1M, max(20k, 20 % of free RAM / 30 KB)) records (~30 GB at 1M) | `--buffer` |
| sampling | half of each batch uniform, half in proportion to the record's spread (max - min Q over the searched decisions, + 1e-3) | `--prio` |
| replay ratio cap | at most 16 samples trained per record collected | `--replay-ratio` |
| start training | after 2000 records | `--min-records` |
| switch workers to the network | after 1000 training steps (before that: network-free search) | `--warmup-steps` |
| network-free control visits | 10 % of visits after the switch, for a same-hours comparison | `--control-share` |
| publish weights to workers | every 250 steps | |
| augmentation | one of 16 symmetries per batch: flip x, flip y, swap x/y, flip layers + swap dies (pair conflicts are invariant) | |

### Search

| setting | value | flag |
|---|---|---|
| simulations, full-depth decision | 128 (recorded as training data) | `--n-full` |
| simulations, fast decision | 8 budget (not recorded); every remaining candidate gets at least one per round, so ~30 in practice | `--n-fast` |
| share of full-depth decisions | 0.25 | `--p-full` |
| actions considered at the root | 32 on full decisions, 16 on fast ones (Gumbel top-k, then sequential halving) | `--considered`, `--considered-fast` |
| Gumbel noise scale | 1.0 (`--gumbel-scale`) on every decision; it picks the candidates and, through the halving, the executed decision (a worse reroute is rejected, so exploring is cheap) | `--no-explore-fast` turns it off on fast decisions |
| network requests | one per halving round: every candidate's simulation is encoded, rolled back, and evaluated together (identical decisions, ~5x fewer round trips) | |
| sigma constants | c_visit = 50, c_scale = 0.1 | |
| network-free evaluator (warm-up) | completes pending nets (KEEP, else DELAY); unroutable net = 2 x its old delay | |

### Improvement loop

| setting | value |
|---|---|
| neighbourhood size | random 3-8 nets per search reroute; a case whose last visits brought no gain gets up to 8 + 2 x (visits without gain) nets, max 16 (`--k-max`, `--k-max-stalled`) |
| neighbourhood construction | seed net + nets blocking its ideal tree within its box + 3, one more level |
| seed choice | random, weight = excess / (1 + recent failures) |
| reroute region | union of the nets' pin boxes + 6 |
| accept | new neighbourhood delay <= old (training visits) |
| polish visits | `--polish-workers` (default half the workers) only polish scored cases: `--polish-minutes` (30) network-free visits (32/8 simulations, 16 candidates, 3-8 nets), no training records. A worse reroute is kept while the case total stays within `--polish-tau` (0.3 %) of the visit's best total, the margin shrinking linearly to 0 over the visit (record-to-record travel); the visit ends on its best routing, saved with its replay log. The dispatcher sends a polish worker to the free scored case with the highest recent gain per polish minute (never-polished first), the priority growing with the time since its last polish. Measured on the hard tier from a long run's routings, 1 h per case: +2.3 % vs +0.9 % with the strict rule. |
| greedy phase per visit | 3-net neighbourhoods, DELAY mode, until 200 tries in a row bring no gain or 15 s |
| search reroutes per visit | clamp(round(#nets x excess / lower bound), 4, 64); none if excess < 0.3 % |

### Routing modes

| mode | search-cost change |
|---|---|
| KEEP | reinstall the net's previous route |
| DELAY | true delay (ties broken towards fewer cells and away from pending pins' columns) |
| SPARE_MID | cheapest (middle) layers x 1.5 |
| LEARNED_PRICE | + 2.0 x predicted future occupancy |
| OUTER_TRUNK | cheapest layers x 3 |
| FAR_SINK_FIRST | attach the farthest sink first |
| COMPACT | minimum-wire growth |

All modes add tie-breaks of 1e-3 per reserved pin-column cell and 1e-5 per new cell, and fall
back to DELAY if blocked.

### Initial routing

| setting | value |
|---|---|
| method | negotiated congestion, cost = delay x (1 + history) x (1 + pressure x occupancy) |
| schedules (pressure0, pressure multiplier, history factor) | (0.3, 1.2, 0.3), (0.5, 1.3, 0.5), (0.5, 1.5, 0.5); first only on grids > 300k vertices |
| net orders | largest box first, and the reverse |
| iterations | up to 300 |

### Run

| setting | value | flag |
|---|---|---|
| wall time | 23.5 h including export (kept across resumes) | `--hours` |
| reserved for stop + export | 20 min | `--export-minutes` |
| Drive sync | every 30 min | `--sync-minutes` |
| workers | min(CPUs - 4, 64, (free RAM GB - 16) / 3, 0.55 x GPU GB); 44 on a G4 (48 CPUs) | `--workers` |
| cases per worker at once | min(4, #cases / workers) | `--concurrency` |
| network evaluation | one inference process holds the network on the GPU; workers send encoded states, it batches whatever is pending (up to 256, 2 ms gather window) and reloads weights every 2 s if the learner published | |
| network | 10.33 M parameters (`tiny` preset with `--smoke`) | |
