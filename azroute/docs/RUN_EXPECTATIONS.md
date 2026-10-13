# What to expect from a 24 h run

Expectations for `colab_train.ipynb` on a G4 runtime with the default settings
([ARCHITECTURE.md](ARCHITECTURE.md)). Scores are the leaderboard's aggregate per tier:
the geometric mean over cases of baseline delay / our delay (above 1 beats the baseline).
"Lower bound" is the sum of every net's ideal delay with all other nets ignored; no routing
can go below it, and it is usually not reachable.

## Evidence so far (local runs, CPU)

| measurement | result |
|---|---|
| initial routing, every case run so far (intro, hard, scale, congested 1-3, designs) | legal, checker delay = core delay, beats the baseline on every case |
| initial routing aggregate vs baseline | intro 1.10, scale 1.07, hard 1.14, congested 1.11, designs 1.23 |
| initial routing / lower bound | intro 1.06, scale 1.08, hard 1.54, congested 1.48, designs 1.48 |
| greedy rerouting, hard case_01, 20 s | 1.165 -> 1.271 vs baseline, then stalls at 1.31 x lower bound |
| greedy rerouting, hard case_05 | 1.130 -> 1.259 vs baseline |
| network-free search after greedy stalls, hard case_01 / case_05, 150 s | 9414 -> 9382 / 16913 -> 16891: small gains greedy could not find, but greedy kept improving case_05 faster in the same time |
| 5-minute smoke runs, hard case_01 + case_02 | 1.27-1.29 vs baseline |
| network-guided search | runs and stays legal on every tier; with an untrained network most reroutes are rejected (expected) |

Not measured yet: the stress case (530 x 530) end to end, anything on a GPU, and whether a
trained network makes the search beat greedy rerouting.

## Expected outcome

| tier | initial routing (measured) | + greedy rerouting (estimate) | learned search on top (uncertain) |
|---|---:|---:|---|
| intro | ~1.10 | ~1.13-1.17 | about 0: already ~1.06 x lower bound |
| scale | ~1.07 | ~1.10-1.13 | about 0-1 % |
| hard | ~1.14 | ~1.26-1.32 | +0-5 % |
| congested | ~1.11 | ~1.20-1.30 | +0-5 % |
| designs | ~1.23 | ~1.30-1.40 | +0-5 % |
| stress | not run yet | | |

* **Near-certain:** every case legal and above its baseline; the export only writes routings
  the independent m3d checker accepts.
* **Likely:** most of the gain comes in the first hours, from the initial routing plus greedy
  rerouting on every visit. The greedy estimates extrapolate from two hard cases.
* **Uncertain:** whether the network-guided search adds anything beyond that. The contended
  tiers (hard, congested, designs) still sit about 30-40 % above the lower bound after greedy
  rerouting, so there is room; the question is whether the learned search finds it faster
  than plain rerouting does.

## Timeline

| time | what happens |
|---|---|
| 0-1 h | workers build initial routings (seconds for small cases, minutes for congested/designs; the stress case may take tens of minutes), run greedy rerouting, and produce training records with the network-free search |
| ~0.5-2 h | the learner reaches 2000 records and starts training; after 1000 steps (~64k records at replay ratio 8 and batch 512) it hands the network to the workers |
| 2-24 h | network-guided search on every visit; the network is republished every 250 steps; learning rate drops x0.3 at 70 % and x0.1 at 90 % of the run |
| last 20 min | workers stop, state is synced to Drive, checked submissions are exported |

## What to watch (notebook cell 6)

| signal | healthy | if not |
|---|---|---|
| `run.log` first line | sensible workers / batch / buffer for the G4 | set `--workers` / `--batch` |
| `network published to workers at step 1000` | within ~2 h | record rate is low: lower `--n-full` (e.g. 32) or `--min-records` |
| `nvidia-smi` GPU utilisation | anything; low is fine | low with idle workers: more workers; pinned at 100 % with slow visits: `--model small` |
| training losses | policy KL and value MSE fall over the first hours, then flatten | flat from the start: the network is not learning; the run still improves routings via greedy rerouting |
| share of search reroutes that improve | network bar rises above the network-free bar within a few hours | stays below: learned search is not paying off on this budget |
| aggregate vs baseline per tier | rises fast in hour 1, keeps creeping up on hard/congested/designs | flat after hour 2 on every tier: both search variants have stalled |

## Things that could go wrong

* **Stress case initial routing** is the one path never run end to end. It uses a single
  negotiation schedule on grids over 300k vertices; if it fails, that case is logged and
  skipped, and the other 44 continue.
* **Colab disconnects:** the run lives in a background process; if the VM itself is recycled,
  rerun cells 1-5 and it resumes from the last 30-minute Drive sync with its original deadline.
* **Memory:** training stores about 35 MB of activations per sample in bf16 (up to ~70 MB where
  tensors stay fp32), so batch 512 needs roughly 17-35 GB; the auto batch is 512 only on GPUs with
  80 GB or more (G4: 96 GB). Workers take roughly 1 GB of GPU memory and 2-3 GB of RAM each; the
  worker count is capped by both, keeping 45 % of the GPU for the learner.
