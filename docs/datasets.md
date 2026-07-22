# Datasets at a glance

All numbers below come directly from `main.py`'s per-`--dataset` branch and
`loaders/load.py`. For how a window then gets turned into 12 injected-type
copies, see [`anomaly_injection.md`](./anomaly_injection.md) — this file is
only about the raw data shape and train/test split, before injection.

## Summary table

Train shape is `(n_features, n_time)` per entity, `n_time` varies per entity —
ranges below are **verified by direct inspection**, not estimated (see
"How these numbers were verified" below for exactly what was checked and
what wasn't).

| Dataset | n_features | window_size | window_step | # entities | model granularity | train shape per entity `(n_features, n_time)` |
|---|---|---|---|---|---|---|
| `anomaly_archive` (UCR) | 1 | 100 | **adaptive per entity**: 1 if `train_end<10k`, 10 if `10k≤train_end<100k`, 100 if `train_end≥100k` | 250 series | 1 model per series | `(1, T)` — **not verified**, would require downloading the full ~350MB UCR archive; the code's own 10k/100k step thresholds imply `T` spans roughly 1,000s to 100,000s+ |
| `iops` (AIOps/KPI) | 1 | 100 | 10 (fixed) | 29 KPI curves | 1 model per curve | `(1, T)`, `T ∈ [8,784, 146,255]`, mean ≈ 103,588 (verified: all 29 train files) |
| `smd` (Server Machine) | 38 | 100 | 10 (fixed) | 28 machines | 1 model per machine | `(38, T)`, `T ∈ [23,693, 28,695]` (verified on 4/28 machines sampled — not exhaustive, but SMD machines are known to be similar order of magnitude) |
| `smap` (NASA) | 25 | 100 | 10 (fixed) | **54 channels** (verified count, see note) | **1 shared model across all channels** | `(25, T)`, `T ∈ [312, 2,881]`, mean ≈ 2,555 (verified: all 54 train channels) |
| `msl` (NASA) | 55 | 100 | 10 (fixed) | **27 channels** (verified count, see note) | **1 shared model across all channels** | `(55, T)`, `T ∈ [439, 4,308]`, mean ≈ 2,159 (verified: all 27 train channels) |

Every dataset uses `batch_size=128` and the same validation split rule: the
last 10% of the train portion is held out as `val` (`train_length =
int(Y.shape[1] * 0.9)`, see `loaders/load.py`) — this is the split
`tsne_embeddings.png`/`ts_example_plots` are generated from.

**Note on entity counts**: `main.py`'s hardcoded `each_entity_list` for smap/msl
(used only for the per-channel `test_each` loop) lists ~50/23 names and
contains a duplicate (`'P-2'` appears twice for smap). The 54/27 counts above
come from actually counting `.npy` files inside `dataset/NASA.zip` — that's
the real number of channels that get pooled into the combined training
`Dataset`, and is the more trustworthy number.

### How these numbers were verified

- **SMAP/MSL**: unzipped `dataset/NASA.zip` in-memory (already present in this
  repo) and read every `train/*.npy` file's real shape directly with numpy —
  no estimation.
- **IOPS**: read every `*.train.out` file's line count directly from the
  locally-present `TSB-UAD-Public.zip`.
- **SMD**: fetched 4 of the 28 machine files directly from the dataset's
  public GitHub source (`raw.githubusercontent.com/.../ServerMachineDataset`)
  and counted lines/columns — a partial sample, not all 28.
- **anomaly_archive**: not verified — the full UCR archive is a single
  ~350MB zip with no way to fetch one file at a time, so this wasn't
  downloaded just to report a number here. Ask if you'd like it downloaded
  for a full check.

## `machine` (SMD) vs `channel` (SMAP/MSL) — what's the difference?

Both are just "one Entity" in the code (`loaders/dataset.py`'s `Entity`,
shape `(n_features, n_time)`) — the difference is domain terminology, not
a structural one:

- **SMD "machine"**: one real physical/virtual server. Its 38 features are
  38 different performance metrics (CPU, memory, network, ...) collected
  simultaneously from that one machine over time.
- **SMAP/MSL "channel"**: one telemetry channel from a spacecraft subsystem.
  Per the benchmark's own design, each channel bundles its one monitored
  telemetry value together with a fixed set of encoded command-context
  columns — always 25 columns total for SMAP, 55 for MSL, regardless of
  which specific channel.

The functionally important difference for this codebase isn't the naming —
it's that SMD machines average **~25,000+** timesteps each (long enough to
train their own model), while SMAP/MSL channels average only **~2,000–2,500**
timesteps each (roughly 10x shorter) — which is the concrete, now-verified
reason channels get pooled into one shared model instead of getting a model
each (see the section below).

## Why `anomaly_archive`'s window_step is special

Every other dataset uses a fixed `window_step=10` no matter how long the
series is. `anomaly_archive` is the one case where **the raw series length
varies enormously between entities** — some UCR series are only a few
thousand points, others run past 100,000 — so `main.py` reads each entity's
own `train_end` (parsed straight out of the filename) and picks the stride
accordingly:

```python
if train_end < 10000:
    args.window_step = 1
elif train_end < 100000:
    args.window_step = 10
else:
    args.window_step = 100
```

Short series get windowed densely (step 1, maximize the number of training
windows available); long series get windowed sparsely (step 100) so the
window count doesn't explode. The other four datasets don't need this because
their entities are all roughly the same order of magnitude in length already.

## How smap/msl's shared model is actually built — a worked example

**"channel" is not the same thing as "feature."** A channel (e.g. `A-6`,
`G-3`) is one *entity* — the SMAP/MSL equivalent of one SMD "machine" — and
each channel is itself a multivariate series with 25 (SMAP) / 55 (MSL)
features. Of those, only **1 feature is the actual monitored telemetry
value**; the rest are one-hot flags encoding which spacecraft command was
active at that timestep (this is how the benchmark itself is built, not
something this codebase adds).

Take 4 real SMAP channels (verified shapes, from `dataset/NASA.zip`):

| channel | shape `(features, T)` |
|---|---|
| `A-6` | (25, 682) |
| `G-3` | (25, 2624) |
| `E-3` | (25, 2880) |
| `P-2` | (25, 2821) |

`T` (timesteps) differs per channel, so these **can't be stacked into one
`(4, 25, T)` tensor** — the code just keeps them as a plain Python list of
separate `(25, T_channel)` matrices (54 of them for the real, full SMAP run).

**Windows are sliced independently per channel** (`window_size=100`,
`window_step=10`) — never crossing from one channel into another:

| channel | T | windows produced: `(T-100)/10 + 1` |
|---|---|---|
| `A-6` | 682 | 59 |
| `G-3` | 2624 | 253 |
| `E-3` | 2880 | 279 |
| `P-2` | 2821 | 273 |

Every window, regardless of source channel, comes out as the same `(25, 100)`
shape. **All windows from all channels are then pooled into one flat list**
(just these 4 channels already give 59+253+279+273 = 864 windows, ×12 for
the injected anomaly types; the real run pools all 54 channels). Because
`train_dataloader` shuffles, **one training batch of 128 windows can contain
a mix from completely different channels** — e.g. some from `A-6`, some from
`T-3`, some from `P-4`. The model has no idea which channel a given window
came from; it only ever sees anonymous `(25, 100)` matrices. **Exactly one
model is trained on this entire pooled set.**

MSL works identically — 27 channels, 55 features each, same pooling and
shuffling mechanics.

## How a window is actually fed to the model

1. `load_data(...)` returns each entity's raw series as `Y`: shape
   `(n_features, n_time)` (see `loaders/dataset.py`'s `Entity`).
2. `Loader_aug` slides a `(n_features, window_size=100)` window across `Y`
   with the given `window_step`, and — per window position — injects each of
   the 12 anomaly types (see `anomaly_injection.md` for exactly how).
3. Before hitting the model, `main.py` transposes each window to
   `(window_size, n_features)` (`inputs.transpose(2, 1)`), since `ConvEncoder`
   expects `n_features` as the Conv1d channel dimension (see the multivariate
   explanation earlier in this conversation / `models/cnn.py`).
