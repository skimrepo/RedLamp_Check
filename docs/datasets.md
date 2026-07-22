# Datasets at a glance

All numbers below come directly from `main.py`'s per-`--dataset` branch and
`loaders/load.py`. For how a window then gets turned into 12 injected-type
copies, see [`anomaly_injection.md`](./anomaly_injection.md) — this file is
only about the raw data shape and train/test split, before injection.

## Summary table

| Dataset | n_features | window_size | window_step | # entities | model granularity | train/test split source | normalization |
|---|---|---|---|---|---|---|---|
| `anomaly_archive` (UCR) | 1 | 100 | **adaptive per entity**: 1 if `train_end<10k`, 10 if `10k≤train_end<100k`, 100 if `train_end≥100k` | 250 series | 1 model per series | split point embedded in the filename itself (`train_end`) | MinMaxScaler fit on that series' train portion |
| `iops` (AIOps/KPI) | 1 | 100 | 10 (fixed) | 29 KPI curves | 1 model per curve | separate `{name}.train.out` / `{name}.test.out` files | MinMaxScaler fit on train |
| `smd` (Server Machine) | 38 | 100 | 10 (fixed) | 28 machines | 1 model per machine | separate `train/`, `test/`, `test_label/` files per machine | **none** — already normalized upstream |
| `smap` (NASA) | 25 | 100 | 10 (fixed) | ~50 channels | **1 shared model across all channels** | separate `train/{channel}.npy`, `test/{channel}.npy` + `labeled_anomalies.csv` | MinMaxScaler fit on train, per channel |
| `msl` (NASA) | 55 | 100 | 10 (fixed) | 23 channels | **1 shared model across all channels** | same as smap | same as smap |

Every dataset uses `batch_size=128` and the same validation split rule: the
last 10% of the train portion is held out as `val` (`train_length =
int(Y.shape[1] * 0.9)`, see `loaders/load.py`) — this is the split
`tsne_embeddings.png`/`ts_example_plots` are generated from.

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

## Why smap/msl share one model but smd/anomaly_archive/iops don't

Individual SMAP/MSL channels are comparatively **short** telemetry traces —
too short on their own to train a useful encoder+classifier. So `main.py`
passes `dataparams.entities='smap'` (or `'msl'`) to the loader, which pulls
in *every* channel of that spacecraft and concatenates them into one
`Dataset`, training a single shared model across all ~50 (or 23) channels.
Testing still happens per-channel afterward (the `test_each` loop), just
reusing that one shared model. SMD machines and individual UCR/IOPS series
are long enough that each gets its own dedicated model instead.

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
