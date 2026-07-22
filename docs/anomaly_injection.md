# How pseudo-anomaly injection works (multivariate case)

This documents the exact mechanics of `loaders/loader_aug.py`'s `Loader_aug`,
specifically how it decides *which* features get an anomaly and *where*,
for multivariate datasets (SMD: 38 features, SMAP: 25, MSL: 55). Univariate
datasets (`anomaly_archive`, `iops`, both `n_features=1`, `min_features=max_features=1`)
are the trivial case of everything below (always exactly 1 feature, the only one).

## 1. One window position → 12 independent copies, one per type

`Loader_aug._inject_anomalies()` slides across an entity's time series
(`window_step` stride) and, **for every window position and every configured
`anomaly_type`**, calls `select_anomalies(anomaly_type, Y, window_start, window_end)`
once. Each call produces one self-contained window copy labeled with exactly
one type — types are never mixed within a single window copy. The "same" time
range therefore appears many times in the dataset: once as `normal`, once as
`spike`, once as `flip`, etc.

## 2. How many features get hit, within one window+type call

Most injection functions (`flip`, `speedup`, `noise`, `cutoff`, `average`,
`scale`, `wander`, `contextual`, `upsidedown`, `mixture`) share this pattern:

```python
if min_features == max_features:
    n_anom_features = max_features
else:
    n_anom_features = np.random.randint(low=min_features, high=max_features, size=1)[0] + 1
loc_features_list = np.random.randint(low=0, high=n_features, size=n_anom_features)
```

For SMD (`min_features=1`, `max_features=n_features=38`, set in `main.py`):
`n_anom_features = randint(1, 38) + 1` → **uniformly random in [2, 38]**,
average ≈ **20 out of 38 features** — a much larger fraction than intuition
might suggest.

`loc_features_list` is drawn **with replacement** (`np.random.randint`), so a
feature index can repeat. `n_anom_features` is the number of *draws*, not a
guaranteed count of *distinct* features — the actual distinct-feature count
can be slightly lower, especially when `n_anom_features` is large relative to
`n_features` (as it typically is for SMD).

`spike` is the one exception to the for-loop pattern below: it vectorizes
`loc_time`/`loc_features` instead of looping, and adds a **single shared**
random magnitude (`np.random.normal(size=1)`) to every selected location —
positions differ per feature, magnitude does not.

## 3. Do different features get hit at different positions/strengths?

**Yes.** All of the for-loop-based functions look like this:

```python
for loc_feature in loc_features_list:
    anomaly_start = np.random.randint(...)   # independent per feature
    anomaly_end = np.random.randint(...)     # independent per feature
    # speedup also redraws its own freq (0.5x or 2x) per feature here
    # scale/wander/contextual also redraw their own magnitude per feature here
    ...
```

Each feature in `loc_features_list` gets its **own independently-sampled
sub-range** within the window (and, depending on the type, its own
independently-sampled magnitude/frequency/drift too). So within one window
copy labeled e.g. `speedup`, feature 3 might be sped up over `[10:40]` while
feature 17 is slowed down over `[60:95]` — same type name, different position
and even different direction, per feature.

The only case where every affected feature shares the exact same range is
`min_range == window_size` (whole-window anomaly) — not the case with the
current default config (`--min_range 1`).

## 4. `Z` (the "normal" counterfactual) vs `Y` (the injected version)

`select_anomalies()` returns `(Y_temp, Z_temp, mask_temp)`. `Y_temp` is what
actually gets used as model input; `Z_temp` is meant to represent "what this
window would look like if it were normal." For **every injection type except
`speedup`**, `Z_temp` is simply the untouched original signal. For `speedup`,
`Z_temp`'s anomaly region is flattened to its own mean instead of the true
original — because a sped-up/slowed-down segment doesn't have a single
well-defined "normal" value at each original timestamp, the authors used the
segment mean as a placeholder reconstruction target.

Note `main.py`'s training loop never reads `batch['Z']` — only `batch['Y']`,
`batch['anomaly_mask']`, and `batch['label']` feed into the loss. `Z` is
otherwise unused by training, but it's exactly what you want for a visual
"before vs after" comparison (see `ts_example_plots/`).

## Summary table

| Question | Answer |
|---|---|
| Same type across all affected features in one window? | Yes — type is fixed per window copy |
| Same position for every affected feature? | No — each feature independently samples its own sub-range (except whole-window mode) |
| How many features, typically (SMD)? | Uniform in [2, 38], average ≈ 20 |
| Exactly N *distinct* features guaranteed? | No — feature indices are drawn with replacement, so duplicates (and thus fewer distinct features) are possible |
| Is `Z` always the true original? | Yes, except `speedup`, where the anomaly region is replaced with its own mean |
