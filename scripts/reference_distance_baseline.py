"""
Reference-Distance baseline: a model-free anomaly score built purely from
z-normalized nearest-neighbor distance (Matrix-Profile-style) to a sample of
an entity's own TRAIN-split windows -- guaranteed anomaly-free by
construction for UCR/KPI (see main.get_meta_data()'s train_end/
anomaly_start_in_test fields for anomaly_archive; iops keeps train/test in
disjoint .train.out/.test.out files entirely). No ConvAEC, no GPU, no
bestmodel.pkl -- pure data pipeline (datautils' dataloaders) + numpy/scipy.

Why: DS_1's score_comparison.png plots showed Cross-AnomSim's score
oscillating with an entity's own normal periodic waveform rather than
tracking the real anomaly, because it never saw that entity's actual normal
shape (only synthetic AnomSim data). This script tests the cheapest possible
version of "what if we just gave the model this entity's own normal
reference" -- if nearest-neighbor distance to real train windows alone gets
close to Self's VUS-ROC, that's strong evidence a reference-conditioned
architecture (vs. just diversifying AnomSim's synthetic waveforms) is the
better investment. If it doesn't help, the opposite.

Scope matches DS_1 exactly (UCR/anomaly_archive only): reads
result/DS_1/entity_metadata.csv for the entity list. Resumable: entities
already in the output CSV are skipped unless --force.

Does not modify main.py, cross_inference.py, datautils.py, or
full_reproduction_metrics.py -- only imports from them.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import main
import datautils

import cross_inference as ci
import full_reproduction_metrics as frm

from TSB_UAD.vus.metrics import get_metrics

DATASET = 'anomaly_archive'
CFG = dict(n_features=1, min_features=1, max_features=1)
METRIC_KEYS = ['VUS_ROC', 'VUS_PR', 'R_AUC_ROC', 'R_AUC_PR', 'RF']


def _collect_windows(dataloader):
    """(n_windows, window_size) real-valued windows, channel 0 only (n_features=1
    for anomaly_archive/iops) -- same transpose main.test() applies to batch['Y']."""
    windows = []
    for batch in dataloader:
        y = batch['Y'].transpose(2, 1).numpy()  # (batch, window, n_features)
        windows.append(y[:, :, 0])
    return np.concatenate(windows, axis=0) if windows else np.empty((0, 0))


def _znorm(windows):
    mean = windows.mean(axis=1, keepdims=True)
    std = windows.std(axis=1, keepdims=True)
    return (windows - mean) / (std + 1e-8)


def _min_distance_to_reference(test_windows, reference_windows, chunk_size=5000):
    """Chunked cdist so peak memory stays bounded even for the longest UCR
    entities (~190k timesteps) -- avoids materializing one giant
    n_test x n_reference distance matrix at once."""
    n = test_windows.shape[0]
    out = np.empty(n)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        out[start:end] = cdist(test_windows[start:end], reference_windows).min(axis=1)
    return out


def score_entity_reference_distance(run_name, entity, seed, max_reference_windows, rng, include_curves=False):
    """include_curves=True additionally returns 'score'/'real_labels'/'raw_series'
    (same shape as full_reproduction_metrics.score_entity's own include_curves
    option) so callers can plot this alongside Self/Cross-OpenSource/
    Cross-AnomSim's curves without any special-casing."""
    _, disk_cfg = ci.discover_entity(run_name, DATASET, entity, seed)
    dataparams = ci.build_dataparams(DATASET, entity, CFG, disk_cfg)

    # anomaly_types=['normal'] disables Loader_aug's pseudo-anomaly injection
    # entirely (Loader_aug._inject_anomalies only builds windows for the types
    # listed) -- both reference and query windows are real, unmodified data.
    train_dl, val_dl = datautils.load_dataloader_aug(
        dataparams, anomaly_types=['normal'], anomaly_types_for_dict=ci.ANOMALY_TYPES, group='train')
    reference_windows = np.concatenate(
        [_collect_windows(train_dl), _collect_windows(val_dl)], axis=0)
    if reference_windows.shape[0] == 0:
        print(f'[skip] {entity}: no train windows found')
        return None
    if reference_windows.shape[0] > max_reference_windows:
        idx = rng.choice(reference_windows.shape[0], size=max_reference_windows, replace=False)
        reference_windows = reference_windows[idx]
    reference_windows = _znorm(reference_windows)

    test_dl = datautils.load_dataloader_aug(
        dataparams, anomaly_types=['normal'], anomaly_types_for_dict=ci.ANOMALY_TYPES, group='test_all')
    test_windows = _collect_windows(test_dl)
    window_size = test_windows.shape[1]
    real_labels = frm.real_ground_truth_labels(DATASET, entity)

    dist_score = _min_distance_to_reference(_znorm(test_windows), reference_windows)
    dist_score = np.concatenate([np.zeros(window_size - 1), dist_score])
    dist_score = main.convolve_minmax_score(dist_score, w=int(window_size / 2))

    if len(dist_score) != len(real_labels):
        print(f'[skip] {entity}: length mismatch (score={len(dist_score)}, labels={len(real_labels)})')
        return None
    if real_labels.sum() == 0:
        print(f'[skip] {entity}: no real anomaly in test labels (range metrics undefined)')
        return None

    all_metrics = get_metrics(dist_score, real_labels, metric='all', slidingWindow=window_size)
    metrics = {k: all_metrics[k] for k in METRIC_KEYS}
    anomaly_idxs = np.where(real_labels == 1)[0]
    peak_idx = int(np.argmax(dist_score))
    peak_in_range = int(anomaly_idxs.min() <= peak_idx <= anomaly_idxs.max())

    if include_curves:
        raw_series = np.concatenate([np.zeros(window_size - 1), test_windows[:, -1]])
        return dict(metrics=metrics, peak_in_range=peak_in_range,
                    score=dist_score, real_labels=real_labels, raw_series=raw_series)
    return dict(metrics=metrics, peak_in_range=peak_in_range)


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', default='test')
    parser.add_argument('--seed', type=int, default=0,
                         help='Which seed\'s trained model_dir to resolve disk_cfg from -- no model '
                              'weights are ever loaded, this only recovers window_size/downsampling.')
    parser.add_argument('--entity_metadata_csv', default='./result/DS_1/entity_metadata.csv')
    parser.add_argument('--max_reference_windows', type=int, default=300)
    parser.add_argument('--rng_seed', type=int, default=0)
    parser.add_argument('--out_csv', default='./result/DS_2/reference_distance_metrics.csv')
    parser.add_argument('--force', action='store_true',
                         help='Recompute every entity even if already present in out_csv.')
    args = parser.parse_args()

    out_dir = os.path.dirname(args.out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    entities = pd.read_csv(args.entity_metadata_csv)['entity'].astype(str).str.zfill(3).tolist()
    print(f'{len(entities)} entities loaded from {args.entity_metadata_csv}')

    rows = []
    already_done = set()
    if not args.force and os.path.isfile(args.out_csv):
        prior = pd.read_csv(args.out_csv)
        rows = prior.to_dict('records')
        already_done = set(prior['entity'].astype(str).str.zfill(3))
        print(f'Resuming from {args.out_csv}: {len(already_done)} entities already scored, skipping those.')

    rng = np.random.default_rng(args.rng_seed)
    for entity in entities:
        if entity in already_done:
            continue
        try:
            result = score_entity_reference_distance(
                args.run_name, entity, args.seed, args.max_reference_windows, rng)
        except FileNotFoundError:
            print(f'[skip] {entity}: no trained model dir found (needed only for disk_cfg)')
            continue
        if result is None:
            continue
        row = dict(entity=entity, **result['metrics'])
        row['peak_in_range'] = result['peak_in_range']
        rows.append(row)
        print(f'  {entity}: {result["metrics"]}, peak_in_range={result["peak_in_range"]}')
        pd.DataFrame(rows).to_csv(args.out_csv, index=False)

    print(f'Done. {len(rows)} entities scored. Wrote {args.out_csv}')


if __name__ == '__main__':
    run()
