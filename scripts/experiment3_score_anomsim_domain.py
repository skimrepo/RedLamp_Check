"""
Experiment 3 scoring: compare Self_A / Cross_without_E / Cross_without_D
checkpoints on target domain A's FIXED held-back test instances (see
AnomSim's scripts/carve_experiment3_fixed_test_ids.py) -- never seen by any
of the three models during training. One ground-truth anomaly is injected
into each test instance ONCE, with a fixed injection_seed shared across every
model being compared (cached to .npz), so all three models are scored
against byte-identical input -- mirroring how full_reproduction_metrics.
score_entity scores a real UCR/KPI entity's single ground-truth anomaly
region, just with AnomSim as the data source instead.

Computes BOTH metric families used elsewhere in this project, so results
stay in the same units as prior experiments:
  - TSB-UAD range-based point-wise detection metrics (VUS_ROC, VUS_PR,
    R_AUC_ROC, R_AUC_PR, RF) -- same as Experiment_2 / full_reproduction_
    metrics.score_entity, via a dense window_step=1 pass
    (local_diagnostic_curves.dense_windows_from_chunk/compute_dense_curves)
    over each test entity's single injected anomaly.
  - 12-class anomaly-type classification accuracy -- same as Experiment_1 /
    Core-Clustering's own classification_accuracy.csv, computed by
    materializing THIS domain's fixed test instances only (reusing
    Core-Clustering's OnlineWindowedDataset/evaluate_classification), not a
    fresh random val_fraction split, so it's the same fixed instances as the
    detection metrics above.

Does not modify main.py, cross_inference.py, domain_generalization.py,
local_diagnostic_curves.py, or any Core-Clustering file -- only imports.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
CORE_CLUSTERING_DEFAULT = os.path.join(REPO_ROOT, '..', 'Core-Clustering')

import main
import utils
import cross_inference as ci
import domain_generalization as dg
import local_diagnostic_curves as ldc

from TSB_UAD.vus.metrics import get_metrics

METRIC_KEYS = ['VUS_ROC', 'VUS_PR', 'R_AUC_ROC', 'R_AUC_PR', 'RF']
WINDOW_SIZE = 100


def load_fixed_test_ids(dataset_dir, domain, filename='_experiment3_fixed_test_ids.json'):
    path = os.path.join(dataset_dir, filename)
    with open(path) as f:
        all_ids = json.load(f)
    if domain not in all_ids:
        raise ValueError(f"domain {domain!r} not found in {path}; available: {list(all_ids)}")
    return all_ids[domain]


def build_labeled_test_entity(dataset_dir, domain, base_instance_id, anomaly_types, injection_seed, get_anomaly):
    """Loads the domain's stored (already [0,1]-normalized) base series and
    injects exactly ONE ground-truth anomaly -- type chosen deterministically
    from (injection_seed, base_instance_id) -- into the WHOLE series,
    mirroring a real UCR/KPI test entity's single contiguous anomaly region.
    Fixed seed so every model scored against this entity sees byte-identical
    input. AnomSim's own mask convention is 0=anomalous/1=normal; TSB-UAD
    expects the opposite (1=anomalous), hence the inversion below."""
    entity_dir = os.path.join(dataset_dir, f'{domain}_b{base_instance_id}')
    Y = np.load(os.path.join(entity_dir, 'Y.npy'))

    rng = np.random.default_rng([injection_seed, int(base_instance_id)])
    anomaly_type = anomaly_types[int(rng.integers(len(anomaly_types)))]
    y_injected, _z, mask = get_anomaly(anomaly_type)().apply(Y, rng)
    real_labels = 1 - np.asarray(mask)[0].astype(int)
    return np.asarray(y_injected), real_labels, anomaly_type


def score_model_on_entity(model, device, y_injected, real_labels, window_size):
    if y_injected.shape[1] < window_size:
        return None
    windows = ldc.dense_windows_from_chunk(y_injected, window_size)
    curves = ldc.compute_dense_curves(windows, model, device)
    score = curves['score']
    if len(score) != len(real_labels):
        print(f'[skip] length mismatch: score={len(score)} vs real_labels={len(real_labels)}')
        return None
    if real_labels.sum() == 0:
        print('[skip] no ground-truth anomaly in real_labels (range metrics undefined)')
        return None
    all_metrics = get_metrics(score, real_labels, metric='all', slidingWindow=window_size)
    return {k: all_metrics[k] for k in METRIC_KEYS}


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset_dir', required=True, help='AnomSim_v2 base-pool directory')
    parser.add_argument('--domain', default='square', help='Target domain (A)')
    parser.add_argument('--model_names', nargs='+', required=True)
    parser.add_argument('--model_dirs', nargs='+', required=True, help='One Core-Clustering checkpoint dir per model_name (same order)')
    parser.add_argument('--seed', type=int, default=0, help='Model-training seed label, recorded in output rows only')
    parser.add_argument(
        '--injection_seed', type=int, default=20260807,
        help='Fixed seed for ground-truth anomaly injection -- must be identical across every '
             'model scored, so all three see byte-identical test entities.',
    )
    parser.add_argument('--core_clustering_dir', default=CORE_CLUSTERING_DEFAULT)
    parser.add_argument('--window_size', type=int, default=WINDOW_SIZE)
    parser.add_argument(
        '--window_step', type=int, default=10,
        help='window_step for the classification-accuracy pass only (detection metrics always '
             'use a dense window_step=1 pass, matching full_reproduction_metrics.score_entity).',
    )
    parser.add_argument('--out_csv', default=None)
    parser.add_argument('--cache_dir', default=None)
    parser.add_argument('--shard_index', type=int, default=0)
    parser.add_argument('--num_shards', type=int, default=1)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    if len(args.model_names) != len(args.model_dirs):
        parser.error('--model_names and --model_dirs must have the same length')

    sys.path.insert(0, args.core_clustering_dir)
    sys.path.insert(0, os.path.join(args.core_clustering_dir, '..', 'AnomSim'))
    from anomsim.anomalies.base import get_anomaly
    from core_clustering.redlamp_compat import REDLAMP_ANOMALY_TYPES
    from core_clustering.online_dataset import load_base_pool, materialize_windows
    from core_clustering.metrics import evaluate_classification

    anomaly_types = [t for t in REDLAMP_ANOMALY_TYPES if t != 'normal']

    out_csv = args.out_csv or f'./result/Experiment_3/Results/experiment3_scores_seed{args.seed}.csv'
    cache_dir = args.cache_dir or './result/Experiment_3/cache'
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    test_ids = load_fixed_test_ids(args.dataset_dir, args.domain)
    if args.num_shards > 1:
        test_ids = [i for j, i in enumerate(test_ids) if j % args.num_shards == args.shard_index]
    print(f'{args.domain}: {len(test_ids)} fixed test instance(s) to score (shard {args.shard_index}/{args.num_shards})')

    device = torch.device('cpu')
    model_args = ci.build_model_args(dg.CFG, args.window_size)
    params = utils.AttrDict(seed=args.seed)
    params.override(main.model_parameters(model_args))

    # ---- Build (once) + cache the labeled test entities, shared by every model ----
    labeled = {}
    for base_id in test_ids:
        cache_path = os.path.join(cache_dir, f'{args.domain}_b{base_id}_labeled_seed{args.injection_seed}.npz')
        if not args.force and os.path.isfile(cache_path):
            data = np.load(cache_path, allow_pickle=True)
            labeled[base_id] = (data['y_injected'], data['real_labels'], str(data['anomaly_type']))
            continue
        y_injected, real_labels, anomaly_type = build_labeled_test_entity(
            args.dataset_dir, args.domain, base_id, anomaly_types, args.injection_seed, get_anomaly)
        np.savez(cache_path, y_injected=y_injected, real_labels=real_labels, anomaly_type=anomaly_type)
        labeled[base_id] = (y_injected, real_labels, anomaly_type)

    # ---- Classification-accuracy pool: this domain's fixed test instances only ----
    manifest_path = os.path.join(args.dataset_dir, '_manifest.jsonl')
    with open(manifest_path) as f:
        all_entity_dirs = [json.loads(line)['entity_dir'] for line in f if line.strip()]
    keep = {f'{args.domain}_b{i}' for i in test_ids}
    exclude = [d for d in all_entity_dirs if d not in keep]
    pool = load_base_pool(args.dataset_dir, exclude_entity_dirs=exclude)

    rows = []
    for model_name, model_dir in zip(args.model_names, args.model_dirs):
        if not os.path.isfile(os.path.join(model_dir, 'bestmodel.pkl')):
            print(f'[skip] {model_name}: no bestmodel.pkl at {model_dir}')
            continue
        model = ldc.load_convaec_model(model_dir, params, device)

        detection_rows = []
        for base_id in test_ids:
            y_injected, real_labels, anomaly_type = labeled[base_id]
            metrics = score_model_on_entity(model, device, np.asarray(y_injected), np.asarray(real_labels), args.window_size)
            if metrics is None:
                print(f'  [skip] {model_name}/{args.domain}_b{base_id}: scoring failed')
                continue
            row = dict(model=model_name, domain=args.domain, base_instance_id=int(base_id),
                       seed=args.seed, anomaly_type=anomaly_type, **metrics)
            detection_rows.append(row)
            print(f'  {model_name}/{args.domain}_b{base_id} ({anomaly_type}): {metrics}')

        acc_result = None
        if len(pool.Y) > 0:
            Y_mat, labels_mat, _ds, _idx = materialize_windows(
                pool, np.arange(len(pool.Y)), args.window_size, args.window_step, REDLAMP_ANOMALY_TYPES)
            acc_result = evaluate_classification(model, Y_mat, labels_mat, args.domain, device=str(device))
            print(f'  {model_name}: classification_accuracy={acc_result.accuracy:.4f} (n={acc_result.n_total})')

        for row in detection_rows:
            row['classification_accuracy'] = acc_result.accuracy if acc_result is not None else np.nan
            row['classification_n_total'] = acc_result.n_total if acc_result is not None else np.nan
            rows.append(row)

    if rows:
        df = pd.DataFrame(rows)
        if not args.force and os.path.isfile(out_csv):
            prior = pd.read_csv(out_csv)
            key_cols = ['model', 'domain', 'base_instance_id', 'seed']
            prior = prior[~prior.set_index(key_cols).index.isin(df.set_index(key_cols).index)]
            df = pd.concat([prior, df], ignore_index=True)
        df.to_csv(out_csv, index=False)
        print(df.groupby('model')[METRIC_KEYS + ['classification_accuracy']].mean())
    print(f'Done. Wrote {out_csv}')


if __name__ == '__main__':
    run()
