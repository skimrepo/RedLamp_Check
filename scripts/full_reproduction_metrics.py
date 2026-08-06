"""
Full paper-reproduction check: the same 5 range-based metrics as
test_set_anomaly_metrics.py (VUS_ROC, VUS_PR, R_AUC_ROC, R_AUC_PR, RF), plus
UCR's peak-in-range accuracy, but scored for EVERY dedicated per-entity model
under anomaly_archive (up to 250) and iops (up to 29) — not just the 3+3
holdout aliases — then averaged per dataset. This is what's directly
comparable to the paper's own Table 3 RedLamp row (UCR / AIOps).

Matches the paper's own methodology (Section 4.1.1): "we trained and tested
separately for each of the subdatasets, and present the average results of
five runs." So each entity is scored under multiple random seeds (default
0-4 via --seeds; main.py already keys each entity's model_dir on --seed, so
training additional seeds needs no code change, just rerunning main.py with
--seed 1/2/3/4). The average is taken in the same two levels as the paper:
first across seeds within an entity, then across entities within a dataset.

Alongside that mean-based aggregate, also writes a "best seed" cherry-picked
ceiling (full_reproduction_metrics_best.csv / _best_summary.csv, see
aggregate_best()): per entity, the max across seeds for EACH metric
independently (so a single entity's VUS_ROC and VUS_PR "best" may come from
two different seeds) — an optimistic upper bound on "how good could this get
if we could always pick the best-performing seed per metric", not the
paper's own methodology.

Robust to partial training: any (entity, seed) pair without a bestmodel.pkl
yet is skipped, not errored, and every output file is rewritten after every
newly-scored (entity, seed) pair — so this can be run repeatedly while
training/scoring is still ongoing to check progress so far.

Does not modify main.py, cross_inference.py, domain_generalization.py, or
continuous_pool_scaling.py — only imports from them.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import main
import datautils
import utils
from loaders.load import load_iops, load_anomaly_archive

import cross_inference as ci
import domain_generalization as dg
import continuous_pool_scaling as cps

from TSB_UAD.vus.metrics import get_metrics


METRIC_KEYS = ['VUS_ROC', 'VUS_PR', 'R_AUC_ROC', 'R_AUC_PR', 'RF']

# RedLamp row values reported in the paper (arXiv:2505.20765), averaged over
# all subdatasets of each domain: VUS_ROC/VUS_PR/RF from Table 3, R_AUC_ROC/
# R_AUC_PR from Table 5. No paper-reported number exists for peak_in_range
# (the UCR-only accuracy from Section 4.1.3), so it is not compared here.
PAPER_REFERENCE = {
    'anomaly_archive': {'VUS_ROC': 0.897, 'VUS_PR': 0.492, 'RF': 0.234, 'R_AUC_ROC': 0.902, 'R_AUC_PR': 0.517},
    'iops':            {'VUS_ROC': 0.911, 'VUS_PR': 0.448, 'RF': 0.235, 'R_AUC_ROC': 0.940, 'R_AUC_PR': 0.636},
}


def discover_dataset_entities(run_name, dataset):
    base = f'./result/{run_name}/{dataset}'
    if not os.path.isdir(base):
        return []
    return sorted(d for d in os.listdir(base)
                  if os.path.isdir(os.path.join(base, d)) and not d.startswith('_'))


def real_ground_truth_labels(dataset, real_name):
    if dataset == 'iops':
        test_ds = load_iops(group='test', filename=real_name, downsampling=1,
                             root_dir='./dataset', validation=False, verbose=False)
    elif dataset == 'anomaly_archive':
        test_ds = load_anomaly_archive(group='test', datasets=real_name, downsampling=1,
                                        root_dir='./dataset', validation=False, verbose=False)
    else:
        raise ValueError(f'unsupported dataset {dataset!r}')
    return test_ds.entities[0].labels.reshape(-1)


def score_entity(run_name, dataset, real_name, seed, params, device, model_dir=None, include_curves=False):
    """Scores one entity's own test set. By default against its own dedicated
    self-model (ci.discover_entity); pass model_dir to score against a fixed
    external model instead (used by full_cross_domain_metrics.py to score a
    domain-excluded pooled model against every entity, without retraining).

    include_curves=True additionally returns the aligned score/real_labels
    arrays (and the point-wise input series) instead of just the 5 scalar
    metrics -- used by scripts/analyze_ds1_gap_entities.py to plot actual
    score curves for a handful of specific entities, without needing every
    caller of this function to pay for carrying full-length arrays around."""
    try:
        own_model_dir, disk_cfg = ci.discover_entity(run_name, dataset, real_name, seed)
    except FileNotFoundError:
        return None
    if model_dir is None:
        model_dir = own_model_dir
    if not os.path.isfile(f'{model_dir}/bestmodel.pkl'):
        return None

    dataparams = ci.build_dataparams(dataset, real_name, dg.CFG, disk_cfg)
    test_dl = datautils.load_dataloader_aug(
        dataparams, anomaly_types=['normal'], anomaly_types_for_dict=ci.ANOMALY_TYPES, group='test_all')
    real_labels = real_ground_truth_labels(dataset, real_name)

    inputs, prediction, anomaly_mask, label, pred_label, pred_enc = main.test(test_dl, model_dir, params, device)
    if include_curves:
        # mse_score/ce_score: the two independently min-max-normalized halves
        # anomaly_scoreing() averages together -- if they don't peak/trough at
        # the same timesteps, the blended score never reaches close to 0 or 1
        # (see analyze_score_oscillation.py, which checks exactly this).
        score, mse_score, ce_score = main.anomaly_scoreing(inputs, prediction, pred_label, return_components=True)
        B = inputs.shape[0]
        mse_raw = main.mse(inputs.reshape(B, -1), prediction.reshape(B, -1))
    else:
        score = main.anomaly_scoreing(inputs, prediction, pred_label)
    _, window_size, _ = inputs.shape
    score = np.concatenate([np.zeros(window_size - 1), score])

    if len(score) != len(real_labels):
        print(f'[skip] length mismatch for {dataset}/{real_name}: '
              f'aligned score={len(score)} vs real labels={len(real_labels)}')
        return None

    if real_labels.sum() == 0:
        # A handful of UCR subdatasets have anomaly_start == anomaly_end (a
        # zero-length window, i.e. no real anomaly in the test split at all).
        # TSB-UAD's range-based metrics assume at least one anomaly region and
        # crash on an empty one (IndexError in RangeAUC_volume_opt), so these
        # entities are excluded from the average rather than the whole run
        # dying on them.
        print(f'[skip] {dataset}/{real_name}: no real anomaly in test labels (range metrics undefined)')
        return None

    all_metrics = get_metrics(score, real_labels, metric='all', slidingWindow=window_size)
    metrics = {k: all_metrics[k] for k in METRIC_KEYS}

    peak_in_range = np.nan
    if dataset == 'anomaly_archive':
        anomaly_idxs = np.where(real_labels == 1)[0]
        if len(anomaly_idxs) > 0:
            peak_idx = int(np.argmax(score))
            peak_in_range = int(anomaly_idxs.min() <= peak_idx <= anomaly_idxs.max())

    result = dict(metrics=metrics, peak_in_range=peak_in_range)
    if include_curves:
        # Point-wise input series: last timestep of each window (window_step=1
        # so consecutive windows overlap by W-1), channel 0, zero-padded front
        # to align with `score`/`real_labels` -- same transform main.py's own
        # (dataset == 'anomaly_archive') plotting block applies.
        raw_series = np.concatenate([np.zeros(window_size - 1), inputs[:, -1, 0]])
        result['score'] = score
        result['real_labels'] = real_labels
        result['raw_series'] = raw_series
        result['mse_score'] = np.concatenate([np.zeros(window_size - 1), mse_score])
        result['ce_score'] = np.concatenate([np.zeros(window_size - 1), ce_score])
        # Model's own reconstruction (x_hat), aligned the same way as raw_series
        # (last timestep of each window) -- used by build_ucr_test_diagnostics.py
        # to overlay reconstruction on top of the raw signal in Panel 1.
        result['reconstruction'] = np.concatenate([np.zeros(window_size - 1), prediction[:, -1, 0]])
        # Pre-normalization MSE (before convolve_minmax_score's smoothing+
        # [0,1] scaling) -- mse_score's normalization is per-entity, so its
        # absolute scale isn't comparable across entities; this is.
        result['mse_raw'] = np.concatenate([np.zeros(window_size - 1), mse_raw])
    return result


def build_comparison(summary):
    comparison_rows = []
    for dataset in summary.index:
        for metric in METRIC_KEYS:
            ours = summary.loc[dataset, metric]
            paper = PAPER_REFERENCE[dataset][metric]
            comparison_rows.append(dict(dataset=dataset, metric=metric, ours=ours, paper=paper, delta=ours - paper))
    return pd.DataFrame(comparison_rows)


def aggregate(raw_rows):
    """Two-level average matching the paper: mean across seeds within an
    entity first (entity_df), then mean across entities within a dataset
    (summary). Returns (entity_df, summary) or (None, None) if raw_rows is
    empty."""
    if not raw_rows:
        return None, None
    raw_df = pd.DataFrame(raw_rows)

    entity_df = raw_df.groupby(['dataset', 'entity'], as_index=False).agg(
        **{k: (k, 'mean') for k in METRIC_KEYS},
        peak_in_range=('peak_in_range', 'mean'),
        n_seeds=('seed', 'nunique'),
    )

    summary = entity_df.groupby('dataset')[METRIC_KEYS].mean()
    summary['n_entities'] = entity_df.groupby('dataset').size()
    summary['avg_seeds_per_entity'] = entity_df.groupby('dataset')['n_seeds'].mean()
    return entity_df, summary


def aggregate_best(raw_rows):
    """Optimistic-ceiling counterpart to aggregate(): per (dataset, entity),
    takes the max across seeds INDEPENDENTLY for each metric (so a single
    entity's VUS_ROC and VUS_PR "best" values may come from two different
    seeds) -- this is "if we could always cherry-pick the best-performing
    seed per metric", not a single winning seed's full row. Then averages
    those per-metric maxes across entities within a dataset, same as
    aggregate()'s mean does. Returns (entity_df, summary) or (None, None) if
    raw_rows is empty."""
    if not raw_rows:
        return None, None
    raw_df = pd.DataFrame(raw_rows)

    entity_df = raw_df.groupby(['dataset', 'entity'], as_index=False).agg(
        **{k: (k, 'max') for k in METRIC_KEYS},
        peak_in_range=('peak_in_range', 'max'),
        n_seeds=('seed', 'nunique'),
    )

    summary = entity_df.groupby('dataset')[METRIC_KEYS].mean()
    summary['n_entities'] = entity_df.groupby('dataset').size()
    summary['avg_seeds_per_entity'] = entity_df.groupby('dataset')['n_seeds'].mean()
    return entity_df, summary


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', default='test')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4],
                         help='Seeds to average over per entity, matching the paper\'s "average results of '
                              'five runs" (Section 4.1.1). Each seed needs its own main.py --seed N training run '
                              'first; seeds not yet trained for a given entity are skipped, not errored.')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--out_csv', default=None)
    parser.add_argument('--force', action='store_true',
                         help='Recompute every (entity, seed) pair even if already present in the raw CSV '
                              '(default: resume, skipping pairs already scored in a prior run).')
    args_cli = parser.parse_args()

    device = utils.init_dl_program(args_cli.gpu, seed=args_cli.seeds[0])
    out_csv = args_cli.out_csv or f'./result/{args_cli.run_name}/full_reproduction_metrics.csv'
    raw_csv = out_csv.replace('.csv', '_raw.csv')
    summary_csv = out_csv.replace('.csv', '_summary.csv')
    comparison_csv = out_csv.replace('.csv', '_vs_paper.csv')
    best_csv = out_csv.replace('.csv', '_best.csv')
    best_summary_csv = out_csv.replace('.csv', '_best_summary.csv')

    # Architecture hyperparams only — params.seed itself is never read during
    # inference (main.test() just loads bestmodel.pkl and runs model.eval()),
    # so one shared params object is reused across every seed's checkpoint.
    model_args = ci.build_model_args(dg.CFG, cps.WINDOW_SIZE)
    params = utils.AttrDict(seed=args_cli.seeds[0])
    params.override(main.model_parameters(model_args))

    rows = []
    already_done = set()
    if not args_cli.force and os.path.isfile(raw_csv):
        prior = pd.read_csv(raw_csv)
        rows = prior.to_dict('records')
        already_done = set(zip(prior['dataset'], prior['entity'], prior['seed']))
        print(f'Resuming from {raw_csv}: {len(already_done)} (entity, seed) pairs already scored, skipping those.')
    elif not args_cli.force and os.path.isfile(out_csv):
        # One-time migration from the pre-multi-seed CSV (one row per entity,
        # no 'seed' column, implicitly seed=0) into the new raw cache — avoids
        # redundantly re-running inference for seed=0 results that already exist.
        old = pd.read_csv(out_csv)
        if 'seed' not in old.columns:
            old['seed'] = 0
            rows = old.to_dict('records')
            already_done = set(zip(old['dataset'], old['entity'], old['seed']))
            print(f'Migrated {len(already_done)} pre-existing seed=0 results from {out_csv} into the raw cache.')

    def save_all():
        pd.DataFrame(rows).to_csv(raw_csv, index=False)
        entity_df, summary = aggregate(rows)
        if entity_df is None:
            return None
        entity_df.to_csv(out_csv, index=False)
        summary.to_csv(summary_csv)
        build_comparison(summary).to_csv(comparison_csv, index=False)

        best_entity_df, best_summary = aggregate_best(rows)
        best_entity_df.to_csv(best_csv, index=False)
        best_summary.to_csv(best_summary_csv)
        return summary

    for dataset in ['anomaly_archive', 'iops']:
        entities = discover_dataset_entities(args_cli.run_name, dataset)
        print(f'{dataset}: found {len(entities)} entity directories, seeds={args_cli.seeds}')
        for real_name in entities:
            for seed in args_cli.seeds:
                if (dataset, real_name, seed) in already_done:
                    continue
                result = score_entity(args_cli.run_name, dataset, real_name, seed, params, device)
                if result is None:
                    print(f'[skip] {dataset}/{real_name} seed={seed}: no bestmodel.pkl yet or scoring failed')
                    continue
                row = dict(dataset=dataset, entity=real_name, seed=seed, **result['metrics'])
                row['peak_in_range'] = result['peak_in_range']
                rows.append(row)
                print(f'  {dataset}/{real_name} seed={seed}: {result["metrics"]}')
                save_all()

    summary = save_all()
    if summary is not None:
        print(summary)
        print(build_comparison(summary).to_string(index=False))
    print(f'Done. {len(rows)} (entity, seed) pairs scored. Wrote {raw_csv}, {out_csv}, {summary_csv}, '
          f'{comparison_csv}, {best_csv}, and {best_summary_csv}')


if __name__ == '__main__':
    run()
