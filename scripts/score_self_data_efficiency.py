"""
Scores every train_self_data_efficiency.py (--entity, --n_pct, --seed) model
against that entity's REAL, FULL, UNTOUCHED test set -- test data is never
truncated, only training data is, so scores are directly comparable across
n_pct for the same entity. Bypasses cross_inference.discover_entity entirely
(unlike full_reproduction_metrics.score_entity) -- these models live at a
custom path, and a matching pre-existing Self baseline at every one of
--seeds isn't guaranteed to exist for every entity, so disk_cfg is built
directly from the SAME window_step train_self_data_efficiency.py computed
for that entity (main.py's own dynamic rule), not looked up from an
existing result/ folder.

Records ALL data, at 3 levels (filenames use --out_prefix, default
"self_data_efficiency"):
  - {prefix}_raw.csv: one row per (entity, n_pct, seed) -- every individual
    result, so any average below can be independently re-derived/verified
    later.
  - {prefix}_per_entity_avg.csv: one row per (entity, n_pct) -- mean/std
    across seeds.
  - {prefix}_overall_summary.csv: one row per n_pct -- mean/std of the
    per-entity seed-averaged metric ACROSS ALL SCORED ENTITIES. This is the
    "how does each metric change with n_pct, across UCR as a whole" table
    -- plot_self_data_efficiency.py reads this one directly.

--score_mode mse_only scores with pure reconstruction error instead of the
default (mse_score+ce_score)/2 blend -- use this for models trained with
train_self_data_efficiency.py --c_loss_ratio 0 (whose classifier head never
got a gradient), and pass a distinct --out_prefix (e.g.
"self_reconstruction_only") so its CSVs don't overwrite the combined-loss
run's.

Resumable: skips (entity, n_pct, seed) triples already in the raw CSV
unless --force; entities/n_pct combos with no bestmodel.pkl (not yet
trained, or skipped for being too short) are silently skipped, not errored.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import main
import utils
import datautils
import cross_inference as ci
import domain_generalization as dg
import full_reproduction_metrics as frm
from TSB_UAD.vus.metrics import get_metrics

DATASET = 'anomaly_archive'
DEFAULT_ENTITIES = [str(i).zfill(3) for i in range(1, 251)]
DEFAULT_N_PCTS = [1, 3, 5, 10, 25, 50, 75, 100]
DEFAULT_SEEDS = [0, 1, 2]
METRIC_KEYS = frm.METRIC_KEYS


def window_step_for(train_end):
    if train_end < 10000:
        return 1
    elif train_end < 100000:
        return 10
    else:
        return 100


def fmt_pct(n_pct):
    return str(int(n_pct)) if float(n_pct).is_integer() else str(n_pct)


def score_custom(entity, model_dir, window_step, params, device, score_mode='combined'):
    """Mirrors full_reproduction_metrics.score_entity's body exactly, minus
    the ci.discover_entity lookup -- disk_cfg is passed in directly instead
    of being discovered from an existing result/ folder.

    score_mode='combined' (default): main.anomaly_scoreing's usual
    (mse_score+ce_score)/2 blend -- use this for normally-trained models.
    score_mode='mse_only': pure reconstruction-error score, no classifier
    signal at all -- use this for models trained with train_self_data_efficiency.py
    --c_loss_ratio 0, whose classifier head never received a gradient and
    would just inject noise into the blended score."""
    disk_cfg = dict(downsampling=1, batch_size=128, window_size=100, window_step=window_step)
    dataparams = ci.build_dataparams(DATASET, entity, dg.CFG, disk_cfg)
    test_dl = datautils.load_dataloader_aug(
        dataparams, anomaly_types=['normal'], anomaly_types_for_dict=ci.ANOMALY_TYPES, group='test_all')
    real_labels = frm.real_ground_truth_labels(DATASET, entity)

    inputs, prediction, anomaly_mask, label, pred_label, pred_enc = main.test(test_dl, model_dir, params, device)
    if score_mode == 'mse_only':
        _, score, _ = main.anomaly_scoreing(inputs, prediction, pred_label, return_components=True)
    else:
        score = main.anomaly_scoreing(inputs, prediction, pred_label)
    _, window_size, _ = inputs.shape
    score = np.concatenate([np.zeros(window_size - 1), score])

    if len(score) != len(real_labels):
        print(f'[skip] {entity}: length mismatch (score={len(score)} vs real_labels={len(real_labels)})')
        return None
    if real_labels.sum() == 0:
        print(f'[skip] {entity}: no real anomaly in test labels (range metrics undefined)')
        return None

    all_metrics = get_metrics(score, real_labels, metric='all', slidingWindow=window_size)
    metrics = {k: all_metrics[k] for k in METRIC_KEYS}

    peak_in_range = np.nan
    anomaly_idxs = np.where(real_labels == 1)[0]
    if len(anomaly_idxs) > 0:
        peak_idx = int(np.argmax(score))
        peak_in_range = int(anomaly_idxs.min() <= peak_idx <= anomaly_idxs.max())
    return dict(metrics=metrics, peak_in_range=peak_in_range)


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--entities', nargs='+', default=DEFAULT_ENTITIES)
    parser.add_argument('--n_pcts', type=float, nargs='+', default=DEFAULT_N_PCTS)
    parser.add_argument('--seeds', type=int, nargs='+', default=DEFAULT_SEEDS)
    parser.add_argument('--models_dir', default='./result/DS_2/achievability/self_data_efficiency_models')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--out_dir', default='./result/DS_2/achievability')
    parser.add_argument('--out_prefix', default='self_data_efficiency',
                         help='Filename prefix for the 3 output CSVs -- pass something like '
                              '"self_reconstruction_only" when scoring --score_mode mse_only models so they '
                              'don\'t overwrite the combined-loss run\'s CSVs.')
    parser.add_argument('--score_mode', choices=['combined', 'mse_only'], default='combined',
                         help='mse_only for models trained with --c_loss_ratio 0 (see train_self_data_efficiency.py)')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    raw_csv = os.path.join(args.out_dir, f'{args.out_prefix}_raw.csv')
    per_entity_csv = os.path.join(args.out_dir, f'{args.out_prefix}_per_entity_avg.csv')
    overall_csv = os.path.join(args.out_dir, f'{args.out_prefix}_overall_summary.csv')

    device = utils.init_dl_program(args.gpu, seed=args.seeds[0])
    model_args = ci.build_model_args(dg.CFG, dg.WINDOW_SIZE)

    rows = []
    already_done = set()
    if not args.force and os.path.isfile(raw_csv):
        prior = pd.read_csv(raw_csv)
        rows = prior.to_dict('records')
        already_done = set(zip(prior['entity'], prior['n_pct'], prior['seed']))
        print(f'Resuming: {len(already_done)} (entity, n_pct, seed) already scored.')

    os.makedirs(args.out_dir, exist_ok=True)
    for entity in args.entities:
        try:
            train_end = int(main.get_meta_data(entity)['train_end'])
        except Exception as e:
            print(f'[skip] {entity}: no metadata ({e})')
            continue
        window_step = window_step_for(train_end)

        for n_pct in args.n_pcts:
            for seed in args.seeds:
                if (entity, n_pct, seed) in already_done:
                    continue
                model_dir = os.path.join(args.models_dir, entity, f'n{fmt_pct(n_pct)}_seed{seed}')
                if not os.path.isfile(os.path.join(model_dir, 'bestmodel.pkl')):
                    continue  # not trained yet, or skipped as too-short -- not an error

                params = utils.AttrDict(seed=seed)
                params.override(main.model_parameters(model_args))
                result = score_custom(entity, model_dir, window_step, params, device, score_mode=args.score_mode)
                if result is None:
                    continue

                row = dict(entity=entity, n_pct=n_pct, seed=seed, **result['metrics'])
                row['peak_in_range'] = result['peak_in_range']
                rows.append(row)
                print(f'{entity}/n{fmt_pct(n_pct)}/seed{seed}: VUS_ROC={row["VUS_ROC"]:.4f}')

        pd.DataFrame(rows).to_csv(raw_csv, index=False)  # incremental save, one entity at a time

    raw_df = pd.DataFrame(rows)
    if raw_df.empty:
        print('No results scored -- nothing to aggregate.')
        return
    print(f'Wrote {raw_csv} ({len(raw_df)} rows)')

    per_entity = raw_df.groupby(['entity', 'n_pct'], as_index=False).agg(
        **{f'{k}_mean': (k, 'mean') for k in METRIC_KEYS},
        **{f'{k}_std': (k, 'std') for k in METRIC_KEYS},
        peak_in_range_mean=('peak_in_range', 'mean'),
        n_seeds=('seed', 'nunique'),
    )
    per_entity.to_csv(per_entity_csv, index=False)
    print(f'Wrote {per_entity_csv} ({len(per_entity)} rows)')

    overall_rows = []
    for n_pct in args.n_pcts:
        sub = per_entity[per_entity['n_pct'] == n_pct]
        if sub.empty:
            continue
        row = dict(n_pct=n_pct, n_entities=len(sub))
        for k in METRIC_KEYS:
            row[f'{k}_mean'] = sub[f'{k}_mean'].mean()
            row[f'{k}_std'] = sub[f'{k}_mean'].std()
        row['peak_in_range_mean'] = sub['peak_in_range_mean'].mean()
        overall_rows.append(row)
    overall_df = pd.DataFrame(overall_rows).sort_values('n_pct')
    overall_df.to_csv(overall_csv, index=False)
    print(f'Wrote {overall_csv} ({len(overall_df)} rows)')
    print(overall_df.to_string(index=False))


if __name__ == '__main__':
    run()
