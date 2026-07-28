"""
Real test-set anomaly detection metrics (VUS-PR etc.) for KPI/UCR, matching
the RedLamp paper's own evaluation protocol (arXiv:2505.20765, Section 4.1.3):
5 threshold-independent range-based metrics (Range F-score, Range-AUC-ROC,
Range-AUC-PR, VUS-ROC, VUS-PR) via the TSB-UAD reference implementation,
plus a UCR-only "peak-in-range" detection accuracy (an anomaly is considered
detected if the highest anomaly score falls within the true anomaly range).

Install (only the `vus` submodule is used here, which only needs numpy/sklearn
— skip tsb-uad's other heavy declared deps like tensorflow with --no-deps):
    pip install tsb-uad --no-deps

Scores 39 (model, data) pairs:
  - 6 self-baselines: each of kpi_1..3/ucr_1..3's own dedicated model on its
    own test set.
  - 9 continuous_n{3,5,10,50,100,200,400,800,944} pooled models x kpi_1..3
    (fair — none of these pools ever contained any KPI/IOPS data).
  - 1 continuous_n697_excl_ucr pooled model x (kpi_1..3 + ucr_1..3) (fair for
    both — this pool contains neither KPI nor UCR at all).
  continuous_n* models are NOT scored against ucr_1..3, since those pools
  contain other UCR series (contaminating that specific comparison).

Reuses main.test()/main.anomaly_scoreing() directly (both already fully
generic over model_dir/test_dataloader) and cross_inference.py/
domain_generalization.py's entity resolution helpers. Does not modify any
existing file. No retraining — this only runs inference (main.test()) over
already-trained checkpoints.
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


CONTINUOUS_N_VALUES = [3, 5, 10, 50, 100, 200, 400, 800, 944]
NO_UCR_N = 697
METRIC_KEYS = ['VUS_ROC', 'VUS_PR', 'R_AUC_ROC', 'R_AUC_PR', 'RF']


def real_ground_truth_labels(dataset, real_name):
    """Load the real per-timestep ground truth for this entity's test split,
    directly from the loader (bypassing Loader_aug entirely, which never
    threads real labels through anywhere in the existing pipeline)."""
    if dataset == 'iops':
        test_ds = load_iops(group='test', filename=real_name, downsampling=1,
                             root_dir='./dataset', validation=False, verbose=False)
    elif dataset == 'anomaly_archive':
        test_ds = load_anomaly_archive(group='test', datasets=real_name, downsampling=1,
                                        root_dir='./dataset', validation=False, verbose=False)
    else:
        raise ValueError(f'unsupported dataset {dataset!r}')
    return test_ds.entities[0].labels.reshape(-1)


def build_test_dataloader(run_name, dataset, real_name, seed):
    _, disk_cfg = ci.discover_entity(run_name, dataset, real_name, seed)
    dataparams = ci.build_dataparams(dataset, real_name, dg.CFG, disk_cfg)
    return datautils.load_dataloader_aug(
        dataparams, anomaly_types=['normal'], anomaly_types_for_dict=ci.ANOMALY_TYPES, group='test_all')


def score_pair(model_dir, params, device, test_dl, real_labels):
    inputs, prediction, anomaly_mask, label, pred_label, pred_enc = main.test(test_dl, model_dir, params, device)
    score = main.anomaly_scoreing(inputs, prediction, pred_label)
    _, window_size, _ = inputs.shape
    score = np.concatenate([np.zeros(window_size - 1), score])

    if len(score) != len(real_labels):
        raise ValueError(f'length mismatch for {model_dir}: aligned score={len(score)} vs '
                          f'real labels={len(real_labels)} — do not silently truncate, investigate')

    anomaly_idxs = np.where(real_labels == 1)[0]
    if len(anomaly_idxs) == 0:
        # TSB-UAD's range-based metrics assume at least one anomaly region and
        # crash (IndexError) on an all-normal test split — not expected among
        # these 6 hand-picked entities, but guarded here for consistency with
        # full_reproduction_metrics.py, which hits this on some UCR entities.
        return None, None

    all_metrics = get_metrics(score, real_labels, metric='all', slidingWindow=window_size)
    metrics = {k: all_metrics[k] for k in METRIC_KEYS}

    peak_idx = int(np.argmax(score))
    peak_in_range = int(anomaly_idxs.min() <= peak_idx <= anomaly_idxs.max())
    return metrics, peak_in_range


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', default='test')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--out_csv', default=None)
    args_cli = parser.parse_args()

    device = utils.init_dl_program(args_cli.gpu, seed=args_cli.seed)
    out_csv = args_cli.out_csv or f'./result/{args_cli.run_name}/test_set_metrics.csv'

    model_args = ci.build_model_args(dg.CFG, cps.WINDOW_SIZE)
    params = utils.AttrDict(seed=args_cli.seed)
    params.override(main.model_parameters(model_args))

    # Build each of the 6 data entities' test_all dataloader + real ground
    # truth once (shared across every model scored against that data column).
    print('Resolving the 6 data entities (kpi_1..3, ucr_1..3)...')
    data_info = {}
    for alias, (dataset, real_name) in dg.ENTITY_ALIASES.items():
        test_dl = build_test_dataloader(args_cli.run_name, dataset, real_name, args_cli.seed)
        real_labels = real_ground_truth_labels(dataset, real_name)
        self_model_dir, _ = ci.discover_entity(args_cli.run_name, dataset, real_name, args_cli.seed)
        data_info[alias] = dict(dataset=dataset, test_dl=test_dl, real_labels=real_labels,
                                 self_model_dir=self_model_dir, is_ucr=(dataset == 'anomaly_archive'))
        print(f'  {alias}: real test length={len(real_labels)}, anomalies={int(real_labels.sum())}')

    rows = []

    def add_row(model_alias, data_alias, model_dir):
        if not os.path.isfile(f'{model_dir}/bestmodel.pkl'):
            print(f'[skip] {model_dir}/bestmodel.pkl does not exist yet — model={model_alias} data={data_alias}')
            return
        info = data_info[data_alias]
        print(f'Scoring model={model_alias} data={data_alias}...')
        metrics, peak_in_range = score_pair(model_dir, params, device, info['test_dl'], info['real_labels'])
        if metrics is None:
            print(f'[skip] {data_alias}: no real anomaly in test labels (range metrics undefined)')
            return
        row = dict(model=model_alias, data=data_alias, **metrics)
        row['peak_in_range'] = peak_in_range if info['is_ucr'] else ''
        rows.append(row)
        print(f'  -> {metrics}')
        # Write after every row, not just at the end — so a run started before
        # every model exists yet (e.g. before the UCR-free pool finishes
        # training) still saves everything computed so far instead of losing
        # it all if a later row's model is missing.
        pd.DataFrame(rows).to_csv(out_csv, index=False)

    # 6 self-baselines: each entity's own dedicated model on its own test set.
    for alias, info in data_info.items():
        add_row(f'{alias}_self', alias, info['self_model_dir'])

    # 9 continuous_n* pooled models, scored against kpi_1..3 only (fair — no
    # KPI/IOPS data was ever in any of these pools).
    for n in CONTINUOUS_N_VALUES:
        model_dir = f'./result/{args_cli.run_name}/_pooled/continuous_n{n}/{args_cli.seed}'
        for alias in ['kpi_1', 'kpi_2', 'kpi_3']:
            add_row(f'continuous_n{n}', alias, model_dir)

    # 1 UCR-free pooled model, scored against all 6 (fair for both KPI and UCR).
    no_ucr_model_dir = f'./result/{args_cli.run_name}/_pooled/continuous_n{NO_UCR_N}_excl_ucr/{args_cli.seed}'
    for alias in dg.DATA_ALIASES:
        add_row(f'continuous_n{NO_UCR_N}_excl_ucr', alias, no_ucr_model_dir)

    print(f'Done. Wrote {len(rows)}/39 rows to {out_csv}')


if __name__ == '__main__':
    run()
