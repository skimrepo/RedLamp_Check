"""
Full paper-reproduction check: the same 5 range-based metrics as
test_set_anomaly_metrics.py (VUS_ROC, VUS_PR, R_AUC_ROC, R_AUC_PR, RF), plus
UCR's peak-in-range accuracy, but scored for EVERY dedicated per-entity model
under anomaly_archive (up to 250) and iops (up to 29) — not just the 3+3
holdout aliases — then averaged per dataset. This is what's directly
comparable to the paper's own Table 3 RedLamp row (UCR / AIOps), since the
paper's numbers are themselves an average over all subdatasets.

Robust to partial training: entities without a bestmodel.pkl yet (main.py's
run over the full entity_list is still in progress) are skipped, not errored,
and the CSV + summary are rewritten after every entity — so this can be run
repeatedly while training is still ongoing to check progress so far.

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


def score_entity(run_name, dataset, real_name, seed, params, device):
    model_dir, disk_cfg = ci.discover_entity(run_name, dataset, real_name, seed)
    if not os.path.isfile(f'{model_dir}/bestmodel.pkl'):
        return None

    dataparams = ci.build_dataparams(dataset, real_name, dg.CFG, disk_cfg)
    test_dl = datautils.load_dataloader_aug(
        dataparams, anomaly_types=['normal'], anomaly_types_for_dict=ci.ANOMALY_TYPES, group='test_all')
    real_labels = real_ground_truth_labels(dataset, real_name)

    inputs, prediction, anomaly_mask, label, pred_label, pred_enc = main.test(test_dl, model_dir, params, device)
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

    peak_in_range = ''
    if dataset == 'anomaly_archive':
        anomaly_idxs = np.where(real_labels == 1)[0]
        if len(anomaly_idxs) > 0:
            peak_idx = int(np.argmax(score))
            peak_in_range = int(anomaly_idxs.min() <= peak_idx <= anomaly_idxs.max())

    return dict(metrics=metrics, peak_in_range=peak_in_range)


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', default='test')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--out_csv', default=None)
    args_cli = parser.parse_args()

    device = utils.init_dl_program(args_cli.gpu, seed=args_cli.seed)
    out_csv = args_cli.out_csv or f'./result/{args_cli.run_name}/full_reproduction_metrics.csv'
    summary_csv = out_csv.replace('.csv', '_summary.csv')

    model_args = ci.build_model_args(dg.CFG, cps.WINDOW_SIZE)
    params = utils.AttrDict(seed=args_cli.seed)
    params.override(main.model_parameters(model_args))

    rows = []
    for dataset in ['anomaly_archive', 'iops']:
        entities = discover_dataset_entities(args_cli.run_name, dataset)
        print(f'{dataset}: found {len(entities)} entity directories')
        for real_name in entities:
            result = score_entity(args_cli.run_name, dataset, real_name, args_cli.seed, params, device)
            if result is None:
                print(f'[skip] {dataset}/{real_name}: no bestmodel.pkl yet or scoring failed')
                continue
            row = dict(dataset=dataset, entity=real_name, **result['metrics'])
            row['peak_in_range'] = result['peak_in_range']
            rows.append(row)
            print(f'  {dataset}/{real_name}: {result["metrics"]}')

            df = pd.DataFrame(rows)
            df.to_csv(out_csv, index=False)
            summary = df.groupby('dataset')[METRIC_KEYS].mean()
            summary['n_entities'] = df.groupby('dataset').size()
            summary.to_csv(summary_csv)

    if rows:
        df = pd.DataFrame(rows)
        summary = df.groupby('dataset')[METRIC_KEYS].mean()
        summary['n_entities'] = df.groupby('dataset').size()
        print(summary)
    print(f'Done. {len(rows)} entities scored. Wrote {out_csv} and {summary_csv}')


if __name__ == '__main__':
    run()
