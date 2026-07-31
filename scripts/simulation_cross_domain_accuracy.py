"""
"Cross-AnomSim" classification accuracy: score a model trained ENTIRELY on
AnomSim-generated synthetic data (via Core-Clustering, a sibling project --
never trained on any real RedLamp series) against every UCR/KPI entity's own
validation split, using the same 12-class pseudo-anomaly classification
accuracy metric as self_accuracy_report.py (Self) and
full_cross_domain_accuracy.py (Cross-OpenSource) -- so Self vs
Cross-OpenSource vs Cross-AnomSim are all directly comparable in the same
units, matching Experiment_1's metric.

This is the classification-accuracy counterpart to
simulation_cross_domain_metrics.py (which computes the paper's VUS-based
real test-set metrics for the same kind of externally-trained checkpoint) --
same --sim_model_dir / run_summary.json compatibility-check pattern, but
scoring logic borrowed from full_cross_domain_accuracy.py instead (entity's
own val split via ci.discover_entity + datautils.load_dataloader_aug,
self_accuracy_report.compute_self_accuracy for the actual accuracy number).

No retraining: pure inference over an already-trained Core-Clustering
checkpoint. Resumable: reruns skip entities already present in out_csv
unless --force is passed.

Does not modify main.py, cross_inference.py, datautils.py,
self_accuracy_report.py, full_cross_domain_accuracy.py, or
simulation_cross_domain_metrics.py -- only imports from them.
"""
import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import main
import datautils
import utils
import cross_inference as ci

import full_reproduction_metrics as frm
import self_accuracy_report as sar
import continuous_pool_scaling as cps

CFG = dict(n_features=1, min_features=1, max_features=1)  # shared by both UCR and KPI

# Must match Core-Clustering's core_clustering/redlamp_compat.py verbatim --
# this IS main.py's own default anomaly_types order (main.py:480).
REDLAMP_ANOMALY_TYPES = ['normal', 'spike', 'flip', 'speedup', 'noise', 'cutoff',
                         'average', 'scale', 'wander', 'contextual', 'upsidedown', 'mixture']


def check_sim_model_compatible(sim_model_dir):
    """Same best-effort check as simulation_cross_domain_metrics.py's
    function of the same name -- kept as a separate copy rather than an
    import so this script has no dependency on that one, since they may be
    run independently."""
    summary_path = os.path.join(sim_model_dir, 'run_summary.json')
    if not os.path.isfile(summary_path):
        print(f'[warn] no run_summary.json next to {sim_model_dir} -- skipping '
              f'hyperparameter compatibility check, proceeding on trust')
        return
    with open(summary_path) as f:
        summary = json.load(f)
    hp = summary.get('model_hyperparameters', {})
    problems = []
    if hp.get('n_features') != 1:
        problems.append(f"n_features={hp.get('n_features')} (need 1)")
    if hp.get('n_time') != cps.WINDOW_SIZE:
        problems.append(f"n_time={hp.get('n_time')} (need {cps.WINDOW_SIZE})")
    if hp.get('embedding_dim') != 128:
        problems.append(f"embedding_dim={hp.get('embedding_dim')} (need 128)")
    if hp.get('classes') != len(REDLAMP_ANOMALY_TYPES):
        problems.append(f"classes={hp.get('classes')} (need {len(REDLAMP_ANOMALY_TYPES)})")
    if problems:
        raise ValueError(
            f"{sim_model_dir} is not architecturally compatible with RedLamp's ConvAEC: "
            + '; '.join(problems)
            + '. Retrain via Core-Clustering with matching n_time/embedding_dim and '
              '--class_list redlamp.'
        )


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', default='test')
    parser.add_argument('--sim_model_dir', required=True,
                         help='Path to a Core-Clustering training run output dir '
                              '(e.g. /path/to/Core-Clustering/outputs/cross_anomsim/<seed>), '
                              'containing bestmodel.pkl (+ optionally run_summary.json).')
    parser.add_argument('--datasets', nargs='*', default=['anomaly_archive', 'iops'])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--out_csv', default=None)
    parser.add_argument('--force', action='store_true',
                         help='Recompute every entity even if already present in out_csv.')
    args_cli = parser.parse_args()

    if not os.path.isfile(os.path.join(args_cli.sim_model_dir, 'bestmodel.pkl')):
        raise FileNotFoundError(f'No bestmodel.pkl found under {args_cli.sim_model_dir}')
    check_sim_model_compatible(args_cli.sim_model_dir)

    device = utils.init_dl_program(args_cli.gpu, seed=args_cli.seed)
    out_csv = args_cli.out_csv or f'./result/{args_cli.run_name}/simulation_cross_domain_accuracy.csv'
    summary_csv = out_csv.replace('.csv', '_summary.csv')

    rows = []
    already_done = set()
    if not args_cli.force and os.path.isfile(out_csv):
        prior = pd.read_csv(out_csv)
        rows = prior.to_dict('records')
        already_done = set(zip(prior['dataset'], prior['entity']))
        print(f'Resuming from {out_csv}: {len(already_done)} entities already scored, skipping those.')

    for dataset in args_cli.datasets:
        entities = frm.discover_dataset_entities(args_cli.run_name, dataset)
        print(f'{dataset}: found {len(entities)} entity directories, Cross-AnomSim model={args_cli.sim_model_dir}')
        for entity in entities:
            if (dataset, entity) in already_done:
                continue
            try:
                _, disk_cfg = ci.discover_entity(args_cli.run_name, dataset, entity, args_cli.seed)
            except FileNotFoundError:
                print(f'  [skip] {dataset}/{entity}: no self-model trained yet (needed for val split config)')
                continue

            dataparams = ci.build_dataparams(dataset, entity, CFG, disk_cfg)
            _, val_dl = datautils.load_dataloader_aug(dataparams, group='train')

            model_args = ci.build_model_args(CFG, disk_cfg['window_size'])
            params = utils.AttrDict(seed=args_cli.seed)
            params.override(main.model_parameters(model_args))

            accuracy, n_windows = sar.compute_self_accuracy(args_cli.sim_model_dir, params, device, val_dl)
            rows.append(dict(dataset=dataset, entity=entity, val_windows=n_windows, accuracy=accuracy))
            print(f'  {dataset}/{entity}: accuracy={accuracy:.4f} (n={n_windows})')

            df = pd.DataFrame(rows)
            df.to_csv(out_csv, index=False)
            summary = df.groupby('dataset')['accuracy'].mean().to_frame()
            summary['n_entities'] = df.groupby('dataset').size()
            summary.to_csv(summary_csv)

    if rows:
        df = pd.DataFrame(rows)
        summary = df.groupby('dataset')['accuracy'].mean().to_frame()
        summary['n_entities'] = df.groupby('dataset').size()
        print(summary)
    print(f'Done. {len(rows)} entities scored. Wrote {out_csv} and {summary_csv}')


if __name__ == '__main__':
    run()
