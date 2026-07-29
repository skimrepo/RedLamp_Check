"""
Simulation-to-real cross-domain check: score a model trained ENTIRELY on
AnomSim-generated synthetic data (via Core-Clustering, a sibling project --
never trained on any real RedLamp series) against RedLamp's real held-out
test sets (anomaly_archive/UCR, iops/KPI), using the exact same TSB-UAD
range-based metrics as full_reproduction_metrics.py / full_cross_domain_
metrics.py, so "self-trained on real data" vs "trained on a real
domain-excluded pool" vs "trained on 100% simulated data" gaps are all
directly comparable in the same units.

No retraining: the simulation-trained model already exists, produced by
Core-Clustering's own `core-clustering-train` CLI against an AnomSim
windowed dataset. This is pure inference (main.test()) over that already-
trained checkpoint, reusing full_reproduction_metrics.score_entity's
model_dir override exactly like full_cross_domain_metrics.py does for its
domain-excluded pooled models.

IMPORTANT compatibility requirement, checked automatically where possible
(see check_sim_model_compatible below): the simulation-trained model must
have been trained with n_features=1, window_size=100, embedding_dim=128
(Core-Clustering's ModelConfig defaults, so an unmodified `core-clustering-
train` run already matches) AND with class_list=redlamp (Core-Clustering's
--class_list redlamp flag, wired to core_clustering.redlamp_compat.
REDLAMP_ANOMALY_TYPES) so class index 0 is "normal" -- otherwise main.
anomaly_scoreing's "index 0 = normal" assumption silently breaks and
produces a meaningless (not obviously wrong) anomaly score.

Because full_reproduction_metrics.score_entity always discovers each
entity's own disk config (downsampling/batch_size/window_size/window_step)
from its self-trained model's result/ folder name, this script can only
cross-score entities that already have a self-trained checkpoint -- exactly
the same limitation full_cross_domain_metrics.py already has.

Resumable like the other full_* scripts: reruns skip entities already
present in out_csv unless --force is passed.

Does not modify main.py, cross_inference.py, domain_generalization.py,
continuous_pool_scaling.py, full_reproduction_metrics.py, or
full_cross_domain_metrics.py -- only imports from them.
"""
import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils
import main
import domain_generalization as dg
import continuous_pool_scaling as cps
import cross_inference as ci

import full_reproduction_metrics as frm
import full_cross_domain_metrics as fcdm

METRIC_KEYS = frm.METRIC_KEYS

# Must match Core-Clustering's core_clustering/redlamp_compat.py verbatim --
# this IS main.py's own default anomaly_types order (main.py:480), repeated
# here so a mismatch between the two repos is easy to spot in a diff.
REDLAMP_ANOMALY_TYPES = ['normal', 'spike', 'flip', 'speedup', 'noise', 'cutoff',
                         'average', 'scale', 'wander', 'contextual', 'upsidedown', 'mixture']


def check_sim_model_compatible(sim_model_dir):
    """Best-effort sanity check against Core-Clustering's run_summary.json,
    if present next to the checkpoint. Not a hard requirement (the checkpoint
    might have been copied without its summary), but catches the most common
    and most silent failure mode -- wrong class order/count -- with a clear
    error instead of a garbage anomaly score."""
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
                              '(e.g. /path/to/Core-Clustering/outputs/<run_id>), '
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
    out_csv = args_cli.out_csv or f'./result/{args_cli.run_name}/simulation_cross_domain_metrics.csv'
    summary_csv = out_csv.replace('.csv', '_summary.csv')
    vs_self_csv = out_csv.replace('.csv', '_vs_self.csv')

    model_args = ci.build_model_args(dg.CFG, cps.WINDOW_SIZE)
    params = utils.AttrDict(seed=args_cli.seed)
    params.override(main.model_parameters(model_args))

    rows = []
    already_done = set()
    if not args_cli.force and os.path.isfile(out_csv):
        prior = pd.read_csv(out_csv)
        rows = prior.to_dict('records')
        already_done = set(zip(prior['dataset'], prior['entity']))
        print(f'Resuming from {out_csv}: {len(already_done)} entities already scored, skipping those.')

    for dataset in args_cli.datasets:
        entities = frm.discover_dataset_entities(args_cli.run_name, dataset)
        print(f'{dataset}: found {len(entities)} entity directories, simulation model={args_cli.sim_model_dir}')
        for real_name in entities:
            if (dataset, real_name) in already_done:
                continue
            result = frm.score_entity(args_cli.run_name, dataset, real_name, args_cli.seed, params, device,
                                       model_dir=args_cli.sim_model_dir)
            if result is None:
                print(f'[skip] {dataset}/{real_name}: no self-model trained yet or scoring failed')
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
            fcdm.write_vs_self(summary, args_cli.run_name, vs_self_csv)

    if rows:
        df = pd.DataFrame(rows)
        summary = df.groupby('dataset')[METRIC_KEYS].mean()
        summary['n_entities'] = df.groupby('dataset').size()
        print(summary)
        fcdm.write_vs_self(summary, args_cli.run_name, vs_self_csv, verbose=True)
    print(f'Done. {len(rows)} entities scored. Wrote {out_csv}, {summary_csv}, and {vs_self_csv}')


if __name__ == '__main__':
    run()
