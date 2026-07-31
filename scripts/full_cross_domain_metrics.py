"""
Full-scale cross-domain generalization check: score the already-trained
domain-excluded pooled models against EVERY trained entity of the excluded
domain (not just the 3-entity holdout sample in test_set_metrics.csv), so the
"never seen this domain at all" generalization gap is measured at the same
scale as full_reproduction_metrics.py's self-baseline check.

  - anomaly_archive (UCR, up to 250 entities): scored against
    continuous_n697_excl_ucr (SMD+SMAP+MSL only — zero UCR ever included).
  - iops (AIOps, up to 29 entities): scored against continuous_n944 (the full
    continuous-feature pool — AIOps/IOPS was never a candidate source in
    continuous_pool_scaling.py's pool builder to begin with, so this pool
    already qualifies as AIOps-excluded).

No retraining: both pooled models already exist from prior experiments. This
is pure inference (main.test()) over already-trained checkpoints, so it's far
cheaper than the 279-entity self-baseline training run.

Resumable like full_reproduction_metrics.py: reruns skip entities already
present in out_csv unless --force is passed.
"""
import argparse
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

METRIC_KEYS = frm.METRIC_KEYS

CROSS_MODEL_ALIAS = {
    'anomaly_archive': 'continuous_n697_excl_ucr',
    'iops': 'continuous_n944',
}
# organize_experiment1.py moves these two pooled models out of _pooled/ into
# Experiment_1 under these alias names.
EXPERIMENT1_ALIAS = {
    'anomaly_archive': 'ucr_excl_ucr_pool',
    'iops': 'kpi_full_pool',
}


def cross_model_dir(run_name, dataset, seed):
    original = f'./result/{run_name}/_pooled/{CROSS_MODEL_ALIAS[dataset]}/{seed}'
    if os.path.isfile(f'{original}/bestmodel.pkl'):
        return original
    return f'./result/Experiment_1/Models/Cross-OpenSource/{EXPERIMENT1_ALIAS[dataset]}/{seed}'


def write_vs_self(cross_summary, run_name, vs_self_csv, verbose=False):
    self_summary_path = f'./result/{run_name}/full_reproduction_metrics_summary.csv'
    if not os.path.isfile(self_summary_path):
        print(f'[note] {self_summary_path} not found yet — skipping self-vs-cross-domain comparison for now')
        return
    self_summary = pd.read_csv(self_summary_path, index_col='dataset')

    comparison_rows = []
    for dataset in cross_summary.index:
        if dataset not in self_summary.index:
            continue
        for metric in METRIC_KEYS:
            cross_val = cross_summary.loc[dataset, metric]
            self_val = self_summary.loc[dataset, metric]
            comparison_rows.append(dict(dataset=dataset, metric=metric, self_model=self_val,
                                         cross_domain_model=cross_val, gap=self_val - cross_val))
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(vs_self_csv, index=False)
    if verbose:
        print(comparison.to_string(index=False))


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', default='test')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--out_csv', default=None)
    parser.add_argument('--force', action='store_true',
                         help='Recompute every entity even if already present in out_csv.')
    args_cli = parser.parse_args()

    device = utils.init_dl_program(args_cli.gpu, seed=args_cli.seed)
    out_csv = args_cli.out_csv or f'./result/{args_cli.run_name}/full_cross_domain_metrics.csv'
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

    for dataset in ['anomaly_archive', 'iops']:
        model_dir = cross_model_dir(args_cli.run_name, dataset, args_cli.seed)
        if not os.path.isfile(f'{model_dir}/bestmodel.pkl'):
            print(f'[skip] {dataset}: cross-domain model {model_dir} does not exist yet')
            continue

        entities = frm.discover_dataset_entities(args_cli.run_name, dataset)
        print(f'{dataset}: found {len(entities)} entity directories, cross-domain model={model_dir}')
        for real_name in entities:
            if (dataset, real_name) in already_done:
                continue
            result = frm.score_entity(args_cli.run_name, dataset, real_name, args_cli.seed, params, device,
                                       model_dir=model_dir)
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
            write_vs_self(summary, args_cli.run_name, vs_self_csv)

    if rows:
        df = pd.DataFrame(rows)
        summary = df.groupby('dataset')[METRIC_KEYS].mean()
        summary['n_entities'] = df.groupby('dataset').size()
        print(summary)
        write_vs_self(summary, args_cli.run_name, vs_self_csv, verbose=True)
    print(f'Done. {len(rows)} entities scored. Wrote {out_csv}, {summary_csv}, and {vs_self_csv}')


if __name__ == '__main__':
    run()
