"""
Full-scale Cross-OpenSource classification accuracy: evaluate the two
domain-excluded pooled models — continuous_n697_excl_ucr (scored against UCR)
and continuous_n944 (scored against KPI/AIOps; AIOps was never a candidate
pool source to begin with, so this pool already qualifies as AIOps-excluded)
— against EVERY UCR/KPI entity's own validation split (with injected
pseudo-anomalies, the same 12-class task used everywhere else in this repo),
producing per-entity classification accuracy, averaged per dataset.

This is the Cross-OpenSource counterpart to self_accuracy_report.py's Self
accuracy (which already covers every entity's own model on its own val
split) and to domain_generalization.py's small-scale (3+3 holdout entity)
cross-domain accuracy matrix — extended here to the full 250 UCR / 29 KPI
scale, matching full_reproduction_metrics.py/full_cross_domain_metrics.py's
scope.

No retraining: both pooled models already exist. Resumable: reruns skip
entities already present in out_csv unless --force is passed.
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import main
import datautils
import utils
import cross_inference as ci

import full_reproduction_metrics as frm
import full_cross_domain_metrics as fcdm
import self_accuracy_report as sar

CFG = dict(n_features=1, min_features=1, max_features=1)  # shared by both UCR and KPI


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
    out_csv = args_cli.out_csv or f'./result/{args_cli.run_name}/full_cross_domain_accuracy.csv'
    summary_csv = out_csv.replace('.csv', '_summary.csv')

    rows = []
    already_done = set()
    if not args_cli.force and os.path.isfile(out_csv):
        prior = pd.read_csv(out_csv)
        rows = prior.to_dict('records')
        already_done = set(zip(prior['dataset'], prior['entity']))
        print(f'Resuming from {out_csv}: {len(already_done)} entities already scored, skipping those.')

    for dataset in ['anomaly_archive', 'iops']:
        model_dir = fcdm.cross_model_dir(args_cli.run_name, dataset, args_cli.seed)
        if not os.path.isfile(f'{model_dir}/bestmodel.pkl'):
            print(f'[skip] {dataset}: cross-domain model {model_dir} does not exist yet')
            continue

        entities = frm.discover_dataset_entities(args_cli.run_name, dataset)
        print(f'{dataset}: found {len(entities)} entity directories, cross-domain model={model_dir}')
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

            accuracy, n_windows = sar.compute_self_accuracy(model_dir, params, device, val_dl)
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
