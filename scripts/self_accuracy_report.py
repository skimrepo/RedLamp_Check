"""
Self-only validation classification accuracy across every trained model in
every dataset (anomaly_archive, iops, smd, smap, msl).

Unlike domain_generalization.py's 9x6 cross-domain matrix (only valid between
anomaly_archive/iops, which both share n_features=1), the other datasets have
different n_features (smd=38, smap=25, msl=55) — feeding one model's weights
a differently-shaped input is architecturally impossible (Conv1d channel
mismatch), so this script only ever evaluates a model against its own
validation split. Does not modify main.py or cross_inference.py.

Scores every seed in --seeds (default 0-4, matching full_reproduction_metrics.py's
"average results of five runs" methodology) — pure inference over each seed's
already-trained checkpoint, no retraining. Writes a per-(entity,seed) raw CSV,
a seed-mean aggregate (the original self_accuracy_all_datasets.csv, entity-level,
now with an added n_seeds column), and a "best seed" cherry-picked ceiling (max
accuracy across seeds per entity) alongside dataset-level summaries of each —
same three-way split as full_reproduction_metrics.py/aggregate_best().
"""
import argparse
import os
import sys

import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import main
import datautils
import utils

import cross_inference as ci


DATASET_CFGS = {
    'anomaly_archive': dict(n_features=1, min_features=1, max_features=1),
    'iops': dict(n_features=1, min_features=1, max_features=1),
    'smd': dict(n_features=38, min_features=1, max_features=38),
    'smap': dict(n_features=25, min_features=1, max_features=25),
    'msl': dict(n_features=55, min_features=1, max_features=55),
}


def discover_dataset_entities(run_name, dataset):
    base = f'./result/{run_name}/{dataset}'
    if not os.path.isdir(base):
        return []
    return sorted(d for d in os.listdir(base)
                  if os.path.isdir(os.path.join(base, d)) and not d.startswith('_'))


def compute_self_accuracy(model_dir, params, device, val_dataloader):
    model = main.ConvAEC(params).to(device)
    model.load_state_dict(torch.load(f'{model_dir}/bestmodel.pkl'))
    model.eval()

    correct, total = 0, 0
    with torch.no_grad():
        for batch in val_dataloader:
            inputs = batch['Y'].transpose(2, 1).to(device)
            true = batch['label'].argmax(dim=1)
            _, x_out, _ = model(inputs)
            pred = x_out.argmax(dim=1).cpu()
            correct += (pred == true).sum().item()
            total += len(true)
    return correct / total, total


def aggregate_self_accuracy(raw_rows, agg_func):
    """Two-level aggregate matching full_reproduction_metrics.py's aggregate()/
    aggregate_best(): agg_func='mean' gives the paper's own "average results of
    five runs" methodology; agg_func='max' gives the cherry-picked best-seed
    ceiling. model_dir is dropped (it varies per seed, not meaningful once
    aggregated — organize_experiment1.py already drops it from the old
    single-seed CSV anyway)."""
    if not raw_rows:
        return None, None
    raw_df = pd.DataFrame(raw_rows)

    entity_df = raw_df.groupby(['dataset', 'entity'], as_index=False).agg(
        accuracy=('accuracy', agg_func),
        val_windows=('val_windows', 'mean'),
        n_seeds=('seed', 'nunique'),
    )

    summary = entity_df.groupby('dataset')['accuracy'].mean().to_frame()
    summary['n_entities'] = entity_df.groupby('dataset').size()
    summary['avg_seeds_per_entity'] = entity_df.groupby('dataset')['n_seeds'].mean()
    return entity_df, summary


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', default='test')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4],
                         help='Seeds to average over per entity, matching full_reproduction_metrics.py\'s '
                              '"average results of five runs" methodology. Pure inference over each seed\'s '
                              'already-trained checkpoint (no retraining); seeds not yet trained for a given '
                              'entity are skipped, not errored.')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--datasets', nargs='+', default=list(DATASET_CFGS.keys()),
                         help='Restrict to a subset, e.g. --datasets anomaly_archive iops for a faster '
                              'Experiment-1-only run instead of all 5 datasets.')
    parser.add_argument('--out_csv', default=None)
    parser.add_argument('--force', action='store_true',
                         help='Recompute every (entity, seed) pair even if already present in the raw CSV '
                              '(default: resume, skipping pairs already scored in a prior run).')
    args_cli = parser.parse_args()

    device = utils.init_dl_program(args_cli.gpu, seed=args_cli.seeds[0])
    out_csv = args_cli.out_csv or f'./result/{args_cli.run_name}/self_accuracy_all_datasets.csv'
    raw_csv = out_csv.replace('.csv', '_raw.csv')
    summary_csv = out_csv.replace('.csv', '_summary.csv')
    best_csv = out_csv.replace('.csv', '_best.csv')
    best_summary_csv = out_csv.replace('.csv', '_best_summary.csv')

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
        # redundantly re-scoring seed=0 results that already exist.
        old = pd.read_csv(out_csv)
        if 'seed' not in old.columns:
            old['seed'] = 0
            rows = old.to_dict('records')
            already_done = set(zip(old['dataset'], old['entity'], old['seed']))
            print(f'Migrated {len(already_done)} pre-existing seed=0 results from {out_csv} into the raw cache.')

    def save_all():
        pd.DataFrame(rows).to_csv(raw_csv, index=False)
        entity_df, summary = aggregate_self_accuracy(rows, 'mean')
        if entity_df is None:
            return
        entity_df.to_csv(out_csv, index=False)
        summary.to_csv(summary_csv)

        best_entity_df, best_summary = aggregate_self_accuracy(rows, 'max')
        best_entity_df.to_csv(best_csv, index=False)
        best_summary.to_csv(best_summary_csv)

    for dataset in args_cli.datasets:
        cfg = DATASET_CFGS[dataset]
        entities = discover_dataset_entities(args_cli.run_name, dataset)
        if not entities:
            print(f'[skip] no trained entities found for {dataset}')
            continue
        print(f'{dataset}: found {len(entities)} entity directories, seeds={args_cli.seeds}')
        for entity in entities:
            for seed in args_cli.seeds:
                if (dataset, entity, seed) in already_done:
                    continue
                try:
                    model_dir, disk_cfg = ci.discover_entity(args_cli.run_name, dataset, entity, seed)
                except FileNotFoundError:
                    print(f'  [skip] {dataset}/{entity} seed={seed}: no trained model found')
                    continue
                dataparams = ci.build_dataparams(dataset, entity, cfg, disk_cfg)
                _, val_dl = datautils.load_dataloader_aug(dataparams, group='train')

                model_args = ci.build_model_args(cfg, disk_cfg['window_size'])
                params = utils.AttrDict(seed=seed)
                params.override(main.model_parameters(model_args))

                accuracy, n_windows = compute_self_accuracy(model_dir, params, device, val_dl)
                rows.append(dict(dataset=dataset, entity=entity, seed=seed, model_dir=model_dir,
                                  val_windows=n_windows, accuracy=accuracy))
                print(f'  {dataset}/{entity} seed={seed}: accuracy={accuracy:.4f} (n={n_windows})')
                save_all()

    print(f'Done. {len(rows)} (entity, seed) pairs scored. Wrote {raw_csv}, {out_csv}, {summary_csv}, '
          f'{best_csv}, and {best_summary_csv}')


if __name__ == '__main__':
    run()
