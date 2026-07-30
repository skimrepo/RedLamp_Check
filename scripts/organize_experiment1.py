"""
Organize Experiment 1: physically MOVES (not copies) the seed=0 Self models
and the two Cross-OpenSource pooled models out of `result/{run_name}/...`
into a clean `Experiment 1/` folder, and builds `ucr_results.xlsx` /
`kpi_results.xlsx` from the already-produced result CSVs.

ACCEPTED RISK (explicit user decision): moving a Self model's seed=0 folder
away from `result/{run_name}/{dataset}/{entity}/d*_b*_w*_s*/0/` means
`cross_inference.discover_entity()` can no longer find it there. Any future
rerun of a script that calls discover_entity() for a moved entity —
domain_generalization.py, test_set_anomaly_metrics.py, cross_inference.py,
self_accuracy_report.py, or full_cross_domain_metrics.py for an entity not
yet in its output CSV — will fail with FileNotFoundError for that entity.
full_reproduction_metrics.py is unaffected: seed=0 rows are already cached in
full_reproduction_metrics_raw.csv, so it never re-touches those files.
Likewise, moving continuous_n944 means continuous_pool_scaling.py's own
n=944 scaling-curve checkpoint is no longer separately available (its
already-computed eval CSVs are unaffected).

Only seed=0 is moved — seeds 1-4 are still training and untouched here.

Safe to rerun: already-moved entities are skipped (source no longer exists
at the original path); Results/ is always freshly rewritten from whatever's
currently in the source CSVs, so this can be rerun any time to refresh the
Excel workbooks without re-touching already-moved models.
"""
import argparse
import os
import shutil
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cross_inference as ci
import full_reproduction_metrics as frm

SEED = 0

# dataset -> (source pool name under result/{run_name}/_pooled/, destination alias)
CROSS_MODEL_MAP = {
    'anomaly_archive': ('continuous_n697_excl_ucr', 'ucr_excl_ucr_pool'),
    'iops': ('continuous_n944', 'kpi_full_pool'),
}
CROSS_MODEL_SOURCE_NOTE = {
    'anomaly_archive': 'Trained on SMD+SMAP+MSL only (697 series). Zero UCR ever included.',
    'iops': 'Trained on the full continuous-feature pool (944 series: SMD+SMAP+MSL+other-UCR). '
            'AIOps/IOPS was never a candidate source in continuous_pool_scaling.py\'s pool builder '
            'to begin with, so this pool already qualifies as AIOps-excluded.',
}

DATASET_TOTAL = {'anomaly_archive': 250, 'iops': 29}
DATASET_OUT_NAME = {'anomaly_archive': 'ucr', 'iops': 'kpi'}


def move_self_models(run_name, exp_dir):
    moved, skipped, missing = 0, 0, 0
    for dataset in ['anomaly_archive', 'iops']:
        entities = frm.discover_dataset_entities(run_name, dataset)
        for entity in entities:
            dest = os.path.join(exp_dir, 'Models', 'Self', dataset, entity, str(SEED))
            if os.path.isdir(dest):
                skipped += 1
                continue
            try:
                model_dir, _ = ci.discover_entity(run_name, dataset, entity, SEED)
            except FileNotFoundError:
                missing += 1
                continue
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.move(model_dir, dest)
            moved += 1
            print(f'[moved] Self {dataset}/{entity} (seed {SEED}) -> {dest}')
    print(f'Self models: moved {moved}, already-moved (skipped) {skipped}, '
          f'no seed={SEED} checkpoint yet {missing}')


def move_cross_models(run_name, exp_dir):
    for dataset, (pool_name, alias) in CROSS_MODEL_MAP.items():
        dest = os.path.join(exp_dir, 'Models', 'Cross-OpenSource', alias, str(SEED))
        alias_dir = os.path.join(exp_dir, 'Models', 'Cross-OpenSource', alias)
        if os.path.isdir(dest):
            print(f'[skip] {alias}: already moved')
            continue
        src = f'./result/{run_name}/_pooled/{pool_name}/{SEED}'
        if not os.path.isfile(f'{src}/bestmodel.pkl'):
            print(f'[skip] {alias}: {src}/bestmodel.pkl not found yet (not trained, or already moved elsewhere)')
            continue
        os.makedirs(alias_dir, exist_ok=True)
        shutil.move(src, dest)
        with open(os.path.join(alias_dir, 'SOURCE.txt'), 'w') as f:
            f.write(f'Originally: result/{run_name}/_pooled/{pool_name}/{SEED}\n')
            f.write(f'Used to score: {dataset}\n')
            f.write(f'{CROSS_MODEL_SOURCE_NOTE[dataset]}\n')
        print(f'[moved] Cross-OpenSource {alias} <- {src} -> {dest}')


def _read_csv_if_exists(path):
    return pd.read_csv(path) if os.path.isfile(path) else pd.DataFrame()


def build_results(run_name, exp_dir):
    results_dir = os.path.join(exp_dir, 'Results')
    os.makedirs(results_dir, exist_ok=True)

    base = f'./result/{run_name}'
    self_df = _read_csv_if_exists(f'{base}/full_reproduction_metrics.csv')
    self_summary = _read_csv_if_exists(f'{base}/full_reproduction_metrics_summary.csv')
    cross_df = _read_csv_if_exists(f'{base}/full_cross_domain_metrics.csv')
    cross_summary = _read_csv_if_exists(f'{base}/full_cross_domain_metrics_summary.csv')
    vs_self = _read_csv_if_exists(f'{base}/full_cross_domain_metrics_vs_self.csv')

    for dataset, out_name in DATASET_OUT_NAME.items():
        s = self_df[self_df['dataset'] == dataset].drop(columns=['dataset']) if not self_df.empty else pd.DataFrame()
        c = cross_df[cross_df['dataset'] == dataset].drop(columns=['dataset']) if not cross_df.empty else pd.DataFrame()
        if dataset != 'anomaly_archive':
            s = s.drop(columns=['peak_in_range'], errors='ignore')
            c = c.drop(columns=['peak_in_range'], errors='ignore')

        if not s.empty or not c.empty:
            merged = s.merge(c, on='entity', how='outer', suffixes=('_self', '_cross'))
        else:
            merged = pd.DataFrame()

        ss = self_summary[self_summary['dataset'] == dataset] if not self_summary.empty else pd.DataFrame()
        cs = cross_summary[cross_summary['dataset'] == dataset] if not cross_summary.empty else pd.DataFrame()
        vs = vs_self[vs_self['dataset'] == dataset].drop(columns=['dataset']) if not vs_self.empty else pd.DataFrame()

        progress = pd.DataFrame([dict(
            n_entities_self=int(ss['n_entities'].iloc[0]) if not ss.empty else 0,
            n_entities_cross_opensource=int(cs['n_entities'].iloc[0]) if not cs.empty else 0,
            n_entities_total_possible=DATASET_TOTAL[dataset],
        )])

        out_path = os.path.join(results_dir, f'{out_name}_results.xlsx')
        with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
            s.to_excel(writer, sheet_name='Self', index=False)
            c.to_excel(writer, sheet_name='Cross-OpenSource', index=False)
            merged.to_excel(writer, sheet_name='Per-Entity Comparison', index=False)
            progress.to_excel(writer, sheet_name='Summary', index=False, startrow=0)
            ss.to_excel(writer, sheet_name='Summary', index=False, startrow=len(progress) + 2)
            cs.to_excel(writer, sheet_name='Summary', index=False, startrow=len(progress) + 2 + len(ss) + 2)
            vs.to_excel(writer, sheet_name='Summary', index=False,
                        startrow=len(progress) + 2 + len(ss) + 2 + len(cs) + 2)
        print(f'Wrote {out_path} '
              f'(Self rows={len(s)}, Cross-OpenSource rows={len(c)}, progress={progress.iloc[0].to_dict()})')


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', default='test')
    parser.add_argument('--exp_dir', default='./Experiment 1')
    args = parser.parse_args()

    move_self_models(args.run_name, args.exp_dir)
    move_cross_models(args.run_name, args.exp_dir)
    build_results(args.run_name, args.exp_dir)
    print(f'Done. Experiment 1 organized at {args.exp_dir}')


if __name__ == '__main__':
    run()
