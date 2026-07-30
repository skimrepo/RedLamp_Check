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


def _filter(df, dataset, drop_cols=()):
    if df.empty:
        return pd.DataFrame()
    out = df[df['dataset'] == dataset].drop(columns=['dataset'])
    return out.drop(columns=list(drop_cols), errors='ignore')


def build_results(run_name, exp_dir):
    results_dir = os.path.join(exp_dir, 'Results')
    os.makedirs(results_dir, exist_ok=True)

    base = f'./result/{run_name}'
    # Classification accuracy (validation split, injected pseudo-anomalies,
    # 12-class task) — this is Experiment 1's primary metric.
    acc_self_df = _read_csv_if_exists(f'{base}/self_accuracy_all_datasets.csv')
    acc_cross_df = _read_csv_if_exists(f'{base}/full_cross_domain_accuracy.csv')
    acc_cross_summary = _read_csv_if_exists(f'{base}/full_cross_domain_accuracy_summary.csv')

    # Real test-set anomaly-detection metrics (VUS-ROC/VUS-PR/RF/etc) — kept
    # as a secondary reference, comparable to the paper's Table 3.
    vus_self_df = _read_csv_if_exists(f'{base}/full_reproduction_metrics.csv')
    vus_self_summary = _read_csv_if_exists(f'{base}/full_reproduction_metrics_summary.csv')
    vus_cross_df = _read_csv_if_exists(f'{base}/full_cross_domain_metrics.csv')
    vus_cross_summary = _read_csv_if_exists(f'{base}/full_cross_domain_metrics_summary.csv')
    vus_vs_self = _read_csv_if_exists(f'{base}/full_cross_domain_metrics_vs_self.csv')

    for dataset, out_name in DATASET_OUT_NAME.items():
        acc_s = _filter(acc_self_df, dataset, drop_cols=['model_dir'])
        acc_c = _filter(acc_cross_df, dataset)
        acc_merged = (acc_s.merge(acc_c, on='entity', how='outer', suffixes=('_self', '_cross'))
                      if not acc_s.empty or not acc_c.empty else pd.DataFrame())

        acc_self_avg = float(acc_s['accuracy'].mean()) if not acc_s.empty else None
        acc_cross_row = acc_cross_summary[acc_cross_summary['dataset'] == dataset] if not acc_cross_summary.empty else pd.DataFrame()
        acc_cross_avg = float(acc_cross_row['accuracy'].iloc[0]) if not acc_cross_row.empty else None
        acc_summary = pd.DataFrame([dict(
            n_entities_self=len(acc_s),
            n_entities_cross_opensource=int(acc_cross_row['n_entities'].iloc[0]) if not acc_cross_row.empty else 0,
            n_entities_total_possible=DATASET_TOTAL[dataset],
            self_accuracy_avg=acc_self_avg,
            cross_opensource_accuracy_avg=acc_cross_avg,
            gap=(acc_self_avg - acc_cross_avg) if acc_self_avg is not None and acc_cross_avg is not None else None,
        )])

        drop = [] if dataset == 'anomaly_archive' else ['peak_in_range']
        vus_s = _filter(vus_self_df, dataset, drop_cols=drop)
        vus_c = _filter(vus_cross_df, dataset, drop_cols=drop)
        vus_merged = (vus_s.merge(vus_c, on='entity', how='outer', suffixes=('_self', '_cross'))
                      if not vus_s.empty or not vus_c.empty else pd.DataFrame())
        vus_ss = vus_self_summary[vus_self_summary['dataset'] == dataset] if not vus_self_summary.empty else pd.DataFrame()
        vus_cs = vus_cross_summary[vus_cross_summary['dataset'] == dataset] if not vus_cross_summary.empty else pd.DataFrame()
        vus_vs = _filter(vus_vs_self, dataset)

        out_path = os.path.join(results_dir, f'{out_name}_results.xlsx')
        with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
            # Primary: classification accuracy.
            acc_s.to_excel(writer, sheet_name='Self', index=False)
            acc_c.to_excel(writer, sheet_name='Cross-OpenSource', index=False)
            acc_merged.to_excel(writer, sheet_name='Per-Entity Comparison', index=False)
            acc_summary.to_excel(writer, sheet_name='Summary', index=False)
            # Secondary: real test-set VUS-based anomaly detection metrics.
            vus_s.to_excel(writer, sheet_name='Self (VUS Metrics)', index=False)
            vus_c.to_excel(writer, sheet_name='Cross-OpenSource (VUS Metrics)', index=False)
            vus_merged.to_excel(writer, sheet_name='Per-Entity Comparison (VUS)', index=False)
            vus_ss.to_excel(writer, sheet_name='Summary (VUS Metrics)', index=False, startrow=0)
            vus_cs.to_excel(writer, sheet_name='Summary (VUS Metrics)', index=False, startrow=len(vus_ss) + 2)
            vus_vs.to_excel(writer, sheet_name='Summary (VUS Metrics)', index=False,
                             startrow=len(vus_ss) + 2 + len(vus_cs) + 2)
        print(f'Wrote {out_path} (accuracy: self={len(acc_s)} cross={len(acc_c)}, '
              f'VUS: self={len(vus_s)} cross={len(vus_c)}, summary={acc_summary.iloc[0].to_dict()})')


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', default='test')
    parser.add_argument('--exp_dir', default='./result/Experiment 1')
    args = parser.parse_args()

    move_self_models(args.run_name, args.exp_dir)
    move_cross_models(args.run_name, args.exp_dir)
    build_results(args.run_name, args.exp_dir)
    print(f'Done. Experiment 1 organized at {args.exp_dir}')


if __name__ == '__main__':
    run()
