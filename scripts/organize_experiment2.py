"""
Organize Experiment_2 (real-anomaly VUS-based metrics): builds
`ucr_results.xlsx` / `kpi_results.xlsx` (Self / Cross-OpenSource /
Cross-AnomSim / Per-Entity Comparison / Summary) from the paper's 5
range-based metrics (VUS_ROC, VUS_PR, R_AUC_ROC, R_AUC_PR, RF, + UCR-only
peak_in_range), computed against UCR/KPI's REAL test-set ground truth
anomalies -- as opposed to Experiment_1's classification accuracy, which is
computed against injected pseudo-anomalies on the validation split.

AnomSim_v1 has no real ground-truth anomaly labels at all (injection only
ever happens at training time), so it can never appear here -- Experiment_2
only ever covers anomaly_archive (UCR) and iops (KPI).

Does not move or copy any model checkpoints: these are the exact same
physical models already organized under Experiment_1/Models/ by
scripts/organize_experiment1.py, just scored a second way. Only writes a
short README.txt pointing back there for traceability.

Reads (all already produced by existing scripts, no new CSVs here):
  full_reproduction_metrics.csv / _summary.csv       (Self)
  full_cross_domain_metrics.csv / _summary.csv / _vs_self.csv  (Cross-OpenSource)
  simulation_cross_domain_metrics.csv / _summary.csv (Cross-AnomSim -- produced
    separately by scripts/simulation_cross_domain_metrics.py --sim_model_dir
    <cross_anomsim checkpoint>, since that model lives in the sibling
    Core-Clustering repo)

Safe to rerun any time: always freshly rewrites the Excel workbooks from
whatever's currently in the source CSVs.
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from xlsx_merge_utils import merge_3way

DATASET_OUT_NAME = {'anomaly_archive': 'ucr', 'iops': 'kpi'}


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
    self_df = _read_csv_if_exists(f'{base}/full_reproduction_metrics.csv')
    self_summary = _read_csv_if_exists(f'{base}/full_reproduction_metrics_summary.csv')
    cross_df = _read_csv_if_exists(f'{base}/full_cross_domain_metrics.csv')
    cross_summary = _read_csv_if_exists(f'{base}/full_cross_domain_metrics_summary.csv')
    vs_self = _read_csv_if_exists(f'{base}/full_cross_domain_metrics_vs_self.csv')
    anomsim_df = _read_csv_if_exists(f'{base}/simulation_cross_domain_metrics.csv')
    anomsim_summary = _read_csv_if_exists(f'{base}/simulation_cross_domain_metrics_summary.csv')

    for dataset, out_name in DATASET_OUT_NAME.items():
        drop = [] if dataset == 'anomaly_archive' else ['peak_in_range']
        s = _filter(self_df, dataset, drop_cols=drop)
        c = _filter(cross_df, dataset, drop_cols=drop)
        a = _filter(anomsim_df, dataset, drop_cols=drop)
        merged = merge_3way([(s, 'self'), (c, 'cross_opensource'), (a, 'cross_anomsim')])

        ss = self_summary[self_summary['dataset'] == dataset] if not self_summary.empty else pd.DataFrame()
        cs = cross_summary[cross_summary['dataset'] == dataset] if not cross_summary.empty else pd.DataFrame()
        asum = anomsim_summary[anomsim_summary['dataset'] == dataset] if not anomsim_summary.empty else pd.DataFrame()
        vs = _filter(vs_self, dataset)

        out_path = os.path.join(results_dir, f'{out_name}_results.xlsx')
        with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
            s.to_excel(writer, sheet_name='Self', index=False)
            c.to_excel(writer, sheet_name='Cross-OpenSource', index=False)
            a.to_excel(writer, sheet_name='Cross-AnomSim', index=False)
            merged.to_excel(writer, sheet_name='Per-Entity Comparison', index=False)
            ss.to_excel(writer, sheet_name='Summary', index=False, startrow=0)
            cs.to_excel(writer, sheet_name='Summary', index=False, startrow=len(ss) + 2)
            asum.to_excel(writer, sheet_name='Summary', index=False, startrow=len(ss) + 2 + len(cs) + 2)
            vs.to_excel(writer, sheet_name='Summary', index=False,
                        startrow=len(ss) + 2 + len(cs) + 2 + len(asum) + 2)
        print(f'Wrote {out_path} (self={len(s)}, cross-opensource={len(c)}, cross-anomsim={len(a)})')

    readme_path = os.path.join(results_dir, 'README.txt')
    if not os.path.isfile(readme_path):
        with open(readme_path, 'w') as f:
            f.write('Experiment_2: real-anomaly VUS-based metrics (VUS_ROC, VUS_PR, R_AUC_ROC, '
                    'R_AUC_PR, RF, + UCR-only peak_in_range), scored against UCR/KPI real test-set '
                    'ground truth. Model checkpoints themselves are not duplicated here -- the exact '
                    'same physical models live under ../Experiment_1/Models/ (Self, Cross-OpenSource, '
                    'Cross-AnomSim), just scored a second way (real test-set anomalies here, vs. '
                    'validation-set injected pseudo-anomalies in Experiment_1).\n\n'
                    'AnomSim_v1 has no real ground-truth anomaly labels, so it never appears in '
                    'Experiment_2 -- only anomaly_archive (UCR) and iops (KPI).\n')


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', default='test')
    parser.add_argument('--exp_dir', default='./result/Experiment_2')
    args = parser.parse_args()

    build_results(args.run_name, args.exp_dir)
    print(f'Done. Experiment_2 organized at {args.exp_dir}')


if __name__ == '__main__':
    run()
