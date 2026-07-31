"""
Build anomsim_results.xlsx: brings Self (Core-Clustering's
scripts/train_self_all.py -> outputs/self_accuracy_all_entities.csv),
Cross-OpenSource (Core-Clustering's
scripts/eval_cross_opensource_on_anomsim.py -> outputs/
cross_opensource_accuracy.csv), and Cross-AnomSim (the all-AnomSim pooled
model's OWN classification_accuracy.csv from its training run) into
RedLamp_Check's Experiment_1/Results/, matching ucr_results.xlsx/
kpi_results.xlsx's sheet-layout convention (organize_experiment1.py, not
modified here).

Cross-AnomSim's AnomSim_v1 self-check is domain-level (9 rows -- the
held-out validation GROUP within each waveform domain, from the model's own
pool-mode training) rather than per-entity like Self/Cross-OpenSource (144
rows) -- it comes for free from that training run and needs no dedicated
eval script, but the sample granularity genuinely differs, so it lives in
its own sheet instead of being force-merged into Per-Entity Comparison.

No VUS sheets: AnomSim_v1 has no ground-truth anomaly labels/test split, so
real detection metrics can never exist for it -- this is deliberate, not a
gap to fill in later.

Takes explicit CLI paths into the sibling Core-Clustering repo (like
simulation_cross_domain_metrics.py's --sim_model_dir) rather than assuming a
relative cross-repo layout.
"""
import argparse
import os

import pandas as pd

N_ANOMSIM_V1_ENTITIES = 144


def _read_csv_if_exists(path):
    return pd.read_csv(path) if path and os.path.isfile(path) else pd.DataFrame()


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--self_accuracy_csv', required=True,
                         help='Core-Clustering outputs/self_accuracy_all_entities.csv')
    parser.add_argument('--cross_opensource_accuracy_csv', required=True,
                         help='Core-Clustering outputs/cross_opensource_accuracy.csv')
    parser.add_argument('--cross_anomsim_accuracy_csv', default=None,
                         help="The Cross-AnomSim pooled model's own classification_accuracy.csv "
                              '(e.g. Core-Clustering/outputs/cross_anomsim/<seed>/'
                              'classification_accuracy.csv) -- domain-level val-group accuracy '
                              'from its own training run. Optional; omit if not trained yet.')
    parser.add_argument('--out_xlsx', default='./result/Experiment_1/Results/anomsim_results.xlsx')
    args = parser.parse_args()

    self_df = _read_csv_if_exists(args.self_accuracy_csv)
    if not self_df.empty:
        self_df = self_df.drop(columns=['seed', 'model_dir'], errors='ignore')
    cross_df = _read_csv_if_exists(args.cross_opensource_accuracy_csv)
    anomsim_df = _read_csv_if_exists(args.cross_anomsim_accuracy_csv)

    merged = (self_df.merge(cross_df, on='entity', how='outer', suffixes=('_self', '_cross_opensource'))
              if not self_df.empty or not cross_df.empty else pd.DataFrame())

    self_avg = float(self_df['accuracy'].mean()) if not self_df.empty else None
    cross_avg = float(cross_df['accuracy'].mean()) if not cross_df.empty else None
    summary = pd.DataFrame([dict(
        n_entities_self=len(self_df),
        n_entities_cross_opensource=len(cross_df),
        n_entities_total_possible=N_ANOMSIM_V1_ENTITIES,
        self_accuracy_avg=self_avg,
        cross_opensource_accuracy_avg=cross_avg,
        gap=(self_avg - cross_avg) if self_avg is not None and cross_avg is not None else None,
    )])

    out_dir = os.path.dirname(args.out_xlsx)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with pd.ExcelWriter(args.out_xlsx, engine='openpyxl') as writer:
        self_df.to_excel(writer, sheet_name='Self', index=False)
        cross_df.to_excel(writer, sheet_name='Cross-OpenSource', index=False)
        merged.to_excel(writer, sheet_name='Per-Entity Comparison', index=False)
        summary.to_excel(writer, sheet_name='Summary', index=False)
        if not anomsim_df.empty:
            anomsim_df.to_excel(writer, sheet_name='Cross-AnomSim (self, by domain)', index=False)
        else:
            pd.DataFrame([{'note': 'not provided yet -- pass --cross_anomsim_accuracy_csv once '
                                    'the all-AnomSim pooled model has been trained'}]).to_excel(
                writer, sheet_name='Cross-AnomSim (self, by domain)', index=False)

    print(f'Wrote {args.out_xlsx} (Self rows={len(self_df)}, Cross-OpenSource rows={len(cross_df)}, '
          f'Cross-AnomSim domain rows={len(anomsim_df)})')


if __name__ == '__main__':
    run()
