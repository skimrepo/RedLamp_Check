"""
Scores each train_anomsim_bimodal_model.py "anomsim_bimodal_seedN" model
(trained on AnomSim_v1 + bimodal_cycle, no real UCR data at all) on all 8
PowerDemand UCR entities' REAL test sets, via
full_reproduction_metrics.score_entity(model_dir=...) -- the same
mechanism already used to score Cross-AnomSim/Cross-OpenSource against
UCR entities, unmodified. Repeats for every seed in --seeds and reports
both per-seed rows and the mean/std across seeds per entity.

Unlike score_ucr_anomsim_loo.py (one "without_X" model per entity), there
is exactly ONE model per seed here -- it never saw any real UCR data
during training, so it's valid to evaluate the SAME model against all 8
entities directly, no leave-one-out needed.

Combines with the same two already-computed baselines pulled from
result/Experiment_2/Results/ucr_results.xlsx's 'Per-Entity Comparison'
sheet (NOT recomputed):
  - Self_<metric>: entity's own dedicated Self model
  - CrossAnomSim_<metric>: the ORIGINAL Cross-AnomSim (AnomSim_v1 only)
  - Bimodal_<metric>: the new AnomSim_v1+bimodal_cycle model

Writes two CSVs:
  - --out_csv (default ..._per_seed.csv): one row per (entity, seed)
  - --out_avg_csv (default ..._avg.csv): one row per entity, Bimodal_<metric>
    replaced by Bimodal_<metric>_mean/Bimodal_<metric>_std across --seeds
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import main
import utils
import cross_inference as ci
import domain_generalization as dg
import full_reproduction_metrics as frm

DATASET = 'anomaly_archive'
DEFAULT_ENTITIES = ['044', '045', '046', '047', '152', '153', '154', '155']
DEFAULT_SEEDS = [0, 1, 2]
METRIC_KEYS = frm.METRIC_KEYS  # ['VUS_ROC', 'VUS_PR', 'R_AUC_ROC', 'R_AUC_PR', 'RF']


def load_baseline_scores(ucr_xlsx, entities):
    df = pd.read_excel(ucr_xlsx, sheet_name='Per-Entity Comparison')
    df['entity'] = df['entity'].astype(str).str.zfill(3)
    return df[df['entity'].isin(entities)].set_index('entity')


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', default='test')
    parser.add_argument('--seeds', nargs='+', type=int, default=DEFAULT_SEEDS)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--entities', nargs='+', default=DEFAULT_ENTITIES)
    parser.add_argument('--models_dir', default='./result/DS_2/achievability/anomsim_bimodal_models')
    parser.add_argument('--ucr_xlsx', default='./result/Experiment_2/Results/ucr_results.xlsx')
    parser.add_argument('--out_csv', default='./result/DS_2/achievability/anomsim_bimodal_powerdemand_comparison_per_seed.csv')
    parser.add_argument('--out_avg_csv', default='./result/DS_2/achievability/anomsim_bimodal_powerdemand_comparison_avg.csv')
    args = parser.parse_args()

    model_args = ci.build_model_args(dg.CFG, dg.WINDOW_SIZE)

    baseline = load_baseline_scores(args.ucr_xlsx, args.entities)
    missing = set(args.entities) - set(baseline.index)
    if missing:
        print(f'[warn] no baseline (Self/Cross-AnomSim) score found for: {sorted(missing)}')

    rows = []
    for seed in args.seeds:
        model_dir = os.path.join(args.models_dir, f'anomsim_bimodal_seed{seed}')
        if not os.path.isfile(os.path.join(model_dir, 'bestmodel.pkl')):
            print(f'[skip] seed{seed}: no bestmodel.pkl at {model_dir} -- run train_anomsim_bimodal_model.py first')
            continue
        device = utils.init_dl_program(args.gpu, seed=seed)

        for entity in args.entities:
            params = utils.AttrDict(seed=seed)
            params.override(main.model_parameters(model_args))
            result = frm.score_entity(args.run_name, DATASET, entity, seed, params, device, model_dir=model_dir)
            if result is None:
                print(f'[skip] {entity}/seed{seed}: scoring failed/unavailable (no real anomaly in test labels, or length mismatch)')
                continue

            row = dict(entity=entity, seed=seed)
            has_baseline = entity in baseline.index
            for m in METRIC_KEYS:
                row[f'Self_{m}'] = baseline.loc[entity, f'{m}_self'] if has_baseline else None
                row[f'CrossAnomSim_{m}'] = baseline.loc[entity, f'{m}_cross_anomsim'] if has_baseline else None
                row[f'Bimodal_{m}'] = result['metrics'][m]
            row['Self_peak_in_range'] = baseline.loc[entity, 'peak_in_range_self'] if has_baseline else None
            row['CrossAnomSim_peak_in_range'] = baseline.loc[entity, 'peak_in_range_cross_anomsim'] if has_baseline else None
            row['Bimodal_peak_in_range'] = result['peak_in_range']
            rows.append(row)

            gap_closed = None
            if has_baseline:
                self_v, cross_v, bimodal_v = row['Self_VUS_ROC'], row['CrossAnomSim_VUS_ROC'], row['Bimodal_VUS_ROC']
                if self_v != cross_v:
                    gap_closed = (bimodal_v - cross_v) / (self_v - cross_v)
            print(f'{entity}/seed{seed}: Self_VUS_ROC={row["Self_VUS_ROC"]}, CrossAnomSim_VUS_ROC={row["CrossAnomSim_VUS_ROC"]}, '
                  f'Bimodal_VUS_ROC={row["Bimodal_VUS_ROC"]:.4f}'
                  + (f', gap_closed={gap_closed:.1%}' if gap_closed is not None else ''))

    per_seed_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    per_seed_df.to_csv(args.out_csv, index=False)
    print(f'\nWrote {args.out_csv} ({len(per_seed_df)} rows)')

    avg_rows = []
    for entity in args.entities:
        sub = per_seed_df[per_seed_df['entity'] == entity] if not per_seed_df.empty else per_seed_df
        if sub.empty:
            continue
        row = dict(entity=entity, n_seeds=len(sub))
        for m in METRIC_KEYS + ['peak_in_range']:
            row[f'Self_{m}'] = sub[f'Self_{m}'].iloc[0]
            row[f'CrossAnomSim_{m}'] = sub[f'CrossAnomSim_{m}'].iloc[0]
            row[f'Bimodal_{m}_mean'] = sub[f'Bimodal_{m}'].mean()
            row[f'Bimodal_{m}_std'] = sub[f'Bimodal_{m}'].std() if len(sub) > 1 else 0.0
        self_v, cross_v = row['Self_VUS_ROC'], row['CrossAnomSim_VUS_ROC']
        row['gap_closed'] = (row['Bimodal_VUS_ROC_mean'] - cross_v) / (self_v - cross_v) if self_v != cross_v else None
        avg_rows.append(row)

    avg_df = pd.DataFrame(avg_rows)
    os.makedirs(os.path.dirname(args.out_avg_csv), exist_ok=True)
    avg_df.to_csv(args.out_avg_csv, index=False)
    print(f'Wrote {args.out_avg_csv} ({len(avg_df)} rows, averaged over seeds {args.seeds})')
    if not avg_df.empty:
        print(avg_df[['entity', 'n_seeds', 'Self_VUS_ROC', 'CrossAnomSim_VUS_ROC', 'Bimodal_VUS_ROC_mean', 'Bimodal_VUS_ROC_std', 'gap_closed']].to_string(index=False))


if __name__ == '__main__':
    run()
