"""
Scores each run_ucr_anomsim_loo_training.py "without_X" model (trained on
AnomSim_v1 + the 7 OTHER UCR entities, having never seen X) on X's own
REAL UCR test set, via full_reproduction_metrics.score_entity(model_dir=...)
-- the same mechanism already used to score Cross-AnomSim/Cross-OpenSource
against UCR entities, unmodified.

Combines with two already-computed baselines pulled directly from
result/Experiment_2/Results/ucr_results.xlsx's 'Per-Entity Comparison'
sheet (NOT recomputed -- both are fixed, deterministic, already-trained
models, so re-scoring them would just reproduce the same numbers):
  - Self_<metric>: entity's own dedicated Self model
  - CrossAnomSim_<metric>: the ORIGINAL Cross-AnomSim (AnomSim_v1 only, no
    UCR at all) -- the baseline this whole experiment is trying to beat
  - LOO_<metric>: the new without_X model (AnomSim_v1 + 7 UCR entities)

One row per entity, metrics as columns (VUS_ROC/VUS_PR/R_AUC_ROC/R_AUC_PR/
RF/peak_in_range x 3 models) -- lets you see directly whether mixing in
real UCR data (even leaving the target out) moved Cross-AnomSim's score on
that entity closer to Self's.
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
METRIC_KEYS = frm.METRIC_KEYS  # ['VUS_ROC', 'VUS_PR', 'R_AUC_ROC', 'R_AUC_PR', 'RF']


def load_baseline_scores(ucr_xlsx, entities):
    df = pd.read_excel(ucr_xlsx, sheet_name='Per-Entity Comparison')
    df['entity'] = df['entity'].astype(str).str.zfill(3)
    return df[df['entity'].isin(entities)].set_index('entity')


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', default='test')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--entities', nargs='+', default=DEFAULT_ENTITIES)
    parser.add_argument('--ucr_domain_name', default='ucr_PowerDemand')
    parser.add_argument('--models_dir', default='./result/DS_2/achievability/loo_powerdemand_models')
    parser.add_argument('--ucr_xlsx', default='./result/Experiment_2/Results/ucr_results.xlsx')
    parser.add_argument('--out_csv', default='./result/DS_2/achievability/ucr_anomsim_loo_comparison.csv')
    args = parser.parse_args()

    device = utils.init_dl_program(args.gpu, seed=args.seed)
    model_args = ci.build_model_args(dg.CFG, dg.WINDOW_SIZE)
    params = utils.AttrDict(seed=args.seed)
    params.override(main.model_parameters(model_args))

    baseline = load_baseline_scores(args.ucr_xlsx, args.entities)
    missing = set(args.entities) - set(baseline.index)
    if missing:
        print(f'[warn] no baseline (Self/Cross-AnomSim) score found for: {sorted(missing)}')

    rows = []
    for entity in args.entities:
        run_id = f'without_{args.ucr_domain_name}_{entity}'
        model_dir = os.path.join(args.models_dir, run_id)
        if not os.path.isfile(os.path.join(model_dir, 'bestmodel.pkl')):
            print(f'[skip] {entity}: no bestmodel.pkl at {model_dir} -- run run_ucr_anomsim_loo_training.py first')
            continue

        result = frm.score_entity(args.run_name, DATASET, entity, args.seed, params, device, model_dir=model_dir)
        if result is None:
            print(f'[skip] {entity}: LOO scoring failed/unavailable (no real anomaly in test labels, or length mismatch)')
            continue

        row = dict(entity=entity)
        has_baseline = entity in baseline.index
        for m in METRIC_KEYS:
            row[f'Self_{m}'] = baseline.loc[entity, f'{m}_self'] if has_baseline else None
            row[f'CrossAnomSim_{m}'] = baseline.loc[entity, f'{m}_cross_anomsim'] if has_baseline else None
            row[f'LOO_{m}'] = result['metrics'][m]
        row['Self_peak_in_range'] = baseline.loc[entity, 'peak_in_range_self'] if has_baseline else None
        row['CrossAnomSim_peak_in_range'] = baseline.loc[entity, 'peak_in_range_cross_anomsim'] if has_baseline else None
        row['LOO_peak_in_range'] = result['peak_in_range']
        rows.append(row)

        gap_closed = None
        if has_baseline:
            self_v, cross_v, loo_v = row['Self_VUS_ROC'], row['CrossAnomSim_VUS_ROC'], row['LOO_VUS_ROC']
            if self_v != cross_v:
                gap_closed = (loo_v - cross_v) / (self_v - cross_v)
        print(f'{entity}: Self_VUS_ROC={row["Self_VUS_ROC"]}, CrossAnomSim_VUS_ROC={row["CrossAnomSim_VUS_ROC"]}, '
              f'LOO_VUS_ROC={row["LOO_VUS_ROC"]:.4f}'
              + (f', gap_closed={gap_closed:.1%}' if gap_closed is not None else ''))

    out_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)
    print(out_df.to_string(index=False))
    print(f'Wrote {args.out_csv}')


if __name__ == '__main__':
    run()
