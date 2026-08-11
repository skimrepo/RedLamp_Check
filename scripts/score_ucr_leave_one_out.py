"""
Scores each train_ucr_leave_one_out.py without_{recording}_seed{seed} model
against BOTH held-out entities of that recording (the SAME model, since
both variants were excluded from its training pool together) via
full_reproduction_metrics.score_entity(model_dir=...).

Combines with two already-computed baselines pulled directly from
result/Experiment_2/Results/ucr_results.xlsx's 'Per-Entity Comparison'
sheet (NOT recomputed):
  - Self_<metric>: entity's own dedicated Self model
  - CrossAnomSim_<metric>: the ORIGINAL Cross-AnomSim (AnomSim_v1 only)

Writes two CSVs (same convention as score_ucr_anomsim_loo.py):
  - --out_csv (default ..._per_seed.csv): one row per (entity, seed)
  - --out_avg_csv (default ..._avg.csv): one row per entity, LOO_<metric>
    replaced by LOO_<metric>_mean/LOO_<metric>_std across --seeds, plus
    gap_closed = (LOO_VUS_ROC_mean - CrossAnomSim_VUS_ROC) /
                 (Self_VUS_ROC - CrossAnomSim_VUS_ROC)

Compare this against the OLD entity-level LOO's
ucr_leave_one_out.csv (from before the leave-one-recording-out fix) --
a large gap between the two is a direct measure of how much of the old
numbers was near-duplicate leakage (see train_ucr_leave_one_out.py's
docstring for the full explanation).
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
RECORDING_PAIRS = {
    'PowerDemand1': ['044', '152'],
    'PowerDemand2': ['045', '153'],
    'PowerDemand3': ['046', '154'],
    'PowerDemand4': ['047', '155'],
}
DEFAULT_SEEDS = [0, 1, 2]
METRIC_KEYS = frm.METRIC_KEYS


def load_baseline_scores(ucr_xlsx, entities):
    df = pd.read_excel(ucr_xlsx, sheet_name='Per-Entity Comparison')
    df['entity'] = df['entity'].astype(str).str.zfill(3)
    return df[df['entity'].isin(entities)].set_index('entity')


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--recordings', nargs='+', default=list(RECORDING_PAIRS.keys()))
    parser.add_argument('--seeds', type=int, nargs='+', default=DEFAULT_SEEDS)
    parser.add_argument('--run_name', default='test')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--ucr_xlsx', default='./result/Experiment_2/Results/ucr_results.xlsx')
    parser.add_argument('--out_csv', default='./result/DS_2/achievability/ucr_leave_one_recording_out_per_seed.csv')
    parser.add_argument('--out_avg_csv', default='./result/DS_2/achievability/ucr_leave_one_recording_out_avg.csv')
    args = parser.parse_args()

    entities = [e for r in args.recordings for e in RECORDING_PAIRS[r]]
    device = utils.init_dl_program(args.gpu, seed=args.seeds[0])
    model_args = ci.build_model_args(dg.CFG, dg.WINDOW_SIZE)

    baseline = load_baseline_scores(args.ucr_xlsx, entities)
    missing_baseline = set(entities) - set(baseline.index)
    if missing_baseline:
        print(f'[warn] no Self/CrossAnomSim score found in {args.ucr_xlsx} for: {sorted(missing_baseline)}')

    rows = []
    for recording in args.recordings:
        held_out_entities = RECORDING_PAIRS[recording]
        for seed in args.seeds:
            model_dir = f'./result/{args.run_name}/_loo_pair/without_{recording}_seed{seed}'
            if not os.path.isfile(os.path.join(model_dir, 'bestmodel.pkl')):
                print(f'[skip] {recording}/seed{seed}: no bestmodel.pkl at {model_dir} -- '
                      f'run train_ucr_leave_one_out.py first')
                continue

            params = utils.AttrDict(seed=seed)
            params.override(main.model_parameters(model_args))

            for held_out in held_out_entities:
                result = frm.score_entity(args.run_name, DATASET, held_out, seed, params, device, model_dir=model_dir)
                if result is None:
                    print(f'[skip] {held_out}/seed{seed}: LOO scoring failed/unavailable '
                          f'(no real anomaly in test labels, or length mismatch)')
                    continue

                row = dict(recording=recording, entity=held_out, seed=seed)
                has_baseline = held_out in baseline.index
                for m in METRIC_KEYS:
                    row[f'Self_{m}'] = baseline.loc[held_out, f'{m}_self'] if has_baseline else None
                    row[f'CrossAnomSim_{m}'] = baseline.loc[held_out, f'{m}_cross_anomsim'] if has_baseline else None
                    row[f'LOO_{m}'] = result['metrics'][m]
                row['Self_peak_in_range'] = baseline.loc[held_out, 'peak_in_range_self'] if has_baseline else None
                row['CrossAnomSim_peak_in_range'] = baseline.loc[held_out, 'peak_in_range_cross_anomsim'] if has_baseline else None
                row['LOO_peak_in_range'] = result['peak_in_range']
                rows.append(row)
                self_vus = row['Self_VUS_ROC']
                print(f'{recording}/{held_out}/seed{seed}: LOO VUS_ROC={result["metrics"]["VUS_ROC"]:.3f}'
                      + (f' (Self={self_vus:.3f})' if self_vus is not None else ' (no Self score found)'))

    per_seed_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    per_seed_df.to_csv(args.out_csv, index=False)
    print(f'\nWrote {args.out_csv} ({len(per_seed_df)} rows)')

    avg_rows = []
    for entity in entities:
        sub = per_seed_df[per_seed_df['entity'] == entity] if not per_seed_df.empty else per_seed_df
        if sub.empty:
            continue
        row = dict(recording=sub['recording'].iloc[0], entity=entity, n_seeds=len(sub))
        for m in METRIC_KEYS + ['peak_in_range']:
            row[f'Self_{m}'] = sub[f'Self_{m}'].iloc[0]
            row[f'CrossAnomSim_{m}'] = sub[f'CrossAnomSim_{m}'].iloc[0]
            row[f'LOO_{m}_mean'] = sub[f'LOO_{m}'].mean()
            row[f'LOO_{m}_std'] = sub[f'LOO_{m}'].std() if len(sub) > 1 else 0.0
        self_v, cross_v = row['Self_VUS_ROC'], row['CrossAnomSim_VUS_ROC']
        row['gap_closed'] = (row['LOO_VUS_ROC_mean'] - cross_v) / (self_v - cross_v) if self_v != cross_v else None
        avg_rows.append(row)

    avg_df = pd.DataFrame(avg_rows)
    os.makedirs(os.path.dirname(args.out_avg_csv), exist_ok=True)
    avg_df.to_csv(args.out_avg_csv, index=False)
    print(f'Wrote {args.out_avg_csv} ({len(avg_df)} rows, averaged over seeds {args.seeds})')
    if not avg_df.empty:
        print(avg_df[['recording', 'entity', 'n_seeds', 'Self_VUS_ROC', 'CrossAnomSim_VUS_ROC',
                       'LOO_VUS_ROC_mean', 'LOO_VUS_ROC_std', 'gap_closed']].to_string(index=False))


if __name__ == '__main__':
    run()
