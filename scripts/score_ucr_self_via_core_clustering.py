"""
Scores each train_ucr_self_via_core_clustering.py "self_X_seedN" model
(Core-Clustering's own core_clustering.online_cli --single_entity code,
trained on that SAME entity/window_step as the existing Self baseline)
against that entity's real UCR test set, via
full_reproduction_metrics.score_entity(model_dir=...) -- the same
mechanism already used elsewhere in this project.

Combines with two already-computed baselines pulled from
result/Experiment_2/Results/ucr_results.xlsx's 'Per-Entity Comparison'
sheet (NOT recomputed):
  - Self_<metric>: the EXISTING Self baseline, trained via RedLamp's own
    main.py (NOT Core-Clustering) -- see compare_redlamp_vs_core_clustering.py's
    docstring for the full breakdown of which codebase trained what.
  - CrossAnomSim_<metric>: the ORIGINAL Cross-AnomSim (AnomSim_v1 only),
    trained via Core-Clustering's online_cli, for reference.

CoreClusteringSelf_<metric> here isolates ONE variable cleanly against the
Self baseline: same entity, same train/val split (90/10), same
window_step (train_ucr_self_via_core_clustering.py replicates main.py's
own per-entity dynamic window_step rule) -- ONLY the training codebase
differs. If CoreClusteringSelf tracks Self closely, that's direct
evidence the two codebases train equivalently; if not, the gap is real and
not explained away by data/window_step differences.

Writes two CSVs (per-seed rows, and per-entity mean/std across seeds),
same convention as score_ucr_anomsim_loo.py / score_anomsim_bimodal_on_powerdemand.py.
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
METRIC_KEYS = frm.METRIC_KEYS


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
    parser.add_argument('--ucr_domain_name', default='ucr_PowerDemand')
    parser.add_argument('--models_dir', default='./result/DS_2/achievability/self_via_core_clustering')
    parser.add_argument('--ucr_xlsx', default='./result/Experiment_2/Results/ucr_results.xlsx')
    parser.add_argument('--out_csv', default='./result/DS_2/achievability/ucr_self_via_core_clustering_comparison_per_seed.csv')
    parser.add_argument('--out_avg_csv', default='./result/DS_2/achievability/ucr_self_via_core_clustering_comparison_avg.csv')
    args = parser.parse_args()

    device = utils.init_dl_program(args.gpu, seed=args.seeds[0])
    model_args = ci.build_model_args(dg.CFG, dg.WINDOW_SIZE)

    baseline = load_baseline_scores(args.ucr_xlsx, args.entities)
    missing = set(args.entities) - set(baseline.index)
    if missing:
        print(f'[warn] no baseline (Self/Cross-AnomSim) score found for: {sorted(missing)}')

    rows = []
    for entity in args.entities:
        for seed in args.seeds:
            run_id = f'self_{args.ucr_domain_name}_{entity}_seed{seed}'
            model_dir = os.path.join(args.models_dir, run_id)
            if not os.path.isfile(os.path.join(model_dir, 'bestmodel.pkl')):
                print(f'[skip] {entity}/seed{seed}: no bestmodel.pkl at {model_dir} -- run train_ucr_self_via_core_clustering.py first')
                continue

            params = utils.AttrDict(seed=seed)
            params.override(main.model_parameters(model_args))
            result = frm.score_entity(args.run_name, DATASET, entity, seed, params, device, model_dir=model_dir)
            if result is None:
                print(f'[skip] {entity}/seed{seed}: scoring failed/unavailable')
                continue

            row = dict(entity=entity, seed=seed)
            has_baseline = entity in baseline.index
            for m in METRIC_KEYS:
                row[f'Self_{m}'] = baseline.loc[entity, f'{m}_self'] if has_baseline else None
                row[f'CrossAnomSim_{m}'] = baseline.loc[entity, f'{m}_cross_anomsim'] if has_baseline else None
                row[f'CoreClusteringSelf_{m}'] = result['metrics'][m]
            row['Self_peak_in_range'] = baseline.loc[entity, 'peak_in_range_self'] if has_baseline else None
            row['CoreClusteringSelf_peak_in_range'] = result['peak_in_range']
            rows.append(row)

            print(f'{entity}/seed{seed}: Self_VUS_ROC={row["Self_VUS_ROC"]}, '
                  f'CoreClusteringSelf_VUS_ROC={row["CoreClusteringSelf_VUS_ROC"]:.4f}')

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
            row[f'CoreClusteringSelf_{m}_mean'] = sub[f'CoreClusteringSelf_{m}'].mean()
            row[f'CoreClusteringSelf_{m}_std'] = sub[f'CoreClusteringSelf_{m}'].std() if len(sub) > 1 else 0.0
        row['abs_diff_vs_Self_VUS_ROC'] = abs(row['CoreClusteringSelf_VUS_ROC_mean'] - row['Self_VUS_ROC'])
        avg_rows.append(row)

    avg_df = pd.DataFrame(avg_rows)
    os.makedirs(os.path.dirname(args.out_avg_csv), exist_ok=True)
    avg_df.to_csv(args.out_avg_csv, index=False)
    print(f'Wrote {args.out_avg_csv} ({len(avg_df)} rows, averaged over seeds {args.seeds})')
    if not avg_df.empty:
        print(avg_df[['entity', 'n_seeds', 'Self_VUS_ROC', 'CoreClusteringSelf_VUS_ROC_mean',
                       'CoreClusteringSelf_VUS_ROC_std', 'abs_diff_vs_Self_VUS_ROC']].to_string(index=False))


if __name__ == '__main__':
    run()
