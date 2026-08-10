"""
Runs BOTH PowerDemand-generalization experiments end-to-end, each across
--seeds (default 0/1/2), and writes one combined summary:

  1. LOO:     AnomSim_v1 + UCR PowerDemand (7 entities held-in, 1 held-out
              per fold) -- build_ucr_anomsim_pool.py -> run_ucr_anomsim_loo_training.py
              -> score_ucr_anomsim_loo.py
  2. Bimodal: AnomSim_v1 + bimodal_cycle (synthetic, no real UCR data) --
              build_anomsim_bimodal_pool.py -> train_anomsim_bimodal_model.py
              -> score_anomsim_bimodal_on_powerdemand.py

Each stage is a plain function call into the existing standalone script
(same code, same CLI contract -- this file adds no new logic of its own
beyond wiring arguments through and combining the two _avg.csv outputs at
the end). Every one of those scripts is independently resumable
(online_cli.py skips a run whose bestmodel.pkl already exists), so
re-running this same command after an interruption -- or after this
process is killed -- just continues from wherever it stopped; nothing
needs to be torn down first.

Runs sequentially: build (both pools) -> train (LOO, then Bimodal) ->
score (LOO, then Bimodal) -> combine. Training is real GPU work sharing
one GPU, so LOO (8 entities x len(seeds) runs) and Bimodal (len(seeds)
runs) are never run concurrently with each other or themselves -- see
run_ucr_anomsim_loo_training.py's docstring for the timing rationale.
With the default 3 seeds: LOO ~= 8*3*20min ~= 8h, Bimodal ~= 3*20min ~= 1h,
so ~9h total on an otherwise-idle GPU.

--skip_build / --skip_train / --skip_score let you re-run just one stage
(e.g. re-score after manually inspecting a training log). --skip_loo /
--skip_bimodal run only one of the two experiments.

Building the UCR side (build_ucr_anomsim_pool.py) needs the real UCR
dataset (./dataset/AnomalyArchive), so this whole script -- unlike the
individual pool-builder for Bimodal alone -- is meant for the server, not
this development machine.
"""
import argparse
import importlib
import os
import sys

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, 'scripts'))

DEFAULT_ENTITIES = ['044', '045', '046', '047', '152', '153', '154', '155']
DEFAULT_SEEDS = [0, 1, 2]
METRIC_KEYS = ['VUS_ROC', 'VUS_PR', 'R_AUC_ROC', 'R_AUC_PR', 'RF']


def call(module_name, argv):
    mod = importlib.import_module(module_name)
    old_argv = sys.argv
    sys.argv = [module_name] + argv
    try:
        mod.run()
    finally:
        sys.argv = old_argv


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--entities', nargs='+', default=DEFAULT_ENTITIES)
    parser.add_argument('--seeds', nargs='+', type=int, default=DEFAULT_SEEDS)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--core_clustering_dir', default='../Core-Clustering')
    parser.add_argument('--anomsim_dir', default='../AnomSim/data/AnomSim_v1')
    parser.add_argument('--bimodal_dir', default='../AnomSim/data/AnomSim_v1_bimodal_part')
    parser.add_argument('--ucr_domain_name', default='ucr_PowerDemand')
    parser.add_argument('--ucr_pool_dir', default='./result/DS_2/achievability/anomsim_plus_ucr_powerdemand')
    parser.add_argument('--bimodal_pool_dir', default='./result/DS_2/achievability/anomsim_plus_bimodal')
    parser.add_argument('--loo_models_dir', default='./result/DS_2/achievability/loo_powerdemand_models')
    parser.add_argument('--bimodal_models_dir', default='./result/DS_2/achievability/anomsim_bimodal_models')
    parser.add_argument('--ucr_xlsx', default='./result/Experiment_2/Results/ucr_results.xlsx')
    parser.add_argument('--out_xlsx', default='./result/DS_2/achievability/powerdemand_experiments_summary.xlsx')
    parser.add_argument('--force_build', action='store_true')
    parser.add_argument('--force_train', action='store_true')
    parser.add_argument('--skip_build', action='store_true')
    parser.add_argument('--skip_train', action='store_true')
    parser.add_argument('--skip_score', action='store_true')
    parser.add_argument('--skip_loo', action='store_true')
    parser.add_argument('--skip_bimodal', action='store_true')
    args = parser.parse_args()

    seeds_str = [str(s) for s in args.seeds]

    # -- 1. build pools --
    if not args.skip_build:
        if not args.skip_loo:
            ucr_manifest = os.path.join(args.ucr_pool_dir, '_manifest.jsonl')
            if args.force_build or not os.path.isfile(ucr_manifest):
                print('=== [build] AnomSim_v1 + UCR PowerDemand pool ===')
                call('build_ucr_anomsim_pool', [
                    '--anomsim_dir', args.anomsim_dir, '--entities', *args.entities,
                    '--ucr_domain_name', args.ucr_domain_name, '--out_dir', args.ucr_pool_dir])
            else:
                print(f'=== [skip build] {ucr_manifest} already exists ===')

        if not args.skip_bimodal:
            bimodal_manifest = os.path.join(args.bimodal_pool_dir, '_manifest.jsonl')
            if args.force_build or not os.path.isfile(bimodal_manifest):
                print('=== [build] AnomSim_v1 + bimodal_cycle pool ===')
                call('build_anomsim_bimodal_pool', [
                    '--anomsim_dir', args.anomsim_dir, '--bimodal_dir', args.bimodal_dir,
                    '--out_dir', args.bimodal_pool_dir])
            else:
                print(f'=== [skip build] {bimodal_manifest} already exists ===')

    # -- 2. train --
    if not args.skip_train:
        if not args.skip_loo:
            print('=== [train] LOO: AnomSim_v1 + UCR PowerDemand (leave-one-out per entity x seed) ===')
            loo_argv = [
                '--pool_dir', args.ucr_pool_dir, '--ucr_domain_name', args.ucr_domain_name,
                '--entities', *args.entities, '--seeds', *seeds_str,
                '--core_clustering_dir', args.core_clustering_dir,
                '--output_dir', args.loo_models_dir, '--gpu', str(args.gpu), '--epochs', str(args.epochs)]
            if args.force_train:
                loo_argv.append('--force')
            call('run_ucr_anomsim_loo_training', loo_argv)

        if not args.skip_bimodal:
            print('=== [train] Bimodal: AnomSim_v1 + bimodal_cycle (single model per seed) ===')
            bimodal_argv = [
                '--pool_dir', args.bimodal_pool_dir, '--seeds', *seeds_str,
                '--core_clustering_dir', args.core_clustering_dir,
                '--output_dir', args.bimodal_models_dir, '--gpu', str(args.gpu), '--epochs', str(args.epochs)]
            if args.force_train:
                bimodal_argv.append('--force')
            call('train_anomsim_bimodal_model', bimodal_argv)

    # -- 3. score --
    loo_avg_csv = './result/DS_2/achievability/ucr_anomsim_loo_comparison_avg.csv'
    bimodal_avg_csv = './result/DS_2/achievability/anomsim_bimodal_powerdemand_comparison_avg.csv'
    if not args.skip_score:
        if not args.skip_loo:
            print('=== [score] LOO ===')
            call('score_ucr_anomsim_loo', [
                '--seeds', *seeds_str, '--gpu', str(args.gpu), '--entities', *args.entities,
                '--ucr_domain_name', args.ucr_domain_name, '--models_dir', args.loo_models_dir,
                '--ucr_xlsx', args.ucr_xlsx])
        if not args.skip_bimodal:
            print('=== [score] Bimodal ===')
            call('score_anomsim_bimodal_on_powerdemand', [
                '--seeds', *seeds_str, '--gpu', str(args.gpu), '--entities', *args.entities,
                '--models_dir', args.bimodal_models_dir, '--ucr_xlsx', args.ucr_xlsx])

    # -- 4. combine --
    print('=== [combine] writing final summary xlsx ===')
    sheets = {}
    loo_df = bimodal_df = None
    if not args.skip_loo and os.path.isfile(loo_avg_csv):
        loo_df = pd.read_csv(loo_avg_csv).set_index('entity')
        sheets['LOO_avg'] = loo_df.reset_index()
    if not args.skip_bimodal and os.path.isfile(bimodal_avg_csv):
        bimodal_df = pd.read_csv(bimodal_avg_csv).set_index('entity')
        sheets['Bimodal_avg'] = bimodal_df.reset_index()

    if loo_df is not None or bimodal_df is not None:
        entities = sorted(set((loo_df.index if loo_df is not None else [])) |
                           set((bimodal_df.index if bimodal_df is not None else [])))
        combined_rows = []
        for entity in entities:
            row = dict(entity=entity)
            src = loo_df if loo_df is not None and entity in loo_df.index else bimodal_df
            for m in METRIC_KEYS:
                row[f'Self_{m}'] = src.loc[entity, f'Self_{m}'] if src is not None and entity in src.index else None
                row[f'CrossAnomSim_{m}'] = src.loc[entity, f'CrossAnomSim_{m}'] if src is not None and entity in src.index else None
            if loo_df is not None and entity in loo_df.index:
                row['LOO_VUS_ROC_mean'] = loo_df.loc[entity, 'LOO_VUS_ROC_mean']
                row['LOO_VUS_ROC_std'] = loo_df.loc[entity, 'LOO_VUS_ROC_std']
                row['LOO_gap_closed'] = loo_df.loc[entity, 'gap_closed']
            if bimodal_df is not None and entity in bimodal_df.index:
                row['Bimodal_VUS_ROC_mean'] = bimodal_df.loc[entity, 'Bimodal_VUS_ROC_mean']
                row['Bimodal_VUS_ROC_std'] = bimodal_df.loc[entity, 'Bimodal_VUS_ROC_std']
                row['Bimodal_gap_closed'] = bimodal_df.loc[entity, 'gap_closed']
            combined_rows.append(row)
        sheets['Combined'] = pd.DataFrame(combined_rows)

    os.makedirs(os.path.dirname(args.out_xlsx), exist_ok=True)
    if sheets:
        with pd.ExcelWriter(args.out_xlsx) as writer:
            for name, df in sheets.items():
                df.to_excel(writer, sheet_name=name, index=False)
        print(f'Wrote {args.out_xlsx} (sheets: {list(sheets.keys())})')
        if 'Combined' in sheets:
            cols = ['entity', 'Self_VUS_ROC', 'CrossAnomSim_VUS_ROC']
            if 'LOO_VUS_ROC_mean' in sheets['Combined'].columns:
                cols += ['LOO_VUS_ROC_mean', 'LOO_VUS_ROC_std', 'LOO_gap_closed']
            if 'Bimodal_VUS_ROC_mean' in sheets['Combined'].columns:
                cols += ['Bimodal_VUS_ROC_mean', 'Bimodal_VUS_ROC_std', 'Bimodal_gap_closed']
            print(sheets['Combined'][cols].to_string(index=False))
    else:
        print('[warn] no _avg.csv found for either experiment -- nothing to combine')


if __name__ == '__main__':
    run()
