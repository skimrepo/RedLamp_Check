"""
Trains + scores 4 model arms per (entity, seed) to isolate whether
RedLamp's own training code and Core-Clustering's training code produce
statistically similar models now that they've been aligned (double-softmax
bug, AMP vs fp32, NaN-batch handling, and the 4 wide-context anomaly
injection types were all fixed to match -- see git history around
"Align RedLamp's own training pipeline with Core-Clustering's"). Each pair
below shares the EXACT SAME data/pool/split -- codebase is the only thing
that varies within a pair:

  RedLamp_Self         main.py itself (train_ucr_self_via_redlamp.py)
  CoreClustering_Self   core_clustering.online_cli --single_entity
                        (train_ucr_self_via_core_clustering.py)
  RedLamp_LOO           main.REDLAMP + Loader_aug, pooled AnomSim_v1(144)
                        + 7 other UCR entities (train_anomsim_ucr_loo_via_redlamp.py)
  CoreClustering_LOO    core_clustering.online_cli, same pool, --exclude
                        (run_ucr_anomsim_loo_training.py, already built)

One thing NOT controlled for, on purpose -- worth knowing before reading
the numbers: main.py picks window_step dynamically per entity (train_end<
10000 -> 1, <100000 -> 10, else 100), matched here in
train_ucr_self_via_core_clustering.py's Self arm. But BOTH LOO arms use a
FIXED window_step=10 (domain_generalization.py's own pooled-training
convention, also online_cli's default) regardless of entity, since a
per-entity dynamic window_step has no meaning once entities are pooled
into one shared dataloader. So Self vs LOO differences may partly reflect
this window_step gap (e.g. PowerDemand entity 044: train_end=9000 -> Self
uses window_step=1, but LOO (either codebase) uses window_step=10 -- 10x
fewer effective training windows for this entity's pooled training) rather
than a RedLamp-vs-Core-Clustering discrepancy. RedLamp_Self vs
CoreClustering_Self isolates codebase cleanly (same window_step, same
entity, same split); RedLamp_LOO vs CoreClustering_LOO isolates codebase
cleanly too (same pool, same window_step) -- it's only the Self-vs-LOO
comparison that conflates codebase with window_step.

Scores all 4 arms via full_reproduction_metrics.score_entity(model_dir=...)
against each entity's real UCR test set -- same mechanism, same code, for
all 4, so scoring itself introduces no asymmetry.

Sequential, real GPU training (4 arms x len(entities) x len(seeds) runs,
though CoreClustering_LOO is typically already trained by
run_ucr_anomsim_loo_training.py from the earlier PowerDemand experiment --
this script only retrains what's missing, everything is resumable).
"""
import argparse
import importlib
import os
import sys

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, 'scripts'))

import main
import utils
import cross_inference as ci
import domain_generalization as dg
import full_reproduction_metrics as frm

DATASET = 'anomaly_archive'
DEFAULT_ENTITIES = ['044', '045', '046', '047', '152', '153', '154', '155']
DEFAULT_SEEDS = [0, 1, 2]
METRIC_KEYS = frm.METRIC_KEYS


def call(module_name, argv):
    mod = importlib.import_module(module_name)
    old_argv = sys.argv
    sys.argv = [module_name] + argv
    try:
        mod.run()
    finally:
        sys.argv = old_argv


def score_one(run_name, entity, seed, model_dir, device):
    if not os.path.isfile(os.path.join(model_dir, 'bestmodel.pkl')):
        return None
    model_args = ci.build_model_args(dg.CFG, dg.WINDOW_SIZE)
    params = utils.AttrDict(seed=seed)
    params.override(main.model_parameters(model_args))
    result = frm.score_entity(run_name, DATASET, entity, seed, params, device, model_dir=model_dir)
    return result['metrics'] if result is not None else None


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--entities', nargs='+', default=DEFAULT_ENTITIES)
    parser.add_argument('--seeds', nargs='+', type=int, default=DEFAULT_SEEDS)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--run_name', default='pipeline_compare')
    parser.add_argument('--anomsim_dir', default='../AnomSim/data/AnomSim_v1')
    parser.add_argument('--ucr_pool_dir', default='./result/DS_2/achievability/anomsim_plus_ucr_powerdemand')
    parser.add_argument('--ucr_domain_name', default='ucr_PowerDemand')
    parser.add_argument('--core_clustering_dir', default='../Core-Clustering')
    parser.add_argument('--self_cc_dir', default='./result/DS_2/achievability/self_via_core_clustering')
    parser.add_argument('--loo_redlamp_run_name', default='pipeline_compare')
    parser.add_argument('--loo_cc_dir', default='./result/DS_2/achievability/loo_powerdemand_models')
    parser.add_argument('--out_csv', default='./result/DS_2/achievability/redlamp_vs_core_clustering_comparison.csv')
    parser.add_argument('--skip_train', action='store_true')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    seeds_str = [str(s) for s in args.seeds]

    if not args.skip_train:
        print('=== [train] RedLamp Self (main.py) ===')
        redlamp_self_argv = ['--entities', *args.entities, '--seeds', *seeds_str,
                              '--run_name', args.run_name, '--gpu', str(args.gpu), '--epoch', str(args.epochs)]
        call('train_ucr_self_via_redlamp', redlamp_self_argv)

        print('=== [train] Core-Clustering Self (online_cli --single_entity) ===')
        cc_self_argv = ['--pool_dir', args.ucr_pool_dir, '--ucr_domain_name', args.ucr_domain_name,
                         '--entities', *args.entities, '--seeds', *seeds_str,
                         '--core_clustering_dir', args.core_clustering_dir,
                         '--output_dir', args.self_cc_dir, '--gpu', str(args.gpu), '--epochs', str(args.epochs)]
        if args.force:
            cc_self_argv.append('--force')
        call('train_ucr_self_via_core_clustering', cc_self_argv)

        print('=== [train] RedLamp LOO (main.REDLAMP, AnomSim_v1 + UCR pool) ===')
        redlamp_loo_argv = ['--entities', *args.entities, '--seeds', *seeds_str,
                             '--anomsim_dir', args.anomsim_dir, '--run_name', args.loo_redlamp_run_name,
                             '--gpu', str(args.gpu)]
        if args.force:
            redlamp_loo_argv.append('--force')
        call('train_anomsim_ucr_loo_via_redlamp', redlamp_loo_argv)

        print('=== [train] Core-Clustering LOO (online_cli, same pool, --exclude) ===')
        cc_loo_argv = ['--pool_dir', args.ucr_pool_dir, '--ucr_domain_name', args.ucr_domain_name,
                        '--entities', *args.entities, '--seeds', *seeds_str,
                        '--core_clustering_dir', args.core_clustering_dir,
                        '--output_dir', args.loo_cc_dir, '--gpu', str(args.gpu), '--epochs', str(args.epochs)]
        if args.force:
            cc_loo_argv.append('--force')
        call('run_ucr_anomsim_loo_training', cc_loo_argv)

    print('=== [score] all 4 arms ===')
    device = utils.init_dl_program(args.gpu, seed=args.seeds[0])
    rows = []
    for entity in args.entities:
        for seed in args.seeds:
            row = dict(entity=entity, seed=seed)

            redlamp_self_dir, _ = None, None
            try:
                redlamp_self_dir, _ = ci.discover_entity(args.run_name, DATASET, entity, seed)
            except FileNotFoundError:
                pass
            m = score_one(args.run_name, entity, seed, redlamp_self_dir, device) if redlamp_self_dir else None
            for k in METRIC_KEYS:
                row[f'RedLamp_Self_{k}'] = m[k] if m else None

            cc_self_dir = os.path.join(args.self_cc_dir, f'self_{args.ucr_domain_name}_{entity}_seed{seed}')
            m = score_one(args.run_name, entity, seed, cc_self_dir, device)
            for k in METRIC_KEYS:
                row[f'CoreClustering_Self_{k}'] = m[k] if m else None

            redlamp_loo_dir = f'./result/{args.loo_redlamp_run_name}/_loo_redlamp/without_{entity}_seed{seed}'
            m = score_one(args.run_name, entity, seed, redlamp_loo_dir, device)
            for k in METRIC_KEYS:
                row[f'RedLamp_LOO_{k}'] = m[k] if m else None

            cc_loo_dir = os.path.join(args.loo_cc_dir, f'without_{args.ucr_domain_name}_{entity}_seed{seed}')
            m = score_one(args.run_name, entity, seed, cc_loo_dir, device)
            for k in METRIC_KEYS:
                row[f'CoreClustering_LOO_{k}'] = m[k] if m else None

            rows.append(row)
            print(f'{entity}/seed{seed}: RedLamp_Self_VUS_ROC={row["RedLamp_Self_VUS_ROC"]}, '
                  f'CoreClustering_Self_VUS_ROC={row["CoreClustering_Self_VUS_ROC"]}, '
                  f'RedLamp_LOO_VUS_ROC={row["RedLamp_LOO_VUS_ROC"]}, '
                  f'CoreClustering_LOO_VUS_ROC={row["CoreClustering_LOO_VUS_ROC"]}')

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f'\nWrote {args.out_csv} ({len(df)} rows)')

    print('\nPer-entity mean absolute difference (codebase-isolated pairs only):')
    for entity in args.entities:
        sub = df[df['entity'] == entity]
        if sub.empty:
            continue
        self_diff = (sub['RedLamp_Self_VUS_ROC'] - sub['CoreClustering_Self_VUS_ROC']).abs().mean()
        loo_diff = (sub['RedLamp_LOO_VUS_ROC'] - sub['CoreClustering_LOO_VUS_ROC']).abs().mean()
        print(f'  {entity}: |RedLamp_Self - CoreClustering_Self| mean={self_diff:.4f}, '
              f'|RedLamp_LOO - CoreClustering_LOO| mean={loo_diff:.4f}')


if __name__ == '__main__':
    run()
