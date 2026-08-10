"""
Core-Clustering-NATIVE "Self" (single-entity) trainer -- the
Core-Clustering-side counterpart to main.py's own per-entity Self training,
so both train on the EXACT SAME entity (that UCR entity's train-portion
timeline, temporally split 90/10) and differ ONLY in which codebase does
the actual training.

Uses core_clustering.online_cli --single_entity <entity_dir>, pointed at
the pool build_ucr_anomsim_pool.py already built (which already has each
UCR entity's train-portion-only Y.npy under ucr_{domain}_{entity}/) --
that pool doesn't need to be UCR-only for this to work; --single_entity
just picks one entity_dir out of however many the pool has.

--val_fraction defaults to 0.1 to match main.py's hardcoded 90/10 train/val
split of the entity's own train portion (online_cli's own CLI default is
0.2, which would NOT match -- see core_clustering/single_entity.py's
load_single_entity_split).

--window_step replicates main.py's own per-entity dynamic rule (main.py
lines ~560-566: 1 if train_end<10000, 10 if <100000, else 100) instead of
domain_generalization.py/online_cli's fixed 10 -- read directly from the
pool's meta.json n_time (== that entity's train_end, since
build_ucr_anomsim_pool.py stores the train portion only), so this really
is an apples-to-apples match of main.py's own choice for that specific
entity, not the fixed pooled-training convention.

Meant for the server (needs build_ucr_anomsim_pool.py's output already
built, which itself needs the real UCR dataset). Resumable: online_cli.py
skips retraining if bestmodel.pkl already exists (pass --force to
override).
"""
import argparse
import json
import os
import subprocess
import sys

DEFAULT_ENTITIES = ['044', '045', '046', '047', '152', '153', '154', '155']
DEFAULT_SEEDS = [0, 1, 2]


def window_step_for(train_end):
    if train_end < 10000:
        return 1
    elif train_end < 100000:
        return 10
    else:
        return 100


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pool_dir', default='./result/DS_2/achievability/anomsim_plus_ucr_powerdemand')
    parser.add_argument('--ucr_domain_name', default='ucr_PowerDemand')
    parser.add_argument('--entities', nargs='+', default=DEFAULT_ENTITIES)
    parser.add_argument('--seeds', nargs='+', type=int, default=DEFAULT_SEEDS)
    parser.add_argument('--val_fraction', type=float, default=0.1)
    parser.add_argument('--core_clustering_dir', default='../Core-Clustering')
    parser.add_argument('--output_dir', default='./result/DS_2/achievability/self_via_core_clustering')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    pool_dir_abs = os.path.abspath(args.pool_dir)
    output_dir_abs = os.path.abspath(args.output_dir)
    core_clustering_dir_abs = os.path.abspath(args.core_clustering_dir)

    n_time_by_entity = {}
    manifest_path = os.path.join(args.pool_dir, '_manifest.jsonl')
    with open(manifest_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            meta = json.loads(line)
            if meta.get('type') == args.ucr_domain_name and 'ucr_entity' in meta:
                n_time_by_entity[meta['ucr_entity']] = meta['n_time']

    results = []
    for seed in args.seeds:
        for entity in args.entities:
            entity_dir_name = f'{args.ucr_domain_name}_{entity}'
            if entity not in n_time_by_entity:
                print(f'[skip] {entity}/seed{seed}: not found in {manifest_path} -- run build_ucr_anomsim_pool.py first')
                continue
            window_step = window_step_for(n_time_by_entity[entity])
            run_id = f'self_{entity_dir_name}_seed{seed}'

            cmd = [sys.executable, '-m', 'core_clustering.online_cli',
                   '--dataset_dir', pool_dir_abs,
                   '--single_entity', entity_dir_name,
                   '--val_fraction', str(args.val_fraction),
                   '--window_step', str(window_step),
                   '--output_dir', output_dir_abs,
                   '--run_id', run_id,
                   '--seed', str(seed),
                   '--gpu', str(args.gpu),
                   '--epochs', str(args.epochs)]
            if args.force:
                cmd.append('--force')

            print(f'[launch] {run_id} (train_end={n_time_by_entity[entity]} -> window_step={window_step})')
            print('  ' + ' '.join(cmd))
            ret = subprocess.run(cmd, cwd=core_clustering_dir_abs).returncode
            model_path = os.path.join(output_dir_abs, run_id, 'bestmodel.pkl')
            ok = ret == 0 and os.path.isfile(model_path)
            results.append((entity, seed, ok, model_path))
            print(f'  {"OK" if ok else f"FAILED (exit {ret})"} -> {model_path}')

    print('\nSummary:')
    for entity, seed, ok, model_path in results:
        print(f'  {entity}/seed{seed}: {"OK" if ok else "FAILED"} {model_path}')
    failed = [(e, s) for e, s, ok, _ in results if not ok]
    if failed:
        print(f'\nWARNING: {len(failed)} run(s) failed: {failed}')


if __name__ == '__main__':
    run()
