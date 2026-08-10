"""
RedLamp-NATIVE "Self" (single-entity) trainer -- a thin loop over main.py
itself (which only accepts one --entities value per invocation), so it can
be called with the same --entities/--seeds list shape as
train_ucr_self_via_core_clustering.py for a clean side-by-side comparison.

No new training logic here -- this just subprocess-invokes main.py exactly
as it's always been invoked (see run_multiseed_training.py for the same
pattern at larger scale), under a dedicated --run_name so its
bestmodel.pkl lands wherever cross_inference.discover_entity's own glob
(./result/{run_name}/{dataset}/{entity}/d*_b*_w*_s*/{seed}) will find it
-- main.py itself picks window_step dynamically per entity (train_end<10000
-> 1, <100000 -> 10, else 100), which is the whole point of comparing
against it (see compare_redlamp_vs_core_clustering.py's docstring).

Meant for the server (needs ./dataset/AnomalyArchive). Resumable: main.py
itself skips training if {model_dir}/test_all/input.npy already exists.
"""
import argparse
import os
import subprocess
import sys

DEFAULT_ENTITIES = ['044', '045', '046', '047', '152', '153', '154', '155']
DEFAULT_SEEDS = [0, 1, 2]


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--entities', nargs='+', default=DEFAULT_ENTITIES)
    parser.add_argument('--seeds', nargs='+', type=int, default=DEFAULT_SEEDS)
    parser.add_argument('--run_name', default='pipeline_compare')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--epoch', type=int, default=100)
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    results = []
    for seed in args.seeds:
        for entity in args.entities:
            cmd = [sys.executable, 'main.py',
                   '--dataset', 'anomaly_archive', '--entities', entity,
                   '--run_name', args.run_name, '--seed', str(seed),
                   '--gpu', str(args.gpu), '--epoch', str(args.epoch)]
            print(f'[launch] {entity}/seed{seed}')
            print('  ' + ' '.join(cmd))
            ret = subprocess.run(cmd, cwd=repo_root).returncode
            ok = ret == 0
            results.append((entity, seed, ok))
            print(f'  {"OK" if ok else f"FAILED (exit {ret})"}')

    print('\nSummary:')
    for entity, seed, ok in results:
        print(f'  {entity}/seed{seed}: {"OK" if ok else "FAILED"}')
    failed = [(e, s) for e, s, ok in results if not ok]
    if failed:
        print(f'\nWARNING: {len(failed)} run(s) failed: {failed}')


if __name__ == '__main__':
    run()
