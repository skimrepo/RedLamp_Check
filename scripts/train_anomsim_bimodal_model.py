"""
Trains one model per seed in --seeds (default 0/1/2) on the merged pool
build_anomsim_bimodal_pool.py already built (AnomSim_v1's 144 entities +
bimodal_cycle's 16 entities, no UCR data at all), using
core_clustering.online_cli UNMODIFIED. No leave-one-out here -- unlike the
UCR PowerDemand experiment, bimodal_cycle is purely synthetic, so the
resulting model can never have seen any real UCR PowerDemand data and is
valid to evaluate against ALL 8 PowerDemand entities directly (see
score_anomsim_bimodal_on_powerdemand.py).

All other online_cli.py args are left at their defaults, matching how the
real production Cross-AnomSim was trained (see
run_ucr_anomsim_loo_training.py's docstring for the same note).

Sequential across seeds by default (shared-GPU-memory reasons, see
run_ucr_anomsim_loo_training.py). 3 seeds x ~20 min ~= 1 hour total.

Resumable: online_cli.py itself skips retraining if output_dir/run_id/
bestmodel.pkl already exists (pass --force to override).
"""
import argparse
import os
import subprocess
import sys

DEFAULT_SEEDS = [0, 1, 2]


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pool_dir', default='./result/DS_2/achievability/anomsim_plus_bimodal')
    parser.add_argument('--seeds', nargs='+', type=int, default=DEFAULT_SEEDS)
    parser.add_argument('--core_clustering_dir', default='../Core-Clustering')
    parser.add_argument('--output_dir', default='./result/DS_2/achievability/anomsim_bimodal_models')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=None, help='Default: online_cli.py\'s own default (128)')
    parser.add_argument('--lr', type=float, default=None, help='Default: online_cli.py\'s own default (0.001) -- '
                                                                 'consider raising alongside --batch_size')
    parser.add_argument('--num_workers', type=int, default=None, help='Default: online_cli.py\'s own default (0)')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    pool_dir_abs = os.path.abspath(args.pool_dir)
    output_dir_abs = os.path.abspath(args.output_dir)
    core_clustering_dir_abs = os.path.abspath(args.core_clustering_dir)

    results = []
    for seed in args.seeds:
        run_id = f'anomsim_bimodal_seed{seed}'

        cmd = [sys.executable, '-m', 'core_clustering.online_cli',
               '--dataset_dir', pool_dir_abs,
               '--output_dir', output_dir_abs,
               '--run_id', run_id,
               '--seed', str(seed),
               '--gpu', str(args.gpu),
               '--epochs', str(args.epochs)]
        if args.batch_size is not None:
            cmd += ['--batch_size', str(args.batch_size)]
        if args.lr is not None:
            cmd += ['--lr', str(args.lr)]
        if args.num_workers is not None:
            cmd += ['--num_workers', str(args.num_workers)]
        if args.force:
            cmd.append('--force')

        print(f'[launch] {run_id}')
        print('  ' + ' '.join(cmd))
        ret = subprocess.run(cmd, cwd=core_clustering_dir_abs).returncode
        model_path = os.path.join(output_dir_abs, run_id, 'bestmodel.pkl')
        ok = ret == 0 and os.path.isfile(model_path)
        results.append((seed, ok, model_path))
        print(f'  {"OK" if ok else f"FAILED (exit {ret})"} -> {model_path}')

    print('\nSummary:')
    for seed, ok, model_path in results:
        print(f'  seed{seed}: {"OK" if ok else "FAILED"} {model_path}')
    failed = [s for s, ok, _ in results if not ok]
    if failed:
        print(f'\nWARNING: {len(failed)} run(s) failed: seeds {failed}')


if __name__ == '__main__':
    run()
