"""
Runs build_anomsim_train_val_diagnostics.py as N concurrent subprocesses,
each over a disjoint shard of the full AnomSim_v1 entity list (index %
num_shards == shard_index, handled by that script's own
--shard_index/--num_shards).

Unlike run_score_oscillation_parallel.py, there is NO merge step: each
entity gets its own PDF files (AnomSim_Train_{entity}.pdf,
AnomSim_Val_{entity}.pdf), so different shards never write the same file
and their outputs are already complete once all shards finish -- just
launch and wait.
"""
import argparse
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT_PATH = os.path.join(REPO_ROOT, 'scripts', 'build_anomsim_train_val_diagnostics.py')
CORE_CLUSTERING_DEFAULT = os.path.join(REPO_ROOT, '..', 'Core-Clustering')


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', default=os.path.join(REPO_ROOT, '..', 'AnomSim', 'data', 'AnomSim_v1'))
    parser.add_argument('--core_clustering_dir', default=CORE_CLUSTERING_DEFAULT)
    parser.add_argument('--cross_anomsim_model_dir', default='./result/Experiment_1/Models/Cross-AnomSim/0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_shards', type=int, default=8)
    parser.add_argument('--out_dir', default='./result/DS_3/train_val_diagnostics/anomsim')
    parser.add_argument('--cache_dir', default='./result/DS_3/curves_cache/anomsim')
    parser.add_argument('--force', action='store_true',
                         help='Passed through to every shard -- recompute everything instead of resuming.')
    args = parser.parse_args()

    log_dir = os.path.join(args.out_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)

    procs = []
    for i in range(args.num_shards):
        cmd = [sys.executable, SCRIPT_PATH,
               '--dataset_dir', args.dataset_dir, '--core_clustering_dir', args.core_clustering_dir,
               '--cross_anomsim_model_dir', args.cross_anomsim_model_dir, '--seed', str(args.seed),
               '--shard_index', str(i), '--num_shards', str(args.num_shards),
               '--out_dir', args.out_dir, '--cache_dir', args.cache_dir]
        if args.force:
            cmd += ['--force']

        log_path = os.path.join(log_dir, f'shard{i}.log')
        log_file = open(log_path, 'w')
        print(f'[launch] shard {i}/{args.num_shards} (log: {log_path})')
        proc = subprocess.Popen(cmd, cwd=REPO_ROOT, stdout=log_file, stderr=subprocess.STDOUT)
        procs.append((i, proc, log_file))

    failed = []
    for i, proc, log_file in procs:
        ret = proc.wait()
        log_file.close()
        print(f'  shard {i}: {"OK" if ret == 0 else f"FAILED (exit {ret})"}')
        if ret != 0:
            failed.append(i)
    if failed:
        print(f'WARNING: shard(s) {failed} failed -- check {log_dir}/shard{{i}}.log.')
    else:
        print(f'All {args.num_shards} shards completed. PDFs are in {args.out_dir}')


if __name__ == '__main__':
    run()
