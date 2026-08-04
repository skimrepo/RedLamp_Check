"""
Runs analyze_score_oscillation.py as N concurrent subprocesses, each over a
disjoint shard of the full entity list (index % num_shards == shard_index,
handled by analyze_score_oscillation.py's own --shard_index/--num_shards),
then merges their separate CSVs back into one. Single entry point for
speeding up the now ~247-entity UCR run on a CPU/memory-rich server without
touching analyze_score_oscillation.py's own per-entity logic.

Run this INSTEAD OF analyze_score_oscillation.py directly when you want the
parallel speedup; produces the exact same oscillation_metrics.csv shape at
the end. Still resumable: each shard's own partial CSV
(result/DS_2/oscillation/shards/oscillation_metrics_shard{i}.csv) is reused
on rerun unless --force, and re-merging is cheap, so it's safe to re-run
this whole script if some shards failed or the server was interrupted.

All shards default to the same --gpu -- fine to share since this workload
is mostly CPU-bound (TSB_UAD's get_metrics, rolling std/correlation), not
GPU-bound (ConvAEC inference on it is comparatively light). If shards
contend too much for GPU memory in practice, lower --num_shards.
"""
import argparse
import os
import subprocess
import sys

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT_PATH = os.path.join(REPO_ROOT, 'scripts', 'analyze_score_oscillation.py')


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', default='test')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--num_shards', type=int, default=8)
    parser.add_argument('--entity_metadata_csv', default=None,
                         help='Passed through to every shard -- restricts to a specific entity list '
                              '(e.g. DS_1\'s gap-selected 46) instead of discovering all ~247.')
    parser.add_argument('--cross_anomsim_model_dir', default=None)
    parser.add_argument('--out_csv', default='./result/DS_2/oscillation/oscillation_metrics.csv')
    parser.add_argument('--force', action='store_true',
                         help='Passed through to every shard -- recompute everything instead of resuming.')
    args = parser.parse_args()

    out_dir = os.path.dirname(args.out_csv) or '.'
    shard_dir = os.path.join(out_dir, 'shards')
    os.makedirs(shard_dir, exist_ok=True)

    procs = []
    shard_csvs = []
    for i in range(args.num_shards):
        shard_csv = os.path.join(shard_dir, f'oscillation_metrics_shard{i}.csv')
        shard_csvs.append(shard_csv)
        cmd = [sys.executable, SCRIPT_PATH,
               '--run_name', args.run_name, '--seed', str(args.seed), '--gpu', str(args.gpu),
               '--shard_index', str(i), '--num_shards', str(args.num_shards),
               '--out_csv', shard_csv]
        if args.entity_metadata_csv:
            cmd += ['--entity_metadata_csv', args.entity_metadata_csv]
        if args.cross_anomsim_model_dir:
            cmd += ['--cross_anomsim_model_dir', args.cross_anomsim_model_dir]
        if args.force:
            cmd += ['--force']

        log_path = os.path.join(shard_dir, f'shard{i}.log')
        log_file = open(log_path, 'w')
        print(f'[launch] shard {i}/{args.num_shards} -> {shard_csv} (log: {log_path})')
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
        print(f'WARNING: shard(s) {failed} failed -- check {shard_dir}/shard{{i}}.log before trusting the merge.')

    frames = [pd.read_csv(p) for p in shard_csvs if os.path.isfile(p)]
    if not frames:
        print('No shard output found -- nothing to merge.')
        return

    merged = pd.concat(frames, ignore_index=True)
    merged['entity'] = merged['entity'].astype(str).str.zfill(3)
    merged = merged.sort_values('entity').drop_duplicates(subset='entity', keep='last')
    merged.to_csv(args.out_csv, index=False)
    print(f'Merged {len(merged)} entities from {len(frames)} shard file(s) into {args.out_csv}')


if __name__ == '__main__':
    run()
