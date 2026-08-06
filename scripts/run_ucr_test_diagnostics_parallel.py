"""
Runs build_ucr_test_diagnostics.py as N concurrent subprocesses, each over a
disjoint shard of the full UCR entity list (index % num_shards ==
shard_index, handled by that script's own --shard_index/--num_shards).

That script already writes one PDF PER SHARD when --num_shards > 1
("..._shard{i}.pdf"), so shards never collide and there's no merge step --
just launch and wait. The result is --num_shards separate PDF files
covering disjoint entities; open whichever you need, or merge them
yourself later if you want one file (not done here -- no PDF-merging
dependency in this repo).
"""
import argparse
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT_PATH = os.path.join(REPO_ROOT, 'scripts', 'build_ucr_test_diagnostics.py')


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', default='test')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--num_shards', type=int, default=8)
    parser.add_argument('--cross_anomsim_model_dir', default=None)
    parser.add_argument('--out_pdf', default='./result/DS_3/test_diagnostics/UCR_Test_anomaly_inference_samples.pdf')
    parser.add_argument('--cache_dir', default='./result/DS_3/curves_cache/test')
    parser.add_argument('--force', action='store_true',
                         help='Passed through to every shard -- recompute everything instead of resuming.')
    args = parser.parse_args()

    out_dir = os.path.dirname(args.out_pdf) or '.'
    log_dir = os.path.join(out_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)

    procs = []
    for i in range(args.num_shards):
        cmd = [sys.executable, SCRIPT_PATH,
               '--run_name', args.run_name, '--seed', str(args.seed), '--gpu', str(args.gpu),
               '--shard_index', str(i), '--num_shards', str(args.num_shards),
               '--out_pdf', args.out_pdf, '--cache_dir', args.cache_dir]
        if args.cross_anomsim_model_dir:
            cmd += ['--cross_anomsim_model_dir', args.cross_anomsim_model_dir]
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
        base, ext = os.path.splitext(args.out_pdf)
        print(f'All {args.num_shards} shards completed: {base}_shard0{ext} .. {base}_shard{args.num_shards - 1}{ext}')


if __name__ == '__main__':
    run()
