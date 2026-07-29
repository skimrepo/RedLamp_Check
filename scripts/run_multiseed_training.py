"""
Launch every (dataset, seed) main.py training job concurrently, sized to
actually use the GPU's headroom instead of running one job at a time. A
single main.py training run was observed at ~30% GPU utilization with ample
VRAM to spare, so running several concurrently (separate OS processes, each
with its own CUDA context — no shared state between them) should meaningfully
cut wall-clock time without OOM risk.

No tmux needed: launch this script itself with nohup so it survives SSH
disconnection on its own.

Each job is `python -u main.py --dataset D --seed S --epoch E --run_name R
--gpu G`. main.py's per-entity skip logic (skips any entity whose
test_all/input.npy already exists) makes every job naturally idempotent, so
this launcher keeps no progress state of its own — if interrupted, just rerun
the same command and already-finished entities are skipped near-instantly.

Usage (run from the repo root):
    mkdir -p logs
    nohup python -u scripts/run_multiseed_training.py \
        --run_name test --seeds 1 2 3 4 --max_parallel 3 \
        > logs/run_multiseed_training.log 2>&1 &
    disown

Then check progress any time with:
    tail -f logs/multiseed_training/anomaly_archive_seed1.log
or just watch the driver log:
    tail -f logs/run_multiseed_training.log
"""
import argparse
import os
import subprocess
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def launch(dataset, seed, args, log_dir):
    log_path = os.path.join(log_dir, f'{dataset}_seed{seed}.log')
    log_file = open(log_path, 'w')
    cmd = [sys.executable, '-u', 'main.py',
           '--dataset', dataset, '--seed', str(seed),
           '--epoch', str(args.epoch), '--run_name', args.run_name,
           '--gpu', str(args.gpu)]
    print(f'[launch] {dataset} seed={seed} -> {log_path}', flush=True)
    proc = subprocess.Popen(cmd, cwd=REPO_ROOT, stdout=log_file, stderr=subprocess.STDOUT)
    return proc, log_file


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', default='test')
    parser.add_argument('--datasets', nargs='+', default=['anomaly_archive', 'iops'])
    parser.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3, 4])
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--max_parallel', type=int, default=3,
                         help='How many main.py jobs to run at once on the same GPU. A single job runs at '
                              '~30%% GPU utilization on its own — raise this if nvidia-smi still shows '
                              'headroom, lower it if jobs start visibly slowing each other down.')
    parser.add_argument('--poll_seconds', type=int, default=15)
    parser.add_argument('--log_dir', default=None)
    args = parser.parse_args()

    log_dir = args.log_dir or os.path.join(REPO_ROOT, 'logs', 'multiseed_training')
    os.makedirs(log_dir, exist_ok=True)

    jobs = [(dataset, seed) for seed in args.seeds for dataset in args.datasets]
    print(f'{len(jobs)} jobs queued: {jobs}', flush=True)
    print(f'Running up to {args.max_parallel} at a time on GPU {args.gpu}. Per-job logs in {log_dir}/', flush=True)

    pending = list(jobs)
    running = {}  # (dataset, seed) -> (proc, log_file, start_time)
    finished = []

    while pending or running:
        while pending and len(running) < args.max_parallel:
            dataset, seed = pending.pop(0)
            proc, log_file = launch(dataset, seed, args, log_dir)
            running[(dataset, seed)] = (proc, log_file, time.time())

        time.sleep(args.poll_seconds)

        for key in list(running.keys()):
            proc, log_file, start_time = running[key]
            ret = proc.poll()
            if ret is not None:
                log_file.close()
                elapsed_min = (time.time() - start_time) / 60
                dataset, seed = key
                status = 'OK' if ret == 0 else f'FAILED (exit code {ret})'
                print(f'[done] {dataset} seed={seed}: {status} after {elapsed_min:.1f} min', flush=True)
                finished.append((key, ret))
                del running[key]

    n_failed = sum(1 for _, ret in finished if ret != 0)
    print(f'All {len(jobs)} jobs finished. {n_failed} failed.', flush=True)
    for (dataset, seed), ret in finished:
        if ret != 0:
            print(f'  FAILED: {dataset} seed={seed} — see {log_dir}/{dataset}_seed{seed}.log', flush=True)


if __name__ == '__main__':
    run()
