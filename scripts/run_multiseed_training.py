"""
Launch main.py training jobs concurrently, sized to actually use the GPU's
headroom instead of running one job at a time. A single main.py training run
was observed at ~30% GPU utilization with ample VRAM to spare, so running
several concurrently (separate OS processes, each with its own CUDA context —
no shared state between them) should meaningfully cut wall-clock time.

A single (dataset, seed) job trains its entities sequentially inside one
process — only 2 datasets x N seeds jobs exist at that granularity, capping
concurrency well below what the GPU can actually take. So each dataset is
split into entity-range chunks (--chunk_size) via main.py's --entity_start/
--entity_end (an additive, opt-in slice — unused, existing invocations are
unaffected), each chunk running as its own subprocess. E.g. chunk_size=25
turns anomaly_archive's 250 entities into 10 chunks and iops's 29 into 2, so
--seeds 1 2 3 4 produces 4 x (10 + 2) = 48 jobs instead of just 8 — enough to
fill a much higher --max_parallel.

No tmux needed: launch this script itself with nohup so it survives SSH
disconnection on its own.

Every job is naturally idempotent (main.py skips any entity whose
test_all/input.npy already exists), so this launcher keeps no progress state
of its own — if interrupted, just rerun the same command and already-finished
entities are skipped near-instantly.

Usage (run from the repo root):
    mkdir -p logs
    nohup python -u scripts/run_multiseed_training.py \
        --run_name test --seeds 1 2 3 4 --chunk_size 25 --max_parallel 20 \
        > logs/run_multiseed_training.log 2>&1 &
    disown

Then check progress any time with:
    tail -f logs/multiseed_training/anomaly_archive_seed1_e000-025.log
or just watch the driver log:
    tail -f logs/run_multiseed_training.log
"""
import argparse
import os
import subprocess
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Must match main.py's hardcoded entity_list lengths (anomaly_archive: 250
# UCR subdatasets; iops: 29 KPI series). Only used to compute chunk
# boundaries here — main.py itself still owns the actual entity_list.
DATASET_SIZES = {'anomaly_archive': 250, 'iops': 29}


def build_jobs(datasets, seeds, chunk_size):
    jobs = []
    for seed in seeds:
        for dataset in datasets:
            n = DATASET_SIZES[dataset]
            for start in range(0, n, chunk_size):
                end = min(start + chunk_size, n)
                jobs.append(dict(dataset=dataset, seed=seed, entity_start=start, entity_end=end))
    return jobs


def launch(job, args, log_dir):
    tag = f"{job['dataset']}_seed{job['seed']}_e{job['entity_start']:03d}-{job['entity_end']:03d}"
    log_path = os.path.join(log_dir, f'{tag}.log')
    log_file = open(log_path, 'w')
    cmd = [sys.executable, '-u', 'main.py',
           '--dataset', job['dataset'], '--seed', str(job['seed']),
           '--epoch', str(args.epoch), '--run_name', args.run_name,
           '--gpu', str(args.gpu),
           '--entity_start', str(job['entity_start']), '--entity_end', str(job['entity_end'])]
    print(f'[launch] {tag} -> {log_path}', flush=True)
    proc = subprocess.Popen(cmd, cwd=REPO_ROOT, stdout=log_file, stderr=subprocess.STDOUT)
    return tag, proc, log_file


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', default='test')
    parser.add_argument('--datasets', nargs='+', default=['anomaly_archive', 'iops'])
    parser.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3, 4])
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--chunk_size', type=int, default=25,
                         help='Entities per job. Smaller = more, finer-grained concurrent jobs (more GPU '
                              'parallelism headroom to fill) but more per-process startup overhead.')
    parser.add_argument('--max_parallel', type=int, default=3,
                         help='How many main.py jobs to run at once on the same GPU. A single job runs at '
                              '~30%% GPU utilization on its own — raise this if nvidia-smi still shows '
                              'headroom, lower it if jobs start visibly slowing each other down.')
    parser.add_argument('--poll_seconds', type=int, default=15)
    parser.add_argument('--log_dir', default=None)
    args = parser.parse_args()

    log_dir = args.log_dir or os.path.join(REPO_ROOT, 'logs', 'multiseed_training')
    os.makedirs(log_dir, exist_ok=True)

    jobs = build_jobs(args.datasets, args.seeds, args.chunk_size)
    print(f'{len(jobs)} jobs queued (chunk_size={args.chunk_size}, seeds={args.seeds}, datasets={args.datasets})',
          flush=True)
    print(f'Running up to {args.max_parallel} at a time on GPU {args.gpu}. Per-job logs in {log_dir}/', flush=True)

    pending = list(jobs)
    running = {}  # tag -> (proc, log_file, start_time)
    finished = []

    while pending or running:
        while pending and len(running) < args.max_parallel:
            job = pending.pop(0)
            tag, proc, log_file = launch(job, args, log_dir)
            running[tag] = (proc, log_file, time.time())

        time.sleep(args.poll_seconds)

        for tag in list(running.keys()):
            proc, log_file, start_time = running[tag]
            ret = proc.poll()
            if ret is not None:
                log_file.close()
                elapsed_min = (time.time() - start_time) / 60
                status = 'OK' if ret == 0 else f'FAILED (exit code {ret})'
                print(f'[done] {tag}: {status} after {elapsed_min:.1f} min '
                      f'({len(running) - 1 + len(pending)} left)', flush=True)
                finished.append((tag, ret))
                del running[tag]

    n_failed = sum(1 for _, ret in finished if ret != 0)
    print(f'All {len(jobs)} jobs finished. {n_failed} failed.', flush=True)
    for tag, ret in finished:
        if ret != 0:
            print(f'  FAILED: {tag} — see {log_dir}/{tag}.log', flush=True)


if __name__ == '__main__':
    run()
