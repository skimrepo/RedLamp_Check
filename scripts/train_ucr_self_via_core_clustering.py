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

Runs up to --max_parallel jobs concurrently (default 4), same pattern as
run_multiseed_training.py -- each --single_entity run trains on just ONE
entity's own small timeline (not a 150+-entity pool), so it's far lighter
than the LOO pool training this repo's other scripts default to
sequential; a single main.py-scale job was observed at ~30% GPU
utilization with ample VRAM to spare, so several of these tiny jobs
should fit comfortably. Raise --max_parallel if nvidia-smi still shows
headroom, lower it if jobs visibly start slowing each other down. Each
job's own stdout/stderr goes to its own file under --log_dir (default
logs/self_via_core_clustering/) since interleaved output from several
concurrent jobs on one terminal is unreadable.

Meant for the server (needs build_ucr_anomsim_pool.py's output already
built, which itself needs the real UCR dataset). Resumable: online_cli.py
skips retraining if bestmodel.pkl already exists (pass --force to
override); if interrupted, just rerun the same command.
"""
import argparse
import json
import os
import subprocess
import sys
import time

DEFAULT_ENTITIES = ['044', '045', '046', '047', '152', '153', '154', '155']
DEFAULT_SEEDS = [0, 1, 2]


def window_step_for(train_end):
    if train_end < 10000:
        return 1
    elif train_end < 100000:
        return 10
    else:
        return 100


def build_jobs(entities, seeds, n_time_by_entity, ucr_domain_name):
    jobs = []
    for seed in seeds:
        for entity in entities:
            if entity not in n_time_by_entity:
                print(f'[skip] {entity}/seed{seed}: not in pool manifest -- run build_ucr_anomsim_pool.py first')
                continue
            n_time = n_time_by_entity[entity]
            jobs.append(dict(entity=entity, seed=seed, n_time=n_time,
                              entity_dir_name=f'{ucr_domain_name}_{entity}',
                              window_step=window_step_for(n_time)))
    return jobs


def launch(job, args, pool_dir_abs, output_dir_abs, core_clustering_dir_abs, log_dir):
    run_id = f"self_{job['entity_dir_name']}_seed{job['seed']}"
    cmd = [sys.executable, '-u', '-m', 'core_clustering.online_cli',
           '--dataset_dir', pool_dir_abs,
           '--single_entity', job['entity_dir_name'],
           '--val_fraction', str(args.val_fraction),
           '--window_step', str(job['window_step']),
           '--output_dir', output_dir_abs,
           '--run_id', run_id,
           '--seed', str(job['seed']),
           '--gpu', str(args.gpu),
           '--epochs', str(args.epochs)]
    if args.force:
        cmd.append('--force')

    log_path = os.path.join(log_dir, f'{run_id}.log')
    log_file = open(log_path, 'w')
    print(f'[launch] {run_id} (train_end={job["n_time"]} -> window_step={job["window_step"]}) -> {log_path}', flush=True)
    proc = subprocess.Popen(cmd, cwd=core_clustering_dir_abs, stdout=log_file, stderr=subprocess.STDOUT)
    model_path = os.path.join(output_dir_abs, run_id, 'bestmodel.pkl')
    return run_id, proc, log_file, model_path


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pool_dir', default='./result/DS_2/achievability/anomsim_plus_ucr_powerdemand')
    parser.add_argument('--ucr_domain_name', default='ucr_PowerDemand')
    parser.add_argument('--entities', nargs='+', default=DEFAULT_ENTITIES)
    parser.add_argument('--seeds', nargs='+', type=int, default=DEFAULT_SEEDS)
    parser.add_argument('--val_fraction', type=float, default=0.1)
    parser.add_argument('--core_clustering_dir', default='../Core-Clustering')
    parser.add_argument('--output_dir', default='./result/DS_2/achievability/self_via_core_clustering')
    parser.add_argument('--log_dir', default=None, help='Default: logs/self_via_core_clustering/')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--max_parallel', type=int, default=4,
                         help='How many online_cli.py --single_entity jobs to run at once on the same GPU.')
    parser.add_argument('--poll_seconds', type=int, default=10)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pool_dir_abs = os.path.abspath(args.pool_dir)
    output_dir_abs = os.path.abspath(args.output_dir)
    core_clustering_dir_abs = os.path.abspath(args.core_clustering_dir)
    log_dir = args.log_dir or os.path.join(repo_root, 'logs', 'self_via_core_clustering')
    os.makedirs(log_dir, exist_ok=True)

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

    jobs = build_jobs(args.entities, args.seeds, n_time_by_entity, args.ucr_domain_name)
    print(f'{len(jobs)} jobs queued. Running up to {args.max_parallel} at a time on GPU {args.gpu}. '
          f'Per-job logs in {log_dir}/', flush=True)

    pending = list(jobs)
    running = {}  # run_id -> (proc, log_file, model_path, start_time)
    finished = []

    while pending or running:
        while pending and len(running) < args.max_parallel:
            job = pending.pop(0)
            run_id, proc, log_file, model_path = launch(
                job, args, pool_dir_abs, output_dir_abs, core_clustering_dir_abs, log_dir)
            running[run_id] = (proc, log_file, model_path, time.time())

        time.sleep(args.poll_seconds)

        for run_id in list(running.keys()):
            proc, log_file, model_path, start_time = running[run_id]
            ret = proc.poll()
            if ret is not None:
                log_file.close()
                elapsed_min = (time.time() - start_time) / 60
                ok = ret == 0 and os.path.isfile(model_path)
                status = 'OK' if ok else f'FAILED (exit {ret}) -- see {log_dir}/{run_id}.log'
                print(f'[done] {run_id}: {status} after {elapsed_min:.1f} min '
                      f'({len(running) - 1 + len(pending)} left)', flush=True)
                finished.append((run_id, ok, model_path))
                del running[run_id]

    print('\nSummary:')
    for run_id, ok, model_path in finished:
        print(f'  {run_id}: {"OK" if ok else "FAILED"} {model_path}')
    failed = [run_id for run_id, ok, _ in finished if not ok]
    if failed:
        print(f'\nWARNING: {len(failed)} run(s) failed: {failed}')


if __name__ == '__main__':
    run()
