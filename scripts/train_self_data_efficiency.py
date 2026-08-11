"""
Data-efficiency experiment for Self models: for each UCR entity, trains
with only n% of the AVAILABLE TRAINING data (n in --n_pcts, default 1/3/5/
10/25/50/75/100) + RedLamp's own anomaly injection augmentation (the same
Loader_aug mechanism main.py's own Self training already always uses), to
see how much that augmentation can make up for having very little real
data. Answers: "how little raw training data can we get away with, given
injection augmentation multiplies the effective training signal?"

n% truncation applies ONLY to the TRAIN split, never validation:
  - Load the entity's full train-portion Y (train_end length, i.e. before
    RedLamp's own 90/10 split) via loaders.load.load_anomaly_archive.
  - Split 90/10 exactly like loaders/load.py's own convention (FIRST 90% ->
    train, LAST 10% -> val).
  - Val stays this FULL, FIXED 10% slice for every n_pct value of a given
    entity -- early-stopping quality shouldn't also degrade with n_pct;
    only the amount of TRAINING data should vary, so this isolates that
    one variable cleanly.
  - Train is truncated to the FIRST n_pct% of that 90% train sub-portion
    (earliest-collected data, as if you'd only gathered that much history).
  - At n_pct=100, this exactly reproduces the existing Self baseline's own
    split -- a built-in sanity anchor to cross-check against.

window_step is fixed PER ENTITY from main.py's own dynamic rule (train_end
<10000 -> 1, <100000 -> 10, else 100), computed ONCE from the entity's
FULL/original train_end and reused across every n_pct for that entity --
NOT recomputed from the truncated length, so window_step never becomes a
second, confounded variable alongside n_pct.

Trains via main.REDLAMP + loaders.loader_aug.Loader_aug directly (RedLamp's
own training loop/injection code, matching main.py's Self training
hyperparameters exactly: window_size=100, batch_size=128, anomaly_types=
the standard 12-type list, min_range=1, min_features=max_features=1) --
does not go through main.py's CLI, since main.py has no hook for
truncating training data before its internal split.

Two modes:
  - Worker (default): trains ONE (--entity, --n_pct, --seed) job.
  - Orchestrator (--orchestrate): builds the full entities x n_pcts x seeds
    job grid and launches worker jobs (subprocess copies of this same
    script) concurrently, --max_parallel at a time -- same pattern as
    run_multiseed_training.py / train_ucr_self_via_core_clustering.py.

SCALE WARNING: the default entity list is ALL 250 anomaly_archive
subdatasets. 250 entities x 8 n_pcts x 3 seeds = 6000 jobs. Even filling
GPU headroom via --max_parallel, this is a multi-day run -- pass a smaller
--entities/--n_pcts for a pilot first if you want a faster initial look.

Resumable: skips a (entity, n_pct, seed) job whose bestmodel.pkl already
exists (pass --force to override). An entity/n_pct combo too short to form
even one training window (n_pct% of its 90% train sub-portion < window_size,
or the entity's own val slice < window_size regardless of n_pct) is skipped
and logged, not treated as an error -- an expected outcome at very low
n_pct for short entities, not a bug.
"""
import argparse
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import main
import utils
import cross_inference as ci
import domain_generalization as dg
from loaders.load import load_anomaly_archive
from loaders.dataset import Entity
from loaders.loader_aug import Loader_aug

DATASET = 'anomaly_archive'
DEFAULT_ENTITIES = [str(i).zfill(3) for i in range(1, 251)]
DEFAULT_N_PCTS = [1, 3, 5, 10, 25, 50, 75, 100]
DEFAULT_SEEDS = [0, 1, 2]
WINDOW_SIZE = dg.WINDOW_SIZE


def window_step_for(train_end):
    if train_end < 10000:
        return 1
    elif train_end < 100000:
        return 10
    else:
        return 100


def fmt_pct(n_pct):
    return str(int(n_pct)) if float(n_pct).is_integer() else str(n_pct)


def load_truncated_split(entity, n_pct):
    """Returns (train_entity, val_entity, window_step), or None if too short
    for even one training window at this n_pct (or the entity's own val
    slice is too short regardless of n_pct)."""
    meta = main.get_meta_data(entity)
    train_end = int(meta['train_end'])
    window_step = window_step_for(train_end)

    train_ds = load_anomaly_archive(group='train', datasets=entity, downsampling=1,
                                     root_dir='./dataset', validation=False, verbose=False)
    Y_full = train_ds.entities[0].Y

    split_point = int(Y_full.shape[1] * 0.9)
    full_train = Y_full[:, :split_point]
    val = Y_full[:, split_point:]
    if val.shape[1] < WINDOW_SIZE:
        return None

    n_time_n = int(round(full_train.shape[1] * n_pct / 100))
    if n_time_n < WINDOW_SIZE:
        return None

    train_n = full_train[:, :n_time_n]
    train_entity = Entity(Y=train_n, name=f'{entity}_train_n{fmt_pct(n_pct)}')
    val_entity = Entity(Y=val, name=f'{entity}_val')
    return train_entity, val_entity, window_step


def train_one(entity, n_pct, seed, output_dir, gpu, epochs, force):
    model_dir = os.path.join(output_dir, entity, f'n{fmt_pct(n_pct)}_seed{seed}')
    if os.path.isfile(os.path.join(model_dir, 'bestmodel.pkl')) and not force:
        print(f'[skip] {model_dir}/bestmodel.pkl exists -- reusing')
        return True

    split = load_truncated_split(entity, n_pct)
    if split is None:
        print(f'[skip] {entity}/n{fmt_pct(n_pct)}: too short for even one window '
              f'(entity itself, or the truncated n_pct train slice)')
        return True  # expected outcome at low n_pct, not a failure
    train_entity, val_entity, window_step = split
    os.makedirs(model_dir, exist_ok=True)

    device = utils.init_dl_program(gpu, seed=seed)
    loader_kwargs = dict(batch_size=128, window_size=WINDOW_SIZE, window_step=window_step,
                          anomaly_types=ci.ANOMALY_TYPES, min_range=1,
                          min_features=dg.CFG['min_features'], max_features=dg.CFG['max_features'],
                          fast_sampling=False, verbose=False)
    train_dl = Loader_aug(dataset=train_entity, shuffle=True, **loader_kwargs)
    val_dl = Loader_aug(dataset=val_entity, shuffle=True, **loader_kwargs)
    print(f'{entity}/n{fmt_pct(n_pct)}/seed{seed}: train={train_entity.Y.shape[1]}pts ({len(train_dl)} windows), '
          f'val={val_entity.Y.shape[1]}pts ({len(val_dl)} windows), window_step={window_step}', flush=True)

    model_args = ci.build_model_args(dg.CFG, WINDOW_SIZE)
    params = utils.AttrDict(batch_size=128, lr=0.001, epoch=epochs, max_grad_norm=1.0, seed=seed)
    params.override(main.model_parameters(model_args))

    model = main.REDLAMP(model_dir=model_dir, params=params, device=device)
    model.train(train_dl, val_dl)
    return os.path.isfile(os.path.join(model_dir, 'bestmodel.pkl'))


def launch_worker(job, args, log_dir):
    tag = f"{job['entity']}_n{fmt_pct(job['n_pct'])}_seed{job['seed']}"
    log_path = os.path.join(log_dir, f'{tag}.log')
    log_file = open(log_path, 'w')
    cmd = [sys.executable, '-u', os.path.abspath(__file__),
           '--entity', job['entity'], '--n_pct', str(job['n_pct']), '--seed', str(job['seed']),
           '--output_dir', args.output_dir, '--gpu', str(args.gpu), '--epochs', str(args.epochs)]
    if args.force:
        cmd.append('--force')
    print(f'[launch] {tag} -> {log_path}', flush=True)
    proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
    return tag, proc, log_file


def orchestrate(args):
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_dir = args.log_dir or os.path.join(repo_root, 'logs', 'self_data_efficiency')
    os.makedirs(log_dir, exist_ok=True)

    jobs = [dict(entity=e, n_pct=n, seed=s) for s in args.seeds for n in args.n_pcts for e in args.entities]
    print(f'{len(jobs)} jobs queued ({len(args.entities)} entities x {len(args.n_pcts)} n_pcts x '
          f'{len(args.seeds)} seeds). Running up to {args.max_parallel} at a time on GPU {args.gpu}. '
          f'Logs in {log_dir}/', flush=True)

    pending = list(jobs)
    running = {}  # tag -> (proc, log_file, start_time)
    finished = []
    while pending or running:
        while pending and len(running) < args.max_parallel:
            job = pending.pop(0)
            tag, proc, log_file = launch_worker(job, args, log_dir)
            running[tag] = (proc, log_file, time.time())

        time.sleep(args.poll_seconds)

        for tag in list(running.keys()):
            proc, log_file, start_time = running[tag]
            ret = proc.poll()
            if ret is not None:
                log_file.close()
                elapsed_min = (time.time() - start_time) / 60
                status = 'OK' if ret == 0 else f'FAILED (exit {ret}) -- see {log_dir}/{tag}.log'
                print(f'[done] {tag}: {status} after {elapsed_min:.1f} min '
                      f'({len(running) - 1 + len(pending)} left)', flush=True)
                finished.append((tag, ret))
                del running[tag]

    n_failed = sum(1 for _, ret in finished if ret != 0)
    print(f'All {len(jobs)} jobs finished. {n_failed} failed.', flush=True)
    for tag, ret in finished:
        if ret != 0:
            print(f'  FAILED: {tag} -- see {log_dir}/{tag}.log', flush=True)


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--orchestrate', action='store_true')
    # worker-mode args
    parser.add_argument('--entity', default=None)
    parser.add_argument('--n_pct', type=float, default=None)
    parser.add_argument('--seed', type=int, default=0)
    # orchestrator-mode args
    parser.add_argument('--entities', nargs='+', default=DEFAULT_ENTITIES)
    parser.add_argument('--n_pcts', type=float, nargs='+', default=DEFAULT_N_PCTS)
    parser.add_argument('--seeds', type=int, nargs='+', default=DEFAULT_SEEDS)
    parser.add_argument('--max_parallel', type=int, default=6)
    parser.add_argument('--poll_seconds', type=int, default=10)
    parser.add_argument('--log_dir', default=None)
    # shared args
    parser.add_argument('--output_dir', default='./result/DS_2/achievability/self_data_efficiency_models')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    if args.orchestrate:
        orchestrate(args)
    else:
        if args.entity is None or args.n_pct is None:
            parser.error('worker mode needs --entity and --n_pct (or pass --orchestrate)')
        ok = train_one(args.entity, args.n_pct, args.seed, args.output_dir, args.gpu, args.epochs, args.force)
        sys.exit(0 if ok else 1)


if __name__ == '__main__':
    run()
