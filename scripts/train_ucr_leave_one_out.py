"""
Leave-one-RECORDING-out pooled training within the UCR "PowerDemand" group
-- NOT leave-one-entity-out (see below for why that was wrong).

The 8 UCR PowerDemand entities are 4 distinct underlying recordings, each
registered TWICE in the UCR Anomaly Archive (a DISTORTED variant and a
plain variant) -- e.g. entity 044 (DISTORTEDPowerDemand1) and 152
(PowerDemand1) are the SAME real recording, confirmed both by identical
train_end/anomaly boundaries in result/DS_1/entity_metadata.csv AND by
visually near-identical waveforms (same spikes at the same positions).

Leaving out just ONE entity (e.g. 044) while keeping its near-twin (152)
in the training pool doesn't test generalization to an unseen pattern --
the model has already seen a near-duplicate of the exact signal it's
tested on. That's almost certainly why the original entity-level LOO
(and the AnomSim+UCR version in run_ucr_anomsim_loo_training.py) showed
such unusually strong, consistent gap_closed numbers (99-108% for
UCR-only, 65-117% with AnomSim added) -- likely inflated by this
near-duplicate leakage rather than reflecting genuine generalization.

This version excludes BOTH variants of a recording together (4 folds
instead of 8), training ONE pooled model per recording per seed from the
OTHER 6 entities (3 other recordings x 2 variants). Scoring is a separate,
fast, sequential step -- see score_ucr_leave_one_out.py.

Pooling follows domain_generalization.py's own established pattern
(load_single_entity_train_val + loaders.dataset.Dataset merging +
Loader_aug + main.REDLAMP.train) -- WINDOW_SIZE/WINDOW_STEP/BATCH_SIZE
match that script's pooled-model convention, not main.py's per-entity
CLI defaults. Pool size (6 entities, window_step=10 fixed, 12 anomaly
types) produces ~100k train windows/epoch -- NOT a bug, just what that
combination works out to; ~50-60s/epoch on a real GPU is plausible for
this size, especially if the GPU is shared with other jobs (check
nvidia-smi if unsure).

Two modes:
  - Worker (default): trains ONE (--recording, --seed) job.
  - Orchestrator (--orchestrate): builds the recordings x seeds job grid
    and launches worker jobs (subprocess copies of this same script)
    concurrently, --max_parallel at a time -- same pattern as
    run_multiseed_training.py / train_self_data_efficiency.py.

Resumable: skips a (recording, seed) job whose bestmodel.pkl already
exists (pass --force to override).
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
from loaders.dataset import Dataset

DATASET = 'anomaly_archive'
RECORDING_PAIRS = {
    'PowerDemand1': ['044', '152'],
    'PowerDemand2': ['045', '153'],
    'PowerDemand3': ['046', '154'],
    'PowerDemand4': ['047', '155'],
}
DEFAULT_SEEDS = [0, 1, 2]


def all_entities():
    return [e for pair in RECORDING_PAIRS.values() for e in pair]


def train_one(recording, seed, run_name, gpu, epochs, force):
    held_out_entities = RECORDING_PAIRS[recording]
    pool = [e for e in all_entities() if e not in held_out_entities]
    model_dir = f'./result/{run_name}/_loo_pair/without_{recording}_seed{seed}'
    if os.path.isfile(f'{model_dir}/bestmodel.pkl') and not force:
        print(f'[skip] {model_dir}/bestmodel.pkl exists -- reusing')
        return True
    os.makedirs(model_dir, exist_ok=True)

    device = utils.init_dl_program(gpu, seed=seed)
    train_entities, val_entities = [], []
    for entity in pool:
        train_entity, val_entity = dg.load_single_entity_train_val(DATASET, entity)
        train_entities.append(train_entity)
        val_entities.append(val_entity)
    train_dataset = Dataset(entities=train_entities, name=f'without_{recording}-train')
    val_dataset = Dataset(entities=val_entities, name=f'without_{recording}-val')
    train_dl = dg.wrap_loader(train_dataset, shuffle=True)
    val_dl = dg.wrap_loader(val_dataset, shuffle=True)
    print(f'without_{recording}/seed{seed}: excluded={held_out_entities}, pool={pool} '
          f'({len(train_dl)} train windows / {len(val_dl)} val windows)', flush=True)

    model_args = ci.build_model_args(dg.CFG, dg.WINDOW_SIZE)
    params = utils.AttrDict(batch_size=dg.BATCH_SIZE, lr=0.001, epoch=epochs, max_grad_norm=1.0, seed=seed)
    params.override(main.model_parameters(model_args))

    model = main.REDLAMP(model_dir=model_dir, params=params, device=device)
    model.train(train_dl, val_dl)
    return os.path.isfile(f'{model_dir}/bestmodel.pkl')


def launch_worker(job, args, log_dir):
    tag = f"without_{job['recording']}_seed{job['seed']}"
    log_path = os.path.join(log_dir, f'{tag}.log')
    log_file = open(log_path, 'w')
    cmd = [sys.executable, '-u', os.path.abspath(__file__),
           '--recording', job['recording'], '--seed', str(job['seed']),
           '--run_name', args.run_name, '--gpu', str(args.gpu), '--epochs', str(args.epochs)]
    if args.force:
        cmd.append('--force')
    print(f'[launch] {tag} -> {log_path}', flush=True)
    proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
    return tag, proc, log_file


def orchestrate(args):
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_dir = args.log_dir or os.path.join(repo_root, 'logs', 'ucr_leave_one_recording_out')
    os.makedirs(log_dir, exist_ok=True)

    jobs = [dict(recording=r, seed=s) for s in args.seeds for r in args.recordings]
    print(f'{len(jobs)} jobs queued ({len(args.recordings)} recordings x {len(args.seeds)} seeds). '
          f'Running up to {args.max_parallel} at a time on GPU {args.gpu}. Logs in {log_dir}/', flush=True)

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
    parser.add_argument('--recording', default=None, choices=list(RECORDING_PAIRS.keys()))
    parser.add_argument('--seed', type=int, default=0)
    # orchestrator-mode args
    parser.add_argument('--recordings', nargs='+', default=list(RECORDING_PAIRS.keys()))
    parser.add_argument('--seeds', type=int, nargs='+', default=DEFAULT_SEEDS)
    parser.add_argument('--max_parallel', type=int, default=4)
    parser.add_argument('--poll_seconds', type=int, default=10)
    parser.add_argument('--log_dir', default=None)
    # shared args
    parser.add_argument('--run_name', default='test')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    if args.orchestrate:
        orchestrate(args)
    else:
        if args.recording is None:
            parser.error('worker mode needs --recording (or pass --orchestrate)')
        ok = train_one(args.recording, args.seed, args.run_name, args.gpu, args.epochs, args.force)
        sys.exit(0 if ok else 1)


if __name__ == '__main__':
    run()
