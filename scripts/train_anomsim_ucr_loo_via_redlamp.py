"""
RedLamp-NATIVE leave-one-out trainer -- the RedLamp-side counterpart to
run_ucr_anomsim_loo_training.py (which is Core-Clustering-side), so both
train on the EXACT SAME pool composition (AnomSim_v1's 144 entities + the
7 OTHER UCR PowerDemand entities, held-out entity excluded) and differ
ONLY in which codebase does the actual training.

Uses main.REDLAMP + loaders.loader_aug.Loader_aug directly (RedLamp's own
training loop, its own anomaly-injection code, NOT AnomSim's), following
domain_generalization.py's own established pooling pattern exactly
(load_single_entity_train_val + Dataset + wrap_loader + main.REDLAMP.train)
-- see that module and train_ucr_leave_one_out.py for the precedent this
extends.

AnomSim_v1 entities have no train/test split concept (unlike UCR), so each
one gets the SAME 90/10 temporal split UCR entities already get via
load_anomaly_archive's own convention (loaders/load.py's
train_length = int(Y.shape[1] * 0.9)) -- entities are just wrapped
directly via loaders.dataset.Entity(Y=..., name=...), which needs nothing
but a raw array (labels default to all-zero, mask to all-ones, matching
how RedLamp's own train-split Entities are built with no labels either).

WINDOW_SIZE=100/WINDOW_STEP=10/BATCH_SIZE=128 match
domain_generalization.py's own pooled-model convention (also
Core-Clustering online_cli's defaults) -- NOT main.py's per-entity dynamic
window_step (see compare_redlamp_vs_core_clustering.py's docstring for why
that distinction matters).

Meant for the server (needs ./dataset/AnomalyArchive for the UCR side);
the AnomSim_v1 side works anywhere. Resumable: an existing
without_{X}_seed{seed}/bestmodel.pkl is reused unless --force.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import main
import utils
import cross_inference as ci
import domain_generalization as dg
from loaders.dataset import Dataset, Entity

DATASET = 'anomaly_archive'
DEFAULT_ENTITIES = ['044', '045', '046', '047', '152', '153', '154', '155']
DEFAULT_SEEDS = [0, 1, 2]


def load_anomsim_entities_as_redlamp(anomsim_dir):
    """Mirrors loaders/load.py's own train/val split (train_length =
    int(n_time * 0.9), first 90% -> train, last 10% -> val) so AnomSim_v1
    entities are pooled under the exact same convention as UCR entities in
    this same pool, not a bespoke one."""
    manifest_path = os.path.join(anomsim_dir, '_manifest.jsonl')
    train_entities, val_entities = [], []
    with open(manifest_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            meta = json.loads(line)
            entity_dir = meta['entity_dir']
            Y = np.load(os.path.join(anomsim_dir, entity_dir, 'Y.npy')).astype(np.float64)
            train_length = int(Y.shape[1] * 0.9)
            train_entities.append(Entity(Y=Y[:, :train_length], name=f'{entity_dir}_train'))
            val_entities.append(Entity(Y=Y[:, train_length:], name=f'{entity_dir}_val'))
    return train_entities, val_entities


def train_pooled_without(held_out, ucr_entities, anomsim_train_entities, anomsim_val_entities,
                          run_name, seed, device, force=False):
    ucr_pool = [e for e in ucr_entities if e != held_out]
    model_dir = f'./result/{run_name}/_loo_redlamp/without_{held_out}_seed{seed}'
    if os.path.isfile(f'{model_dir}/bestmodel.pkl') and not force:
        print(f'[skip] {model_dir}/bestmodel.pkl exists -- reusing')
        return model_dir
    os.makedirs(model_dir, exist_ok=True)

    train_entities, val_entities = list(anomsim_train_entities), list(anomsim_val_entities)
    for entity in ucr_pool:
        train_entity, val_entity = dg.load_single_entity_train_val(DATASET, entity)
        train_entities.append(train_entity)
        val_entities.append(val_entity)

    train_dataset = Dataset(entities=train_entities, name=f'without_{held_out}-train')
    val_dataset = Dataset(entities=val_entities, name=f'without_{held_out}-val')
    train_dl = dg.wrap_loader(train_dataset, shuffle=True)
    val_dl = dg.wrap_loader(val_dataset, shuffle=True)
    print(f'Training without_{held_out}/seed{seed}: {len(train_entities)} entities pooled '
          f'({len(anomsim_train_entities)} AnomSim_v1 + {len(ucr_pool)} UCR), '
          f'{len(train_dl)} train windows / {len(val_dl)} val windows')

    model_args = ci.build_model_args(dg.CFG, dg.WINDOW_SIZE)
    params = utils.AttrDict(batch_size=dg.BATCH_SIZE, lr=0.001, epoch=100, max_grad_norm=1.0, seed=seed)
    params.override(main.model_parameters(model_args))

    model = main.REDLAMP(model_dir=model_dir, params=params, device=device)
    model.train(train_dl, val_dl)
    return model_dir


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--entities', nargs='+', default=DEFAULT_ENTITIES)
    parser.add_argument('--seeds', nargs='+', type=int, default=DEFAULT_SEEDS)
    parser.add_argument('--anomsim_dir', default='../AnomSim/data/AnomSim_v1')
    parser.add_argument('--run_name', default='pipeline_compare')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    device = utils.init_dl_program(args.gpu, seed=args.seeds[0])

    print(f'Loading {args.anomsim_dir} entities once, reused across all folds/seeds...')
    anomsim_train_entities, anomsim_val_entities = load_anomsim_entities_as_redlamp(args.anomsim_dir)
    print(f'  {len(anomsim_train_entities)} AnomSim_v1 entities loaded')

    results = []
    for seed in args.seeds:
        for entity in args.entities:
            model_dir = train_pooled_without(entity, args.entities, anomsim_train_entities,
                                              anomsim_val_entities, args.run_name, seed, device,
                                              force=args.force)
            ok = os.path.isfile(f'{model_dir}/bestmodel.pkl')
            results.append((entity, seed, ok, model_dir))
            print(f'  {"OK" if ok else "FAILED"} -> {model_dir}')

    print('\nSummary:')
    for entity, seed, ok, model_dir in results:
        print(f'  {entity}/seed{seed}: {"OK" if ok else "FAILED"} {model_dir}')
    failed = [(e, s) for e, s, ok, _ in results if not ok]
    if failed:
        print(f'\nWARNING: {len(failed)} run(s) failed: {failed}')


if __name__ == '__main__':
    run()
