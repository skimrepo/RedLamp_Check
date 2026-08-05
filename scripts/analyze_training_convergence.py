"""
DS_3 step 1: did Self's UCR models actually converge? Extracts, for every
UCR entity's Self model, the train/validation MSE (reconstruction) and
Classification loss AT THE BEST EPOCH (the one bestmodel.pkl was saved from)
-- so it can be compared against Cross-AnomSim's own convergence (already
fully available in Core-Clustering's run_summary.json, no extraction needed
there).

main.py's Trainer (the class training Self models) writes one plain-text
file per epoch-loss-series into each entity's model_dir (see main.py's
np.savetxt calls): train_loss.txt/valid_loss.txt (combined, used for
early-stopping), train_loss_ae.txt/valid_loss_ae.txt (MSE reconstruction),
train_loss_c.txt/valid_loss_c.txt (classification). There's no JSON summary
and no explicit "best epoch" marker -- the best epoch is wherever
valid_loss.txt hits its minimum (that's the exact point bestmodel.pkl was
saved from, since main.py's Trainer only checkpoints on improvement).

organize_experiment1.py moves an entity's whole seed-level folder (not just
bestmodel.pkl), so these loss-log text files are still sitting right next to
bestmodel.pkl at whatever location ci.discover_entity resolves to -- no
special-casing needed for already-moved entities.

Pure file parsing: no model loading, no GPU, no re-inference. Should run
fast even across all ~247 UCR entities. Resumable: entities already in the
output CSV are skipped unless --force.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cross_inference as ci
import full_reproduction_metrics as frm

DATASET = 'anomaly_archive'
LOSS_FILES = {
    'valid_loss': 'valid_loss.txt',
    'train_mse': 'train_loss_ae.txt',
    'val_mse': 'valid_loss_ae.txt',
    'train_ce': 'train_loss_c.txt',
    'val_ce': 'valid_loss_c.txt',
}


def read_convergence(model_dir):
    curves = {}
    for key, fname in LOSS_FILES.items():
        path = os.path.join(model_dir, fname)
        if not os.path.isfile(path):
            return None
        curves[key] = np.loadtxt(path)
        if curves[key].ndim == 0:
            curves[key] = curves[key].reshape(1)

    valid_loss = curves['valid_loss']
    if len(valid_loss) == 0:
        return None
    best_epoch = int(np.argmin(valid_loss))

    return dict(
        train_mse=float(curves['train_mse'][best_epoch]),
        val_mse=float(curves['val_mse'][best_epoch]),
        train_ce=float(curves['train_ce'][best_epoch]),
        val_ce=float(curves['val_ce'][best_epoch]),
        best_epoch=best_epoch,
        total_epochs=len(valid_loss),
    )


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', default='test')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out_csv', default='./result/DS_3/convergence/self_convergence.csv')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    out_dir = os.path.dirname(args.out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    entities = frm.discover_dataset_entities(args.run_name, DATASET)
    print(f'{len(entities)} entities discovered for {DATASET} (run_name={args.run_name})')

    rows = []
    already_done = set()
    if not args.force and os.path.isfile(args.out_csv):
        prior = pd.read_csv(args.out_csv)
        rows = prior.to_dict('records')
        already_done = set(prior['entity'].astype(str).str.zfill(3))
        print(f'Resuming from {args.out_csv}: {len(already_done)} entities already done.')

    for entity in entities:
        if entity in already_done:
            continue
        try:
            model_dir, _ = ci.discover_entity(args.run_name, DATASET, entity, args.seed)
        except FileNotFoundError:
            print(f'[skip] {entity}: no trained model found')
            continue

        conv = read_convergence(model_dir)
        if conv is None:
            print(f'[skip] {entity}: loss log files missing/empty at {model_dir}')
            continue

        row = dict(entity=entity, **conv)
        rows.append(row)
        print(f'  {entity}: {row}')
        pd.DataFrame(rows).to_csv(args.out_csv, index=False)

    print(f'Done. {len(rows)} entities analyzed. Wrote {args.out_csv}')


if __name__ == '__main__':
    run()
