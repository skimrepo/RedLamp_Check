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
instead of 8), trains ONE pooled model per recording per seed from the
OTHER 6 entities (3 other recordings x 2 variants), then scores that SAME
model against BOTH held-out entities of that recording. Comparing these
numbers against the old entity-level ones (still in
result/DS_2/achievability/ucr_leave_one_out.csv from a prior run, not
overwritten by --out_csv's new default name) is itself informative --
a large gap between the two directly measures how much of the old
numbers was near-duplicate leakage.

Self baseline numbers are NOT retrained -- pulled directly from
result/Experiment_2/Results/ucr_results.xlsx's 'Per-Entity Comparison'
sheet, same as before.

Pooling follows domain_generalization.py's own established pattern
(load_single_entity_train_val + loaders.dataset.Dataset merging +
Loader_aug + main.REDLAMP.train) -- WINDOW_SIZE/WINDOW_STEP/BATCH_SIZE
match that script's pooled-model convention, not main.py's per-entity
CLI defaults.

Meant to run on the machine that actually has ./dataset/AnomalyArchive.
4 folds x len(seeds) is cheaper than the old 8 folds x len(seeds) despite
each fold's pool being slightly smaller (6 entities instead of 7).

Resumable: an existing without_{recording}_seed{seed}/bestmodel.pkl is
reused unless --force.
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import main
import utils
import cross_inference as ci
import domain_generalization as dg
import full_reproduction_metrics as frm
from loaders.dataset import Dataset

DATASET = 'anomaly_archive'
RECORDING_PAIRS = {
    'PowerDemand1': ['044', '152'],
    'PowerDemand2': ['045', '153'],
    'PowerDemand3': ['046', '154'],
    'PowerDemand4': ['047', '155'],
}
DEFAULT_SEEDS = [0, 1, 2]


def all_entities(recording_pairs):
    return [e for pair in recording_pairs.values() for e in pair]


def train_pooled_without(recording, held_out_entities, entities, run_name, seed, device, force=False):
    pool = [e for e in entities if e not in held_out_entities]
    model_dir = f'./result/{run_name}/_loo_pair/without_{recording}_seed{seed}'
    if os.path.isfile(f'{model_dir}/bestmodel.pkl') and not force:
        print(f'[skip] {model_dir}/bestmodel.pkl exists -- reusing')
        return model_dir
    os.makedirs(model_dir, exist_ok=True)

    train_entities, val_entities = [], []
    for entity in pool:
        train_entity, val_entity = dg.load_single_entity_train_val(DATASET, entity)
        train_entities.append(train_entity)
        val_entities.append(val_entity)
    train_dataset = Dataset(entities=train_entities, name=f'without_{recording}-train')
    val_dataset = Dataset(entities=val_entities, name=f'without_{recording}-val')
    train_dl = dg.wrap_loader(train_dataset, shuffle=True)
    val_dl = dg.wrap_loader(val_dataset, shuffle=True)
    print(f'Training without_{recording}/seed{seed}: excluded={held_out_entities}, pool={pool} '
          f'({len(train_dl)} train windows / {len(val_dl)} val windows)')

    model_args = ci.build_model_args(dg.CFG, dg.WINDOW_SIZE)
    params = utils.AttrDict(batch_size=dg.BATCH_SIZE, lr=0.001, epoch=100, max_grad_norm=1.0, seed=seed)
    params.override(main.model_parameters(model_args))

    model = main.REDLAMP(model_dir=model_dir, params=params, device=device)
    model.train(train_dl, val_dl)
    return model_dir


def load_baseline_scores(ucr_xlsx, entities):
    df = pd.read_excel(ucr_xlsx, sheet_name='Per-Entity Comparison')
    df['entity'] = df['entity'].astype(str).str.zfill(3)
    return df[df['entity'].isin(entities)].set_index('entity')


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--recordings', nargs='+', default=list(RECORDING_PAIRS.keys()))
    parser.add_argument('--seeds', type=int, nargs='+', default=DEFAULT_SEEDS)
    parser.add_argument('--run_name', default='test')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--ucr_xlsx', default='./result/Experiment_2/Results/ucr_results.xlsx')
    parser.add_argument('--out_csv', default='./result/DS_2/achievability/ucr_leave_one_recording_out_per_seed.csv')
    parser.add_argument('--out_avg_csv', default='./result/DS_2/achievability/ucr_leave_one_recording_out_avg.csv')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    entities = all_entities({r: RECORDING_PAIRS[r] for r in args.recordings})
    device = utils.init_dl_program(args.gpu, seed=args.seeds[0])
    model_args = ci.build_model_args(dg.CFG, dg.WINDOW_SIZE)

    baseline = load_baseline_scores(args.ucr_xlsx, entities)
    missing_baseline = set(entities) - set(baseline.index)
    if missing_baseline:
        print(f'[warn] no Self/CrossAnomSim score found in {args.ucr_xlsx} for: {sorted(missing_baseline)}')

    rows = []
    for recording in args.recordings:
        held_out_entities = RECORDING_PAIRS[recording]
        for seed in args.seeds:
            model_dir = train_pooled_without(recording, held_out_entities, entities, args.run_name, seed, device,
                                              force=args.force)

            params = utils.AttrDict(seed=seed)
            params.override(main.model_parameters(model_args))

            for held_out in held_out_entities:
                result = frm.score_entity(args.run_name, DATASET, held_out, seed, params, device, model_dir=model_dir)
                if result is None:
                    print(f'[skip] {held_out}/seed{seed}: LOO scoring failed/unavailable '
                          f'(no real anomaly in test labels, or length mismatch)')
                    continue

                row = dict(recording=recording, entity=held_out, seed=seed)
                has_baseline = held_out in baseline.index
                for m in frm.METRIC_KEYS:
                    row[f'Self_{m}'] = baseline.loc[held_out, f'{m}_self'] if has_baseline else None
                    row[f'CrossAnomSim_{m}'] = baseline.loc[held_out, f'{m}_cross_anomsim'] if has_baseline else None
                    row[f'LOO_{m}'] = result['metrics'][m]
                row['Self_peak_in_range'] = baseline.loc[held_out, 'peak_in_range_self'] if has_baseline else None
                row['CrossAnomSim_peak_in_range'] = baseline.loc[held_out, 'peak_in_range_cross_anomsim'] if has_baseline else None
                row['LOO_peak_in_range'] = result['peak_in_range']
                rows.append(row)
                self_vus = row['Self_VUS_ROC']
                print(f'{recording}/{held_out}/seed{seed}: LOO VUS_ROC={result["metrics"]["VUS_ROC"]:.3f}'
                      + (f' (Self={self_vus:.3f})' if self_vus is not None else ' (no Self score found)'))

    per_seed_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    per_seed_df.to_csv(args.out_csv, index=False)
    print(f'\nWrote {args.out_csv} ({len(per_seed_df)} rows)')

    avg_rows = []
    for entity in entities:
        sub = per_seed_df[per_seed_df['entity'] == entity] if not per_seed_df.empty else per_seed_df
        if sub.empty:
            continue
        row = dict(recording=sub['recording'].iloc[0], entity=entity, n_seeds=len(sub))
        for m in frm.METRIC_KEYS + ['peak_in_range']:
            row[f'Self_{m}'] = sub[f'Self_{m}'].iloc[0]
            row[f'CrossAnomSim_{m}'] = sub[f'CrossAnomSim_{m}'].iloc[0]
            row[f'LOO_{m}_mean'] = sub[f'LOO_{m}'].mean()
            row[f'LOO_{m}_std'] = sub[f'LOO_{m}'].std() if len(sub) > 1 else 0.0
        self_v, cross_v = row['Self_VUS_ROC'], row['CrossAnomSim_VUS_ROC']
        row['gap_closed'] = (row['LOO_VUS_ROC_mean'] - cross_v) / (self_v - cross_v) if self_v != cross_v else None
        avg_rows.append(row)

    avg_df = pd.DataFrame(avg_rows)
    os.makedirs(os.path.dirname(args.out_avg_csv), exist_ok=True)
    avg_df.to_csv(args.out_avg_csv, index=False)
    print(f'Wrote {args.out_avg_csv} ({len(avg_df)} rows, averaged over seeds {args.seeds})')
    if not avg_df.empty:
        print(avg_df[['recording', 'entity', 'n_seeds', 'Self_VUS_ROC', 'CrossAnomSim_VUS_ROC',
                       'LOO_VUS_ROC_mean', 'LOO_VUS_ROC_std', 'gap_closed']].to_string(index=False))


if __name__ == '__main__':
    run()
