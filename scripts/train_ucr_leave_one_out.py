"""
Leave-one-entity-out pooled training within a small, explicit UCR entity
group -- for each entity X in --entities, trains a NEW model pooled from
the OTHER entities in the group (X itself is never in that pool), then
scores that model on X's own test set via
full_reproduction_metrics.score_entity(model_dir=<pooled model>).

Default --entities are the 8 UCR "PowerDemand" entities: 044-047
(DISTORTED variant) and 152-155 (plain variant) -- 4 distinct underlying
recordings, each registered twice in the UCR Anomaly Archive.

Self baseline numbers are NOT retrained here -- pulled directly from
result/Experiment_2/Results/ucr_results.xlsx's 'Per-Entity Comparison'
sheet (already-trained, already-scored Self models; re-evaluating a fixed
deterministic model would just reproduce the same numbers).

Pooling follows domain_generalization.py's own established pattern
(load_single_entity_train_val + loaders.dataset.Dataset merging +
Loader_aug + main.REDLAMP.train) -- WINDOW_SIZE/WINDOW_STEP/BATCH_SIZE
match that script's pooled-model convention exactly, not main.py's
per-entity CLI defaults.

Meant to run on the machine that actually has ./dataset/AnomalyArchive
(the raw UCR data isn't present on this development machine) -- training
8 small pools (7 entities each) is cheap (each Self model here trains in
well under a minute on CPU per prior runs of this size), so this simply
loops through all --entities in one process, no sharding needed.

Resumable: an existing without_{X}/{seed}/bestmodel.pkl is reused unless
--force.
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
DEFAULT_ENTITIES = ['044', '045', '046', '047', '152', '153', '154', '155']


def train_pooled_without(held_out, entities, run_name, seed, device, force=False):
    pool = [e for e in entities if e != held_out]
    model_dir = f'./result/{run_name}/_loo/without_{held_out}/{seed}'
    if os.path.isfile(f'{model_dir}/bestmodel.pkl') and not force:
        print(f'[skip] {model_dir}/bestmodel.pkl exists -- reusing')
        return model_dir
    os.makedirs(model_dir, exist_ok=True)

    train_entities, val_entities = [], []
    for entity in pool:
        train_entity, val_entity = dg.load_single_entity_train_val(DATASET, entity)
        train_entities.append(train_entity)
        val_entities.append(val_entity)
    train_dataset = Dataset(entities=train_entities, name=f'without_{held_out}-train')
    val_dataset = Dataset(entities=val_entities, name=f'without_{held_out}-val')
    train_dl = dg.wrap_loader(train_dataset, shuffle=True)
    val_dl = dg.wrap_loader(val_dataset, shuffle=True)
    print(f'Training without_{held_out}: pool={pool} '
          f'({len(train_dl)} train windows / {len(val_dl)} val windows)')

    model_args = ci.build_model_args(dg.CFG, dg.WINDOW_SIZE)
    params = utils.AttrDict(batch_size=dg.BATCH_SIZE, lr=0.001, epoch=100, max_grad_norm=1.0, seed=seed)
    params.override(main.model_parameters(model_args))

    model = main.REDLAMP(model_dir=model_dir, params=params, device=device)
    model.train(train_dl, val_dl)
    return model_dir


def load_self_scores(ucr_xlsx, entities):
    df = pd.read_excel(ucr_xlsx, sheet_name='Per-Entity Comparison')
    df['entity'] = df['entity'].astype(str).str.zfill(3)
    return df[df['entity'].isin(entities)].set_index('entity')


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--entities', nargs='+', default=DEFAULT_ENTITIES)
    parser.add_argument('--run_name', default='test')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--ucr_xlsx', default='./result/Experiment_2/Results/ucr_results.xlsx')
    parser.add_argument('--out_csv', default='./result/DS_2/achievability/ucr_leave_one_out.csv')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    device = utils.init_dl_program(args.gpu, seed=args.seed)
    model_args = ci.build_model_args(dg.CFG, dg.WINDOW_SIZE)
    params = utils.AttrDict(seed=args.seed)
    params.override(main.model_parameters(model_args))

    self_scores = load_self_scores(args.ucr_xlsx, args.entities)
    missing_self = set(args.entities) - set(self_scores.index)
    if missing_self:
        print(f'[warn] no Self score found in {args.ucr_xlsx} for: {sorted(missing_self)}')

    rows = []
    for held_out in args.entities:
        model_dir = train_pooled_without(held_out, args.entities, args.run_name, args.seed, device, force=args.force)
        result = frm.score_entity(args.run_name, DATASET, held_out, args.seed, params, device, model_dir=model_dir)
        if result is None:
            print(f'[skip] {held_out}: LOO scoring failed/unavailable (no real anomaly in test labels, or length mismatch)')
            continue

        row = dict(entity=held_out)
        has_self = held_out in self_scores.index
        for m in frm.METRIC_KEYS:
            row[f'Self_{m}'] = self_scores.loc[held_out, f'{m}_self'] if has_self else None
            row[f'LOO_{m}'] = result['metrics'][m]
        row['Self_peak_in_range'] = self_scores.loc[held_out, 'peak_in_range_self'] if has_self else None
        row['LOO_peak_in_range'] = result['peak_in_range']
        rows.append(row)
        self_vus = row['Self_VUS_ROC']
        print(f'{held_out}: LOO VUS_ROC={result["metrics"]["VUS_ROC"]:.3f}'
              + (f' (Self={self_vus:.3f})' if self_vus is not None else ' (no Self score found)'))

    out_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)
    print(out_df.to_string(index=False))
    print(f'Wrote {args.out_csv}')


if __name__ == '__main__':
    run()
