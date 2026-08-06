"""
DS_3: Self train/val 4-panel diagnostic PDFs (see local_diagnostic_curves.py
for the template itself -- raw signal+reconstruction, MSE score, CE score,
Anomaly score, all four already-existing quantities from
main.anomaly_scoreing, just visualized over train/val splits instead of
only test).

Covers EVERY UCR entity discovered for --run_name (not a sample) -- one PDF
PER ENTITY per split (Self_Train_{entity}.pdf, Self_Val_{entity}.pdf), each
with 12 sections (Normal + 11 injected types) x 5 pages. The 5 pages per
section show 5 DIFFERENT display windows of the SAME already-computed
dense curve (see pick_sample_positions) -- dense window_step=1 injection
draws an independent random instance of that type at every window
position, so different positions genuinely show different injected
samples, not just different crops of one instance. Inference itself is
only ever run once per (entity, split, type) and cached to .npz; the 5
pages are free reslicing of that same cached curve.

--shard_index/--num_shards split the ENTITY list across concurrent
processes (see run_self_train_val_diagnostics_parallel.py) -- since output
is one file per entity, shards never write the same file and need no
merge step.
"""
import argparse
import os
import sys

import torch
from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import main
import utils
import cross_inference as ci
import domain_generalization as dg
import continuous_pool_scaling as cps
import full_reproduction_metrics as frm
from loaders.load import load_data
from loaders.loader_aug import Loader_aug
import local_diagnostic_curves as ldc

DATASET = 'anomaly_archive'
TYPES = ci.ANOMALY_TYPES  # ['normal', 'spike', 'flip', ...] -- 12 items, index 0 = normal
N_SAMPLES = 5


def load_entity_datasets(entity, disk_cfg):
    """(train_dataset, val_dataset) loaded once per entity -- callers build
    a fresh Loader_aug per (split, type) from these (injection differs by
    type and can't be cached, but there's no need to re-read the entity's
    raw data off disk for every type)."""
    return load_data(dataset=DATASET, group='train', entities=entity, downsampling=disk_cfg['downsampling'],
                      min_length=None, root_dir='./dataset', verbose=False, validation=True)


def windows_for_type(train_dataset, val_dataset, split, anomaly_type, disk_cfg):
    dataset = train_dataset if split == 'train' else val_dataset
    loader = Loader_aug(
        dataset=dataset, batch_size=disk_cfg['batch_size'], window_size=disk_cfg['window_size'],
        window_step=1, anomaly_types=[anomaly_type], anomaly_types_for_dict=ci.ANOMALY_TYPES,
        min_range=1, min_features=dg.CFG['min_features'], max_features=dg.CFG['max_features'],
        fast_sampling=False, shuffle=False, verbose=False)
    # (n_windows, n_features, window_size) -> (n_windows, window_size, n_features), matching
    # main.test()'s own transpose convention for batch['Y'].
    return loader.Y_windows.transpose(2, 1)


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', default='test')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out_dir', default='./result/DS_3/train_val_diagnostics/self')
    parser.add_argument('--cache_dir', default='./result/DS_3/curves_cache/self')
    parser.add_argument('--shard_index', type=int, default=0)
    parser.add_argument('--num_shards', type=int, default=1)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    device = torch.device('cpu')
    model_args = ci.build_model_args(dg.CFG, cps.WINDOW_SIZE)
    params = utils.AttrDict(seed=args.seed)
    params.override(main.model_parameters(model_args))

    entities = frm.discover_dataset_entities(args.run_name, DATASET)
    if args.num_shards > 1:
        entities = [e for i, e in enumerate(entities) if i % args.num_shards == args.shard_index]
    print(f'{len(entities)} entities to process (shard {args.shard_index}/{args.num_shards})')

    os.makedirs(args.out_dir, exist_ok=True)

    for entity in entities:
        try:
            model_dir, disk_cfg = ci.discover_entity(args.run_name, DATASET, entity, args.seed)
        except FileNotFoundError:
            print(f'[skip] {entity}: no trained model found')
            continue
        window_size = disk_cfg['window_size']

        needed_types = [t for t in TYPES for split in ['train', 'val']
                         if args.force or not os.path.isfile(os.path.join(args.cache_dir, f'{entity}_{split}_{t}.npz'))]
        model = None
        train_dataset = val_dataset = None
        if needed_types:
            model = ldc.load_convaec_model(model_dir, params, device)
            train_dataset, val_dataset = load_entity_datasets(entity, disk_cfg)

        curves_by_split_type = {}
        for split in ['train', 'val']:
            for anomaly_type in TYPES:
                cache_path = os.path.join(args.cache_dir, f'{entity}_{split}_{anomaly_type}.npz')

                def compute():
                    windows = windows_for_type(train_dataset, val_dataset, split, anomaly_type, disk_cfg)
                    return ldc.compute_dense_curves(windows, model, device)

                curves_by_split_type[(split, anomaly_type)] = ldc.get_or_compute_curves(cache_path, compute, force=args.force)

        for split in ['train', 'val']:
            out_path = os.path.join(args.out_dir, f'Self_{split.capitalize()}_{entity}.pdf')
            with PdfPages(out_path) as pdf:
                for anomaly_type in TYPES:
                    curves = curves_by_split_type[(split, anomaly_type)]
                    if len(curves['raw_series']) == 0:
                        print(f'[skip] {entity}/{split}/{anomaly_type}: empty split')
                        continue
                    positions = ldc.pick_sample_positions(len(curves['raw_series']), window_size, n=N_SAMPLES)
                    for sample_i, center in enumerate(positions, start=1):
                        focus_start, focus_end = ldc.window_bounds_from_end_index(center, window_size)
                        ldc.plot_diagnostic_page(
                            pdf, curves['raw_series'],
                            [dict(label='self', reconstruction=curves['reconstruction'],
                                  mse_score=curves['mse_score'], ce_score=curves['ce_score'], score=curves['score'],
                                  mse_raw=curves['mse_raw'])],
                            focus_start, focus_end, window_size,
                            title=f'Self | {entity} | {anomaly_type} | {split} | sample {sample_i}/{N_SAMPLES}')
            print(f'Wrote {out_path}')

    print('Done.')


if __name__ == '__main__':
    run()
