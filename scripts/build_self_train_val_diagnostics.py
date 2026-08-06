"""
DS_3: Self train/val 4-panel diagnostic PDFs (see local_diagnostic_curves.py
for the template itself -- raw signal+reconstruction, MSE score, CE score,
Anomaly score, all four already-existing quantities from
main.anomaly_scoreing, just visualized over train/val splits instead of
only test).

Picks --n_entities UCR entities once (seeded) and reuses the SAME entities
across every anomaly type and both splits, for direct comparability. For
each (entity, split, type), builds a DENSE (window_step=1) Loader_aug
restricted to that single anomaly_type (so every window in it independently
gets that type injected -- there's no "clean vs injected" distinction to
manage here, matching how the model was actually trained/evaluated), runs
it through Self's own dedicated checkpoint, and displays one arbitrary
window (the middle of the split) plus 2*window_size of context on each
side.

Computation is grouped by entity: its own checkpoint is loaded once and its
(train_dataset, val_dataset) pair is loaded off disk once (load_data itself
already returns both in one call), then a fresh Loader_aug is built per
(split, type) from those already-loaded datasets -- injection genuinely
differs per type so that part can't be cached, but there's no reason to
re-read the entity's raw data or reload its model checkpoint 24 times
(12 types x 2 splits) over.

Consolidated into just 2 PDFs (Train, Val) with many pages/sections inside,
rather than one PDF per type -- see the plan's PDF-count discussion.

Curves are cached to .npz under --cache_dir so future diagnostics can reuse
them without re-running inference.
"""
import argparse
import os
import sys

import numpy as np
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
    parser.add_argument('--n_entities', type=int, default=5)
    parser.add_argument('--rng_seed', type=int, default=0)
    parser.add_argument('--out_dir', default='./result/DS_3/train_val_diagnostics')
    parser.add_argument('--cache_dir', default='./result/DS_3/curves_cache/self')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    device = torch.device('cpu')
    model_args = ci.build_model_args(dg.CFG, cps.WINDOW_SIZE)
    params = utils.AttrDict(seed=args.seed)
    params.override(main.model_parameters(model_args))

    all_entities = frm.discover_dataset_entities(args.run_name, DATASET)
    rng = np.random.default_rng(args.rng_seed)
    entities = sorted(rng.choice(all_entities, size=min(args.n_entities, len(all_entities)), replace=False).tolist())
    print(f'Selected {len(entities)} entities: {entities}')

    os.makedirs(args.out_dir, exist_ok=True)

    curves_by_key = {}  # (entity, split, type) -> curves dict
    window_size_by_entity = {}
    for entity in entities:
        try:
            model_dir, disk_cfg = ci.discover_entity(args.run_name, DATASET, entity, args.seed)
        except FileNotFoundError:
            print(f'[skip] {entity}: no trained model found')
            continue
        window_size_by_entity[entity] = disk_cfg['window_size']

        needed_types = [t for t in TYPES for split in ['train', 'val']
                         if args.force or not os.path.isfile(os.path.join(args.cache_dir, f'{entity}_{split}_{t}.npz'))]
        model = None
        train_dataset = val_dataset = None
        if needed_types:
            model = ldc.load_convaec_model(model_dir, params, device)
            train_dataset, val_dataset = load_entity_datasets(entity, disk_cfg)

        for split in ['train', 'val']:
            for anomaly_type in TYPES:
                cache_path = os.path.join(args.cache_dir, f'{entity}_{split}_{anomaly_type}.npz')

                def compute():
                    windows = windows_for_type(train_dataset, val_dataset, split, anomaly_type, disk_cfg)
                    return ldc.compute_dense_curves(windows, model, device)

                curves = ldc.get_or_compute_curves(cache_path, compute, force=args.force)
                curves_by_key[(entity, split, anomaly_type)] = curves
                print(f'  computed/cached: {entity} / {anomaly_type} / {split}')

    for split in ['train', 'val']:
        out_path = os.path.join(args.out_dir, f'Self_{split.capitalize()}_inference_samples.pdf')
        with PdfPages(out_path) as pdf:
            for anomaly_type in TYPES:
                for entity in entities:
                    key = (entity, split, anomaly_type)
                    if key not in curves_by_key:
                        continue
                    curves = curves_by_key[key]
                    if len(curves['raw_series']) == 0:
                        print(f'[skip] {entity}/{split}/{anomaly_type}: empty split')
                        continue

                    window_size = window_size_by_entity[entity]
                    focus_end = len(curves['raw_series']) // 2
                    focus_start, focus_end = ldc.window_bounds_from_end_index(focus_end, window_size)

                    ldc.plot_diagnostic_page(
                        pdf, curves['raw_series'],
                        [dict(label='self', reconstruction=curves['reconstruction'],
                              mse_score=curves['mse_score'], ce_score=curves['ce_score'], score=curves['score'])],
                        focus_start, focus_end, window_size,
                        title=f'Self | {entity} | {anomaly_type} | {split}')

        print(f'Wrote {out_path}')

    print('Done.')


if __name__ == '__main__':
    run()
