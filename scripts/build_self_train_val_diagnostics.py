"""
DS_3: Self train/val 4-panel diagnostic PDFs (see local_diagnostic_curves.py
for the template itself -- raw signal+reconstruction, MSE score, CE score,
Anomaly score, all four already-existing quantities from
main.anomaly_scoreing, just visualized over train/val splits instead of
only test).

Each of the N_SAMPLES pages per (entity, split, type) is its own small,
self-contained local experiment: pick ONE focus window position on the
entity's real timeline, build a chunk of that window plus 2*window_size of
REAL (untouched) context on each side, inject ONE fresh random instance of
the anomaly type into ONLY the focus sub-range (context stays real data),
then run a dense window_step=1 pass over just that small chunk and read off
reconstruction/mse_score/ce_score/score exactly as full_reproduction_metrics
always has (last-timestep-of-window alignment) -- see
local_diagnostic_curves.build_local_chunk/dense_windows_from_chunk. This
replaces the earlier design (independently re-injecting into EVERY window of
a whole-split dense pass), which made every window in view carry its own
anomaly and made mse_score/ce_score/score's [0,1] normalization span the
entire split rather than just what's shown on the page.

Injection reuses loaders.loader_aug.Loader_aug.select_anomalies (the exact
per-type logic used everywhere else) via a lightweight instance that skips
its normal __init__ (which would eagerly dense-inject the whole entity --
exactly what's being avoided here).

Covers EVERY UCR entity discovered for --run_name (not a sample) -- one PDF
PER ENTITY per split (Self_Train_{entity}.pdf, Self_Val_{entity}.pdf), each
with 12 sections (Normal + 11 injected types) x N_SAMPLES pages. Each
(entity, split, type, sample) combo is cached to its own .npz (small: only
~5*window_size points of inference each).

--shard_index/--num_shards split the ENTITY list across concurrent
processes (see run_self_train_val_diagnostics_parallel.py) -- since output
is one file per entity, shards never write the same file and need no
merge step.
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
N_SAMPLES = 5


def load_entity_datasets(entity, disk_cfg):
    """(train_dataset, val_dataset) loaded once per entity -- the real Y
    array each split's window positions/injections are drawn from."""
    return load_data(dataset=DATASET, group='train', entities=entity, downsampling=disk_cfg['downsampling'],
                      min_length=None, root_dir='./dataset', verbose=False, validation=True)


def make_injector(window_size):
    """A Loader_aug instance with __init__ bypassed (via __new__) -- its own
    __init__ would eagerly dense-inject anomalies into every window of the
    whole entity, which is exactly the whole-split behavior this script no
    longer wants. select_anomalies/_inject_* only ever read window_size,
    min_range, min_features, max_features, fast_sampling off self, so
    setting just those by hand is enough to reuse the exact same per-type
    injection code for a single, arbitrary window."""
    loader = Loader_aug.__new__(Loader_aug)
    loader.window_size = window_size
    loader.min_range = 1
    loader.min_features = dg.CFG['min_features']
    loader.max_features = dg.CFG['max_features']
    loader.fast_sampling = False
    return loader


def inject_fn_for(injector, anomaly_type):
    """build_local_chunk's inject_fn(window) -> (injected, mask) contract,
    bound to one anomaly type. 'normal' works through the same call --
    select_anomalies returns the window unmodified with an all-ones mask, so
    no special-casing is needed."""
    def inject(window):
        y_temp, _z_temp, mask_temp = injector.select_anomalies(anomaly_type, window, 0, window.shape[1])
        return np.asarray(y_temp), np.asarray(mask_temp)
    return inject


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
        injector = make_injector(window_size)

        cache_path = lambda split, t, i: os.path.join(args.cache_dir, f'{entity}_{split}_{t}_s{i}.npz')
        any_missing = args.force or any(
            not os.path.isfile(cache_path(split, t, i))
            for split in ('train', 'val') for t in TYPES for i in range(1, N_SAMPLES + 1))

        model = None
        train_dataset = val_dataset = None
        if any_missing:
            model = ldc.load_convaec_model(model_dir, params, device)
            train_dataset, val_dataset = load_entity_datasets(entity, disk_cfg)

        curves_by_key = {}
        for split in ('train', 'val'):
            dataset = train_dataset if split == 'train' else val_dataset
            positions = Y = None
            if dataset is not None:
                if len(dataset.entities) == 0 or dataset.entities[0].n_time < window_size:
                    print(f'[skip] {entity}/{split}: empty/too-short split')
                    continue  # nothing cached under this run's dataset could be salvaged either
                entity_obj = dataset.entities[0]
                Y, n_time = entity_obj.Y, entity_obj.n_time
                positions = ldc.pick_sample_positions(n_time, window_size, n=N_SAMPLES)
            # else: dataset wasn't loaded because every (type, sample) cache file for this
            # split already existed (any_missing was False) -- positions/Y stay None, but
            # get_or_compute_curves below will hit the cache and never call compute().

            # Gather every (type, sample) combo that needs computing this run, build
            # ALL their local chunks first, then run ONE batched model forward pass
            # across the lot (up to 12*N_SAMPLES chunks) instead of a separate model()
            # call per combo -- anomaly_scoreing/mse still run per-chunk afterward
            # (see compute_dense_curves_batch), so results are identical either way,
            # just with far less per-call Python/dispatch overhead.
            to_compute = []
            for anomaly_type in TYPES:
                for sample_i in range(1, N_SAMPLES + 1):
                    path = cache_path(split, anomaly_type, sample_i)
                    if args.force or not os.path.isfile(path):
                        if positions is not None:
                            to_compute.append((anomaly_type, sample_i, path))
                        # else: not cached and no data to compute from -- stays unavailable
                    else:
                        curves_by_key[(split, anomaly_type, sample_i)] = ldc.load_curves_npz(path)

            if to_compute:
                windows_list, metas = [], []
                for anomaly_type, sample_i, _path in to_compute:
                    inject_fn = inject_fn_for(injector, anomaly_type)
                    focus_start, focus_end = ldc.window_bounds_from_end_index(positions[sample_i - 1], window_size)
                    chunk, local_focus_start, spans = ldc.build_local_chunk(Y, focus_start, focus_end, inject_fn)
                    windows_list.append(ldc.dense_windows_from_chunk(chunk, window_size))
                    metas.append((local_focus_start, spans))

                batch_results = ldc.compute_dense_curves_batch(windows_list, model, device)
                for (anomaly_type, sample_i, path), (local_focus_start, spans), curves in zip(
                        to_compute, metas, batch_results):
                    curves['local_focus_start'] = local_focus_start
                    curves['anomaly_spans'] = np.array(spans, dtype=int).reshape(-1, 2)
                    ldc.save_curves_npz(path, curves)
                    curves_by_key[(split, anomaly_type, sample_i)] = curves

        for split in ('train', 'val'):
            out_path = os.path.join(args.out_dir, f'Self_{split.capitalize()}_{entity}.pdf')
            with PdfPages(out_path) as pdf:
                for anomaly_type in TYPES:
                    for sample_i in range(1, N_SAMPLES + 1):
                        curves = curves_by_key.get((split, anomaly_type, sample_i))
                        if curves is None:
                            continue
                        local_focus_start = int(curves['local_focus_start'])
                        spans = [tuple(row) for row in curves['anomaly_spans']]
                        ldc.plot_diagnostic_page(
                            pdf, curves['raw_series'],
                            [dict(label='self', reconstruction=curves['reconstruction'],
                                  mse_score=curves['mse_score'], ce_score=curves['ce_score'], score=curves['score'],
                                  mse_raw=curves['mse_raw'])],
                            local_focus_start, local_focus_start + window_size, window_size,
                            title=f'Self | {entity} | {anomaly_type} | {split} | sample {sample_i}/{N_SAMPLES}',
                            real_anomaly_spans=spans)
            print(f'Wrote {out_path}')

    print('Done.')


if __name__ == '__main__':
    run()
