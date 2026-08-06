"""
DS_3: Cross-AnomSim train/val 4-panel diagnostic PDFs -- same template as
build_self_train_val_diagnostics.py (see local_diagnostic_curves.py and that
script's own docstring for the full rationale), just sourced from AnomSim_v1
data (Core-Clustering repo, sibling to this one) instead of UCR, scored with
the Cross-AnomSim checkpoint (already-trained, loaded via main.ConvAEC
exactly like simulation_cross_domain_metrics.py does) instead of a Self
model.

Each of the N_SAMPLES pages per (entity, split, type) is its own small,
self-contained local experiment: pick ONE focus window position on the
entity's real timeline, build a chunk of that window plus 2*window_size of
REAL (untouched) context on each side, inject ONE fresh random instance of
the anomaly type into ONLY the focus sub-range, then run a dense
window_step=1 pass over just that small chunk (see
local_diagnostic_curves.build_local_chunk/dense_windows_from_chunk).
Injection reuses AnomSim's own anomsim.anomalies.base.get_anomaly registry
(the same per-type Anomaly classes OnlineWindowedDataset.__getitem__ calls),
via a fresh unseeded np.random.default_rng() per call for genuine sample-to-
sample randomness -- OnlineWindowedDataset itself seeds deterministically
off (base_seed, row, window, type) for training reproducibility, which isn't
what's wanted here.

Covers EVERY AnomSim_v1 entity (144, not a sample) -- one PDF PER ENTITY
per split (AnomSim_Train_{entity_dir}.pdf, AnomSim_Val_{entity_dir}.pdf;
entity_dir names already encode domain, e.g. "sine_b0"). Each PDF has 12
sections (Normal + 11 injected types) x N_SAMPLES pages, each (entity,
split, type, sample) combo cached to its own small .npz.

Train/val split per entity comes from Core-Clustering's own
load_single_entity_split (temporal 90/10 of that one entity's own
timeline, matching how Self models are evaluated per-entity) -- row 0 =
train, row 1 = val of the resulting 2-row BasePool.

--shard_index/--num_shards split the ENTITY list across concurrent
processes (see run_anomsim_train_val_diagnostics_parallel.py) -- one file
per entity means shards never collide and need no merge step.
"""
import argparse
import os
import sys

import numpy as np
import torch
from matplotlib.backends.backend_pdf import PdfPages

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
CORE_CLUSTERING_DEFAULT = os.path.join(REPO_ROOT, '..', 'Core-Clustering')

import main
import utils
import cross_inference as ci
import domain_generalization as dg
import local_diagnostic_curves as ldc

WINDOW_SIZE = 100
N_SAMPLES = 5


def inject_fn_for(get_anomaly, anomaly_type):
    """build_local_chunk's inject_fn(window) -> (injected, mask) contract,
    bound to one anomaly type, backed by AnomSim's own Anomaly.apply(). A
    fresh rng per call (not seeded off position, unlike OnlineWindowedDataset)
    so each of the N_SAMPLES pages gets a genuinely independent instance."""
    def inject(window):
        rng = np.random.default_rng()
        y, _z, mask = get_anomaly(anomaly_type)().apply(window, rng)
        return np.asarray(y), np.asarray(mask)
    return inject


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', default=os.path.join(REPO_ROOT, '..', 'AnomSim', 'data', 'AnomSim_v1'))
    parser.add_argument('--core_clustering_dir', default=CORE_CLUSTERING_DEFAULT)
    parser.add_argument('--cross_anomsim_model_dir', default='./result/Experiment_1/Models/Cross-AnomSim/0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out_dir', default='./result/DS_3/train_val_diagnostics/anomsim')
    parser.add_argument('--cache_dir', default='./result/DS_3/curves_cache/anomsim')
    parser.add_argument('--shard_index', type=int, default=0)
    parser.add_argument('--num_shards', type=int, default=1)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    sys.path.insert(0, args.core_clustering_dir)
    sys.path.insert(0, os.path.join(args.core_clustering_dir, '..', 'AnomSim'))
    from core_clustering.single_entity import list_entities, load_single_entity_split
    from core_clustering.redlamp_compat import REDLAMP_ANOMALY_TYPES
    from anomsim.anomalies.base import get_anomaly

    device = torch.device('cpu')
    model_args = ci.build_model_args(dg.CFG, WINDOW_SIZE)
    params = utils.AttrDict(seed=args.seed)
    params.override(main.model_parameters(model_args))
    model = ldc.load_convaec_model(args.cross_anomsim_model_dir, params, device)

    entities = list_entities(args.dataset_dir)
    if args.num_shards > 1:
        entities = [e for i, e in enumerate(entities) if i % args.num_shards == args.shard_index]
    print(f'{len(entities)} entities to process (shard {args.shard_index}/{args.num_shards})')

    os.makedirs(args.out_dir, exist_ok=True)

    for entity_dir in entities:
        cache_path = lambda split, t, i: os.path.join(args.cache_dir, f'{entity_dir}_{split}_{t}_s{i}.npz')
        any_missing = args.force or any(
            not os.path.isfile(cache_path(split, t, i))
            for split in ('train', 'val') for t in REDLAMP_ANOMALY_TYPES for i in range(1, N_SAMPLES + 1))

        pool = split_result = None
        if any_missing:
            pool, split_result = load_single_entity_split(args.dataset_dir, entity_dir)

        curves_by_key = {}
        for split in ('train', 'val'):
            positions = Y = None
            if pool is not None:
                row_idx = int(split_result.train_idx[0] if split == 'train' else split_result.val_idx[0])
                n_time = int(pool.n_time[row_idx])
                if n_time < WINDOW_SIZE:
                    print(f'[skip] {entity_dir}/{split}: too-short split')
                    continue
                Y = pool.Y[row_idx]
                positions = ldc.pick_sample_positions(n_time, WINDOW_SIZE, n=N_SAMPLES)
            # else: nothing loaded because every (type, sample) cache file for this split
            # already existed -- get_or_compute_curves below will hit the cache directly.

            # Gather every (type, sample) combo that needs computing this run, build
            # ALL their local chunks first, then run ONE batched model forward pass
            # across the lot instead of a separate model() call per combo -- see
            # compute_dense_curves_batch; results are identical, just far less
            # per-call Python/dispatch overhead.
            to_compute = []
            for anomaly_type in REDLAMP_ANOMALY_TYPES:
                for sample_i in range(1, N_SAMPLES + 1):
                    path = cache_path(split, anomaly_type, sample_i)
                    if args.force or not os.path.isfile(path):
                        if positions is not None:
                            to_compute.append((anomaly_type, sample_i, path))
                    else:
                        curves_by_key[(split, anomaly_type, sample_i)] = ldc.load_curves_npz(path)

            if to_compute:
                windows_list, metas = [], []
                for anomaly_type, sample_i, _path in to_compute:
                    inject_fn = inject_fn_for(get_anomaly, anomaly_type)
                    focus_start, focus_end = ldc.window_bounds_from_end_index(positions[sample_i - 1], WINDOW_SIZE)
                    chunk, local_focus_start, spans = ldc.build_local_chunk(Y, focus_start, focus_end, inject_fn)
                    windows_list.append(ldc.dense_windows_from_chunk(chunk, WINDOW_SIZE))
                    metas.append((local_focus_start, spans))

                batch_results = ldc.compute_dense_curves_batch(windows_list, model, device)
                for (anomaly_type, sample_i, path), (local_focus_start, spans), curves in zip(
                        to_compute, metas, batch_results):
                    curves['local_focus_start'] = local_focus_start
                    curves['anomaly_spans'] = np.array(spans, dtype=int).reshape(-1, 2)
                    ldc.save_curves_npz(path, curves)
                    curves_by_key[(split, anomaly_type, sample_i)] = curves

        for split in ('train', 'val'):
            out_path = os.path.join(args.out_dir, f'AnomSim_{split.capitalize()}_{entity_dir}.pdf')
            with PdfPages(out_path) as pdf:
                for anomaly_type in REDLAMP_ANOMALY_TYPES:
                    for sample_i in range(1, N_SAMPLES + 1):
                        curves = curves_by_key.get((split, anomaly_type, sample_i))
                        if curves is None:
                            continue
                        local_focus_start = int(curves['local_focus_start'])
                        spans = [tuple(row) for row in curves['anomaly_spans']]
                        ldc.plot_diagnostic_page(
                            pdf, curves['raw_series'],
                            [dict(label='cross_anomsim', reconstruction=curves['reconstruction'],
                                  mse_score=curves['mse_score'], ce_score=curves['ce_score'], score=curves['score'],
                                  mse_raw=curves['mse_raw'])],
                            local_focus_start, local_focus_start + WINDOW_SIZE, WINDOW_SIZE,
                            title=f'AnomSim | {entity_dir} | {anomaly_type} | {split} | sample {sample_i}/{N_SAMPLES}',
                            real_anomaly_spans=spans)
            print(f'Wrote {out_path}')

    print('Done.')


if __name__ == '__main__':
    run()
