"""
DS_3: UCR TEST set 4-panel diagnostic PDF -- Self and Cross-AnomSim
overlaid on the same panels (see local_diagnostic_curves.py for the shared
template). Both are already-trained checkpoints (Self's own dedicated
model via cross_inference.discover_entity, Cross-AnomSim's pooled
AnomSim_v1 checkpoint) -- no retraining anywhere.

For every UCR entity, picks positions to diagnose:
  - every real ground-truth anomaly segment (up to 5, longest first --
    almost always exactly 1 for UCR's benchmark convention)
  - Self's mse_score/ce_score argmax and argmin (4 positions)
  - Cross-AnomSim's mse_score/ce_score argmax and argmin (4 positions)
so usually 1+4+4=9 positions/entity. Positions whose windows would be
within window_size of each other are merged into one page (title lists
every matching criterion) rather than shown as near-duplicates -- e.g. if
Self's MSE argmax coincides with the real anomaly, that's ONE page titled
accordingly, not two nearly-identical ones. Every real anomaly segment is
shaded on every page for that entity, regardless of which position the page
is centered on, so it's visually obvious whether a given criterion's pick
landed on/near the real anomaly or somewhere else entirely.

Position SELECTION still reuses full_reproduction_metrics.score_entity(...,
include_curves=True) unchanged (the whole-split dense pass, cached to
.npz) -- finding argmax/argmin/real segments genuinely needs the whole
split's own view. But RENDERING each selected position now recomputes
locally, matching build_self_train_val_diagnostics.py/
build_anomsim_train_val_diagnostics.py: a small chunk of real signal
(2*window_size context on each side of the focus window, no injection --
this is real ground-truth data) is re-run through both models via
local_diagnostic_curves.build_local_chunk/dense_windows_from_chunk/
compute_dense_curves, so mse_score/ce_score/score's [0,1] normalization is
local to what's shown on the page instead of spanning the entire test
split (same reasoning as the train/val redesign -- treat each displayed
focus window + its context as its own small standalone series).

Runs on CPU (matching build_self_train_val_diagnostics.py and
build_anomsim_train_val_diagnostics.py) rather than GPU -- ConvAEC
inference here is lightweight, and launching --num_shards concurrent
processes that each grab a CUDA context on a GPU shared with other jobs
reliably OOMs once num_shards gets past a handful (each process's context
alone can be several hundred MB, on top of whatever memory other jobs on
the box are already holding).

Supports --shard_index/--num_shards (writes one PDF per shard,
"..._shard{i}.pdf") for splitting the ~250-entity sweep across concurrent
processes if a single-process run turns out too slow in practice.
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
import local_diagnostic_curves as ldc

DATASET = 'anomaly_archive'
CURVE_KEYS = ['raw_series', 'reconstruction', 'mse_score', 'ce_score', 'score', 'real_labels', 'mse_raw']


def get_curves_cached(cache_path, compute_fn, force=False):
    """Like ldc.get_or_compute_curves, but tolerates compute_fn returning
    None (score_entity does this for entities it can't score -- missing
    model, length mismatch, or no real anomaly in the test labels) without
    trying to np.savez a None."""
    if not force:
        cached = ldc.load_curves_npz(cache_path)
        if cached is not None:
            return cached
    curves = compute_fn()
    if curves is not None:
        ldc.save_curves_npz(cache_path, curves)
    return curves


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', default='test')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--cross_anomsim_model_dir', default=None,
                         help='Defaults to ./result/Experiment_1/Models/Cross-AnomSim/{seed}')
    parser.add_argument('--out_pdf', default='./result/DS_3/test_diagnostics/UCR_Test_anomaly_inference_samples.pdf')
    parser.add_argument('--cache_dir', default='./result/DS_3/curves_cache/test')
    parser.add_argument('--shard_index', type=int, default=0)
    parser.add_argument('--num_shards', type=int, default=1)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    cross_anomsim_model_dir = args.cross_anomsim_model_dir or f'./result/Experiment_1/Models/Cross-AnomSim/{args.seed}'
    device = torch.device('cpu')
    model_args = ci.build_model_args(dg.CFG, cps.WINDOW_SIZE)
    params = utils.AttrDict(seed=args.seed)
    params.override(main.model_parameters(model_args))
    cross_model = ldc.load_convaec_model(cross_anomsim_model_dir, params, device)

    entities = frm.discover_dataset_entities(args.run_name, DATASET)
    if args.num_shards > 1:
        entities = [e for i, e in enumerate(entities) if i % args.num_shards == args.shard_index]
    print(f'{len(entities)} entities to process (shard {args.shard_index}/{args.num_shards})')

    out_path = args.out_pdf
    if args.num_shards > 1:
        base, ext = os.path.splitext(args.out_pdf)
        out_path = f'{base}_shard{args.shard_index}{ext}'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with PdfPages(out_path) as pdf:
        for entity in entities:
            try:
                self_model_dir, disk_cfg = ci.discover_entity(args.run_name, DATASET, entity, args.seed)
            except FileNotFoundError:
                print(f'[skip] {entity}: no self model found')
                continue
            window_size = disk_cfg['window_size']

            def compute_self():
                result = frm.score_entity(args.run_name, DATASET, entity, args.seed, params, device,
                                           model_dir=self_model_dir, include_curves=True)
                return {k: result[k] for k in CURVE_KEYS} if result is not None else None

            def compute_cross():
                result = frm.score_entity(args.run_name, DATASET, entity, args.seed, params, device,
                                           model_dir=cross_anomsim_model_dir, include_curves=True)
                return {k: result[k] for k in CURVE_KEYS} if result is not None else None

            self_curves = get_curves_cached(os.path.join(args.cache_dir, f'{entity}_self.npz'), compute_self, args.force)
            if self_curves is None:
                print(f'[skip] {entity}: Self scoring failed/unavailable')
                continue
            cross_curves = get_curves_cached(os.path.join(args.cache_dir, f'{entity}_cross_anomsim.npz'), compute_cross, args.force)
            if cross_curves is None:
                print(f'[skip] {entity}: Cross-AnomSim scoring failed/unavailable')
                continue

            real_segments = ldc.find_anomaly_segments(self_curves['real_labels'], max_segments=5)
            self_extremes = ldc.pick_extreme_positions(self_curves, window_size)
            cross_extremes = ldc.pick_extreme_positions(cross_curves, window_size)

            labeled_positions = [((s + e) // 2, 'real anomaly') for s, e in real_segments]
            labeled_positions += [(idx, f'Self {name}') for name, idx in self_extremes.items()]
            labeled_positions += [(idx, f'Cross-AnomSim {name}') for name, idx in cross_extremes.items()]
            merged = ldc.merge_nearby_positions(labeled_positions, window_size)

            self_model = ldc.load_convaec_model(self_model_dir, params, device)
            Y = self_curves['raw_series'][None, :]  # (1, n_time) -- real test signal, same for both models
            real_labels = self_curves['real_labels']

            for center, labels in merged:
                focus_start = center - window_size // 2
                focus_end = focus_start + window_size
                chunk, local_focus_start, _ = ldc.build_local_chunk(Y, focus_start, focus_end, inject_fn=None)
                chunk_start = focus_start - local_focus_start
                chunk_end = chunk_start + chunk.shape[1]
                local_real_segments = ldc.find_anomaly_segments(real_labels[chunk_start:chunk_end], max_segments=5)

                windows = ldc.dense_windows_from_chunk(chunk, window_size)
                self_local = ldc.compute_dense_curves(windows, self_model, device)
                cross_local = ldc.compute_dense_curves(windows, cross_model, device)

                ldc.plot_diagnostic_page(
                    pdf, self_local['raw_series'],
                    [dict(label='self', reconstruction=self_local['reconstruction'],
                          mse_score=self_local['mse_score'], ce_score=self_local['ce_score'], score=self_local['score'],
                          mse_raw=self_local['mse_raw']),
                     dict(label='cross_anomsim', reconstruction=cross_local['reconstruction'],
                          mse_score=cross_local['mse_score'], ce_score=cross_local['ce_score'], score=cross_local['score'],
                          mse_raw=cross_local['mse_raw'])],
                    local_focus_start, local_focus_start + window_size, window_size,
                    title=f'{entity} | ' + ', '.join(labels),
                    real_anomaly_spans=local_real_segments)
            print(f'  {entity}: {len(merged)} page(s)')

    print(f'Wrote {out_path}')


if __name__ == '__main__':
    run()
