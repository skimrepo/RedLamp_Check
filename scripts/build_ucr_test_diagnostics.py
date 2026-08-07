"""
DS_3: UCR TEST set diagnostic PDF -- Self and Cross-AnomSim overlaid on the
same panels (see local_diagnostic_curves.py for the shared template). Both
are already-trained checkpoints (Self's own dedicated model via
cross_inference.discover_entity, Cross-AnomSim's pooled AnomSim_v1
checkpoint) -- no retraining anywhere.

One page PER ENTITY, covering the WHOLE test split (not a local zoom around
a handful of picked positions, unlike the earlier design in this file's
history) -- reconstruction/mse_score/ce_score/score are taken directly from
full_reproduction_metrics.score_entity(..., include_curves=True)'s own
whole-split dense window_step=1 pass, exactly matching the official
RedLamp/TSB_UAD evaluation convention: each window's scalar score is
assigned to that window's LAST timestep only (no averaging across the
window_size overlapping windows that also cover a given timepoint -- see
datautils.py's group='test_all', main.py's anomaly_scoreing/
convolve_minmax_score), and mse_score/ce_score/score's smoothing+[0,1]
normalization spans the ENTIRE split, not a local crop. This makes the
plotted scores the actual published-metric values (VUS_ROC/RF etc. are
computed from this exact same array), not a locally-renormalized
approximation.

The final Anomaly_Norm_Smooth panel also draws each model's TSB_UAD RF
threshold (mean(score) + 3*std(score), computed over that model's own
whole-split score array -- TSB_UAD/vus/basic_metrics.py's own convention,
recomputed fresh per entity/model like TSB_UAD does internally).

Real ground-truth anomaly segments are shaded on every panel. No
argmax/argmin position-picking or multi-page-per-position merging --
removed since every panel is now whole-split already, so different
positions would have looked identical anyway.

Runs on CPU (matching build_self_train_val_diagnostics.py and
build_anomsim_train_val_diagnostics.py) rather than GPU -- ConvAEC
inference here is lightweight, and launching --num_shards concurrent
processes that each grab a CUDA context on a GPU shared with other jobs
reliably OOMs once num_shards gets past a handful (each process's context
alone can be several hundred MB, on top of whatever memory other jobs on
the box are already holding).

Writes one PDF PER ENTITY (UCR_Test_{entity}.pdf under --out_dir) -- makes
it fast to jump straight to a specific entity that looked bad without
having to search through a big multi-entity file. --shard_index/--num_shards
split the ENTITY list across concurrent processes (see
run_ucr_test_diagnostics_parallel.py); since each entity's file is
independent, shards never collide and no merge step is needed.

Cache format (result/DS_3/curves_cache/test/{entity}_{self,cross_anomsim}.npz)
is unchanged from before -- existing cached curves are reused as-is.
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


def rf_threshold(score):
    """TSB_UAD/vus/basic_metrics.py's own RF threshold convention
    (score > mean(score) + 3*std(score)), recomputed here for display since
    TSB_UAD doesn't expose it as a standalone function -- matches exactly
    what get_metrics() computes internally from this same score array."""
    return float(score.mean() + 3 * score.std())


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', default='test')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--cross_anomsim_model_dir', default=None,
                         help='Defaults to ./result/Experiment_1/Models/Cross-AnomSim/{seed}')
    parser.add_argument('--out_dir', default='./result/DS_3/test_diagnostics')
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

    entities = frm.discover_dataset_entities(args.run_name, DATASET)
    if args.num_shards > 1:
        entities = [e for i, e in enumerate(entities) if i % args.num_shards == args.shard_index]
    print(f'{len(entities)} entities to process (shard {args.shard_index}/{args.num_shards})')

    os.makedirs(args.out_dir, exist_ok=True)

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
        total_len = len(self_curves['raw_series'])

        out_path = os.path.join(args.out_dir, f'UCR_Test_{entity}.pdf')
        with PdfPages(out_path) as pdf:
            ldc.plot_diagnostic_page(
                pdf, self_curves['raw_series'],
                [dict(label='self', reconstruction=self_curves['reconstruction'],
                      mse_score=self_curves['mse_score'], ce_score=self_curves['ce_score'],
                      score=self_curves['score'], mse_raw=self_curves['mse_raw'],
                      threshold=rf_threshold(self_curves['score'])),
                 dict(label='cross_anomsim', reconstruction=cross_curves['reconstruction'],
                      mse_score=cross_curves['mse_score'], ce_score=cross_curves['ce_score'],
                      score=cross_curves['score'], mse_raw=cross_curves['mse_raw'],
                      threshold=rf_threshold(cross_curves['score']))],
                focus_start=0, focus_end=total_len, window_size=window_size,
                title=entity, real_anomaly_spans=real_segments)
        print(f'  Wrote {out_path}')

    print('Done.')


if __name__ == '__main__':
    run()
