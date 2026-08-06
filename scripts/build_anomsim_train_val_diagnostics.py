"""
DS_3: Cross-AnomSim train/val 4-panel diagnostic PDFs -- same template as
build_self_train_val_diagnostics.py (see local_diagnostic_curves.py), just
sourced from AnomSim_v1 data (Core-Clustering repo, sibling to this one)
instead of UCR, scored with the Cross-AnomSim checkpoint (already-trained,
loaded via main.ConvAEC exactly like simulation_cross_domain_metrics.py
does) instead of a Self model.

Covers EVERY AnomSim_v1 entity (144, not a sample) -- one PDF PER ENTITY
per split (AnomSim_Train_{entity_dir}.pdf, AnomSim_Val_{entity_dir}.pdf;
entity_dir names already encode domain, e.g. "sine_b0", so no separate
domain folder is needed). Each PDF has 12 sections (Normal + 11 injected
types) x 5 pages, the 5 pages being 5 DIFFERENT display windows of the SAME
already-computed dense curve (see pick_sample_positions) -- dense
window_step=1 injection draws an independent random instance of that type
at every window position, so different positions genuinely show different
injected samples.

Train/val split per entity comes from Core-Clustering's own
load_single_entity_split (temporal 90/10 of that one entity's own
timeline, matching how Self models are evaluated per-entity) -- row 0 =
train, row 1 = val of the resulting 2-row BasePool.

Computation is grouped by (entity, split): each entity's Y.npy is loaded
and its OnlineWindowedDataset built ONCE per split (its `.index` already
covers every anomaly type at construction time), then filtered per type --
inference runs once per (entity, split, type) and is cached to .npz; the 5
pages per section are free reslicing of that same cached curve.

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
WINDOW_STEP = 1
N_SAMPLES = 5


def load_entity_dataset(single_entity_module, online_dataset_module, dataset_dir, entity_dir, split,
                         class_list, seed):
    """Loads one entity's Y.npy + builds its OnlineWindowedDataset ONCE
    (covers every anomaly type already, per its own __init__) -- callers
    filter `.index` by type_idx afterwards, no need to rebuild per type."""
    pool, split_result = single_entity_module.load_single_entity_split(dataset_dir, entity_dir)
    row_idx = int(split_result.train_idx[0] if split == 'train' else split_result.val_idx[0])
    return online_dataset_module.OnlineWindowedDataset(
        pool, np.array([row_idx]), WINDOW_SIZE, WINDOW_STEP, class_list, base_seed=seed)


def windows_for_type(dataset, anomaly_type, class_list):
    type_idx = class_list.index(anomaly_type)
    matching = [i for i, entry in enumerate(dataset.index) if entry[4] == type_idx]
    if not matching:
        return torch.empty(0, WINDOW_SIZE, 1)
    # __getitem__ already returns Y_t as (window_size, n_features) -- matches
    # main.ConvAEC.forward()'s expected input shape directly, no transpose needed.
    return torch.stack([dataset[i][0] for i in matching])


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
    from core_clustering.single_entity import list_entities
    from core_clustering.redlamp_compat import REDLAMP_ANOMALY_TYPES
    import core_clustering.single_entity as single_entity_module
    import core_clustering.online_dataset as online_dataset_module

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
        curves_by_split_type = {}
        for split in ['train', 'val']:
            needed_types = [t for t in REDLAMP_ANOMALY_TYPES
                             if args.force or not os.path.isfile(os.path.join(args.cache_dir, f'{entity_dir}_{split}_{t}.npz'))]
            dataset = None
            if needed_types:
                dataset = load_entity_dataset(single_entity_module, online_dataset_module,
                                               args.dataset_dir, entity_dir, split, REDLAMP_ANOMALY_TYPES, args.seed)
            for anomaly_type in REDLAMP_ANOMALY_TYPES:
                cache_path = os.path.join(args.cache_dir, f'{entity_dir}_{split}_{anomaly_type}.npz')

                def compute():
                    windows = windows_for_type(dataset, anomaly_type, REDLAMP_ANOMALY_TYPES)
                    return ldc.compute_dense_curves(windows, model, device)

                curves_by_split_type[(split, anomaly_type)] = ldc.get_or_compute_curves(cache_path, compute, force=args.force)

        for split in ['train', 'val']:
            out_path = os.path.join(args.out_dir, f'AnomSim_{split.capitalize()}_{entity_dir}.pdf')
            with PdfPages(out_path) as pdf:
                for anomaly_type in REDLAMP_ANOMALY_TYPES:
                    curves = curves_by_split_type[(split, anomaly_type)]
                    if len(curves['raw_series']) == 0:
                        print(f'[skip] {entity_dir}/{split}/{anomaly_type}: empty window set')
                        continue
                    positions = ldc.pick_sample_positions(len(curves['raw_series']), WINDOW_SIZE, n=N_SAMPLES)
                    for sample_i, center in enumerate(positions, start=1):
                        focus_start, focus_end = ldc.window_bounds_from_end_index(center, WINDOW_SIZE)
                        ldc.plot_diagnostic_page(
                            pdf, curves['raw_series'],
                            [dict(label='cross_anomsim', reconstruction=curves['reconstruction'],
                                  mse_score=curves['mse_score'], ce_score=curves['ce_score'], score=curves['score'])],
                            focus_start, focus_end, WINDOW_SIZE,
                            title=f'AnomSim | {entity_dir} | {anomaly_type} | {split} | sample {sample_i}/{N_SAMPLES}')
            print(f'Wrote {out_path}')

    print('Done.')


if __name__ == '__main__':
    run()
