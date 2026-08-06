"""
DS_3: Cross-AnomSim train/val 4-panel diagnostic PDFs -- same template as
build_self_train_val_diagnostics.py (see local_diagnostic_curves.py), just
sourced from AnomSim_v1 data (Core-Clustering repo, sibling to this one)
instead of UCR, scored with the Cross-AnomSim checkpoint (already-trained,
loaded via main.ConvAEC exactly like simulation_cross_domain_metrics.py
does) instead of a Self model.

For each (domain, anomaly_type), picks 5 DIFFERENT entities of that domain
(AnomSim_v1 has 16 entities/domain, always >=5) -- entities, not repeated
windows within one entity, since domain-level diversity is what this is
diagnosing. The one-off "Normal" section instead pools 5 random entities
across the whole dataset (no domain axis).

Train/val split per entity comes from Core-Clustering's own
load_single_entity_split (temporal 90/10 of that one entity's own
timeline, matching how Self models are evaluated per-entity) -- row 0 =
train, row 1 = val of the resulting 2-row BasePool.

Computation is grouped by (entity, split): each entity's Y.npy is loaded
and its OnlineWindowedDataset built ONCE per split (its `.index` already
covers every anomaly type at construction time -- see
core_clustering/online_dataset.py), then filtered per type -- NOT once per
(entity, split, type), which would reload the same entity's data off disk
needlessly for every one of the 11 non-normal types it appears under.

Consolidated into 2 PDFs (Train, Val) with all domain x type sections
inside, rather than one file per (domain, type) -- see the plan's PDF-count
discussion. Curves cached to .npz under --cache_dir, independent of this
run's in-memory grouping, so a rerun (or a future different script) can
reuse them entity-by-entity too.
"""
import argparse
import json
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


def entities_by_domain(dataset_dir, manifest_name='_manifest.jsonl'):
    by_domain = {}
    with open(os.path.join(dataset_dir, manifest_name)) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            meta = json.loads(line)
            by_domain.setdefault(meta['type'], []).append(meta['entity_dir'])
    return by_domain


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
    parser.add_argument('--n_entities', type=int, default=5)
    parser.add_argument('--rng_seed', type=int, default=0)
    parser.add_argument('--out_dir', default='./result/DS_3/train_val_diagnostics')
    parser.add_argument('--cache_dir', default='./result/DS_3/curves_cache/anomsim')
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

    by_domain = entities_by_domain(args.dataset_dir)
    all_entities = list_entities(args.dataset_dir)
    rng = np.random.default_rng(args.rng_seed)

    normal_entities = sorted(rng.choice(all_entities, size=min(args.n_entities, len(all_entities)), replace=False).tolist())
    domain_entities = {
        domain: sorted(np.random.default_rng(args.rng_seed + i).choice(
            ents, size=min(args.n_entities, len(ents)), replace=False).tolist())
        for i, domain in enumerate(sorted(by_domain))
        for ents in [by_domain[domain]]
    }
    print(f'Normal section entities: {normal_entities}')
    for domain, ents in domain_entities.items():
        print(f'  {domain}: {ents}')

    os.makedirs(args.out_dir, exist_ok=True)

    # Display order (what ends up in the PDF, top to bottom).
    sections = [('Normal', 'normal', normal_entities)]
    for domain in sorted(by_domain):
        for anomaly_type in REDLAMP_ANOMALY_TYPES[1:]:
            sections.append((f'{domain}_{anomaly_type}', anomaly_type, domain_entities[domain]))

    # Which (entity -> set of types) needs computing, grouped so each
    # entity's data is loaded from disk exactly once per split.
    entity_types = {}
    for _, anomaly_type, entities in sections:
        for entity_dir in entities:
            entity_types.setdefault(entity_dir, set()).add(anomaly_type)

    for split in ['train', 'val']:
        curves_by_key = {}
        for entity_dir, types in entity_types.items():
            cached_types = {t for t in types
                             if not args.force
                             and os.path.isfile(os.path.join(args.cache_dir, f'{entity_dir}_{split}_{t}.npz'))}
            needed_types = types - cached_types
            dataset = None
            if needed_types:
                dataset = load_entity_dataset(single_entity_module, online_dataset_module,
                                               args.dataset_dir, entity_dir, split, REDLAMP_ANOMALY_TYPES, args.seed)
            for anomaly_type in types:
                cache_path = os.path.join(args.cache_dir, f'{entity_dir}_{split}_{anomaly_type}.npz')

                def compute():
                    windows = windows_for_type(dataset, anomaly_type, REDLAMP_ANOMALY_TYPES)
                    return ldc.compute_dense_curves(windows, model, device)

                curves = ldc.get_or_compute_curves(cache_path, compute, force=args.force)
                curves_by_key[(entity_dir, anomaly_type)] = curves
                print(f'  computed/cached: {entity_dir} / {anomaly_type} / {split}')

        out_path = os.path.join(args.out_dir, f'AnomSim_{split.capitalize()}_inference_samples.pdf')
        with PdfPages(out_path) as pdf:
            for section_name, anomaly_type, entities in sections:
                for entity_dir in entities:
                    curves = curves_by_key[(entity_dir, anomaly_type)]
                    if len(curves['raw_series']) == 0:
                        print(f'[skip] {entity_dir}/{split}/{anomaly_type}: empty window set')
                        continue

                    focus_end = len(curves['raw_series']) // 2
                    focus_start, focus_end = ldc.window_bounds_from_end_index(focus_end, WINDOW_SIZE)

                    ldc.plot_diagnostic_page(
                        pdf, curves['raw_series'],
                        [dict(label='cross_anomsim', reconstruction=curves['reconstruction'],
                              mse_score=curves['mse_score'], ce_score=curves['ce_score'], score=curves['score'])],
                        focus_start, focus_end, WINDOW_SIZE,
                        title=f'AnomSim | {section_name} | {entity_dir} | {split}')

        print(f'Wrote {out_path}')

    print('Done.')


if __name__ == '__main__':
    run()
