"""
Build DS_0: a centralized "what did we actually train/infer on" gallery,
covering every dataset used anywhere in this project -- anomaly_archive
(UCR), iops (KPI), smd, smap, msl, and AnomSim_v1 (Core-Clustering, a
sibling repo). For each entity, copies (never moves) a small representative
sample (1 waveform example, 1 injected-anomaly example, 1 t-SNE embedding
plot) out of the per-entity artifacts main.py / Core-Clustering already
generate unconditionally at training time -- no new inference or plotting,
purely aggregation.

Why this is needed: those per-entity artifacts already exist but are
scattered across three different relocation schemes, none of which any
single existing helper knows about all at once:
  - anomaly_archive/iops: moved into Experiment_1/Models/Self/{dataset}/
    {entity}/{seed}/ by organize_experiment1.py -- ci.discover_entity's own
    fallback already knows this path, reused as-is here.
  - smd/smap/msl: moved into Experiment_0_Exploratory/self_accuracy_report/
    models/{dataset}/{entity}/d*_b*_w*_s*/{seed}/ by
    organize_experiment0_exploratory.py (whole directory shutil.move, so the
    internal d*_b*_w*_s*/{seed}/ts_example_plots/ structure survives
    untouched) -- ci.discover_entity does NOT know about this move, so this
    script has its own small fallback for these three datasets.
  - AnomSim_v1: never moved, lives at Core-Clustering's own
    outputs/self/{entity}/{seed}/plots/{samples/,tsne_by_class.png} (a
    different repo entirely, so it needs its own --core_clustering_outputs_dir).

smap/msl are trained as ONE shared model each (main.py's entity_list =
['smap']/['msl'], not per real channel), so "at least 1 sample per entity"
is already satisfied by their single example set -- there is no per-channel
(e.g. A-1, D-14) plot to gather, by design, not a gap.

Only copies 3 small files per entity (not the full ~36-image example set)
to keep the gallery browsable; coverage.csv's source_path column points
back at the original full example folder for deeper digging. Safe to rerun:
always overwrites the gallery from whatever currently exists at the source
locations, and coverage.csv always reflects the current state (including
entities still missing artifacts, e.g. not yet trained).

Does not modify main.py, cross_inference.py, full_reproduction_metrics.py,
or organize_experiment0_exploratory.py -- only imports from cross_inference
and full_reproduction_metrics.
"""
import argparse
import glob
import os
import shutil
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cross_inference as ci
import full_reproduction_metrics as frm

REDLAMP_DATASETS = ['anomaly_archive', 'iops', 'smd', 'smap', 'msl']
DATASET_OUT_NAME = {'anomaly_archive': 'anomaly_archive', 'iops': 'iops', 'smd': 'smd',
                     'smap': 'smap', 'msl': 'msl'}

# Tried in order; first one whose example PNG actually exists is used as the
# "anomaly_example.png" representative (spike is the most visually obvious).
REPRESENTATIVE_ANOMALY_TYPES = ['spike', 'flip', 'speedup', 'noise', 'cutoff', 'average',
                                 'scale', 'wander', 'contextual', 'upsidedown', 'mixture']


def _first_existing(paths):
    for p in paths:
        if os.path.isfile(p):
            return p
    return None


def discover_entities(run_name, dataset, exp0_dir):
    """Lists entities for a dataset that may have already been fully moved
    away from its original result/{run_name}/{dataset}/ location (smd/smap/
    msl, by organize_experiment0_exploratory.py) -- falls back to listing
    what's under the Experiment_0_Exploratory relocation if the original
    location is now empty/gone."""
    entities = set(frm.discover_dataset_entities(run_name, dataset))
    base = f'{exp0_dir}/self_accuracy_report/models/{dataset}'
    if os.path.isdir(base):
        entities.update(d for d in os.listdir(base)
                         if os.path.isdir(os.path.join(base, d)) and not d.startswith('_'))
    return sorted(entities)


def find_redlamp_model_dir(run_name, dataset, entity, seed, exp0_dir):
    """Resolves one entity's model_dir, trying (in order): its original
    location, ci.discover_entity's own Experiment_1 fallback (anomaly_archive/
    iops only), and the Experiment_0_Exploratory relocation (smd/smap/msl
    only). Returns None if not found anywhere."""
    if dataset in ('anomaly_archive', 'iops'):
        try:
            model_dir, _ = ci.discover_entity(run_name, dataset, entity, seed)
            return model_dir
        except FileNotFoundError:
            return None

    pattern = f'./result/{run_name}/{dataset}/{entity}/d*_b*_w*_s*/{seed}'
    candidates = [m for m in glob.glob(pattern) if os.path.isfile(os.path.join(m, 'bestmodel.pkl'))]
    if not candidates:
        pattern = f'{exp0_dir}/self_accuracy_report/models/{dataset}/{entity}/d*_b*_w*_s*/{seed}'
        candidates = [m for m in glob.glob(pattern) if os.path.isfile(os.path.join(m, 'bestmodel.pkl'))]
    return candidates[0] if candidates else None


def gather_redlamp_entity(model_dir):
    """Returns (waveform_src, anomaly_src, tsne_src, examples_dir), any of
    which may be None if that particular artifact is missing."""
    examples_dir = os.path.join(model_dir, 'ts_example_plots')
    waveform_src = _first_existing([os.path.join(examples_dir, 'normal_1.png')])
    anomaly_src = _first_existing([os.path.join(examples_dir, f'{t}_1.png')
                                    for t in REPRESENTATIVE_ANOMALY_TYPES])
    tsne_path = os.path.join(model_dir, 'tsne_embeddings.png')
    tsne_src = tsne_path if os.path.isfile(tsne_path) else None
    return waveform_src, anomaly_src, tsne_src, examples_dir


def discover_anomsim_entities(core_clustering_outputs_dir):
    base = os.path.join(core_clustering_outputs_dir, 'self')
    if not os.path.isdir(base):
        return []
    return sorted(d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d)))


def find_anomsim_seed_dir(core_clustering_outputs_dir, entity, seed):
    entity_dir = os.path.join(core_clustering_outputs_dir, 'self', entity)
    candidate = os.path.join(entity_dir, str(seed))
    if os.path.isdir(candidate):
        return candidate
    subdirs = sorted(d for d in os.listdir(entity_dir) if os.path.isdir(os.path.join(entity_dir, d)))
    return os.path.join(entity_dir, subdirs[0]) if subdirs else None


def gather_anomsim_entity(seed_dir):
    """Returns (waveform_src, tsne_src, examples_dir). AnomSim's 'domain' in
    {domain}_{i}.png is the entity's own waveform-type (meta['type']), not an
    anomaly-type name like main.py's -- for a single-entity run there's just
    one such domain, so any *_1.png match is that entity's representative
    sample."""
    examples_dir = os.path.join(seed_dir, 'plots', 'samples')
    samples = sorted(glob.glob(os.path.join(examples_dir, '*_1.png')))
    waveform_src = samples[0] if samples else None
    tsne_path = os.path.join(seed_dir, 'plots', 'tsne_by_class.png')
    tsne_src = tsne_path if os.path.isfile(tsne_path) else None
    return waveform_src, tsne_src, examples_dir


def build_gallery(run_name, seed, out_dir, exp0_dir, core_clustering_outputs_dir):
    rows = []

    for dataset in REDLAMP_DATASETS:
        out_name = DATASET_OUT_NAME[dataset]
        entities = discover_entities(run_name, dataset, exp0_dir)
        print(f'{dataset}: found {len(entities)} entity directories')
        for entity in entities:
            model_dir = find_redlamp_model_dir(run_name, dataset, entity, seed, exp0_dir)
            dest_dir = os.path.join(out_dir, out_name, entity)
            if model_dir is None:
                rows.append(dict(dataset=dataset, entity=entity, has_examples=False,
                                  has_tsne=False, source_path=None))
                print(f'  [missing] {dataset}/{entity}: no model_dir found for seed={seed}')
                continue

            waveform_src, anomaly_src, tsne_src, examples_dir = gather_redlamp_entity(model_dir)
            os.makedirs(dest_dir, exist_ok=True)
            if waveform_src:
                shutil.copy2(waveform_src, os.path.join(dest_dir, 'waveform.png'))
            if anomaly_src:
                shutil.copy2(anomaly_src, os.path.join(dest_dir, 'anomaly_example.png'))
            if tsne_src:
                shutil.copy2(tsne_src, os.path.join(dest_dir, 'tsne.png'))

            rows.append(dict(dataset=dataset, entity=entity,
                              has_examples=bool(waveform_src or anomaly_src),
                              has_tsne=bool(tsne_src), source_path=examples_dir))
            print(f'  {dataset}/{entity}: waveform={bool(waveform_src)}, '
                  f'anomaly_example={bool(anomaly_src)}, tsne={bool(tsne_src)}')

    entities = discover_anomsim_entities(core_clustering_outputs_dir)
    print(f'anomsim_v1: found {len(entities)} entity directories')
    for entity in entities:
        seed_dir = find_anomsim_seed_dir(core_clustering_outputs_dir, entity, seed)
        dest_dir = os.path.join(out_dir, 'anomsim_v1', entity)
        if seed_dir is None:
            rows.append(dict(dataset='anomsim_v1', entity=entity, has_examples=False,
                              has_tsne=False, source_path=None))
            print(f'  [missing] anomsim_v1/{entity}: no trained seed dir found')
            continue

        waveform_src, tsne_src, examples_dir = gather_anomsim_entity(seed_dir)
        os.makedirs(dest_dir, exist_ok=True)
        if waveform_src:
            shutil.copy2(waveform_src, os.path.join(dest_dir, 'waveform.png'))
        if tsne_src:
            shutil.copy2(tsne_src, os.path.join(dest_dir, 'tsne.png'))

        rows.append(dict(dataset='anomsim_v1', entity=entity, has_examples=bool(waveform_src),
                          has_tsne=bool(tsne_src), source_path=examples_dir))
        print(f'  anomsim_v1/{entity}: waveform={bool(waveform_src)}, tsne={bool(tsne_src)}')

    coverage_df = pd.DataFrame(rows)
    coverage_csv = os.path.join(out_dir, 'coverage.csv')
    coverage_df.to_csv(coverage_csv, index=False)

    if not coverage_df.empty:
        print(coverage_df.groupby('dataset')[['has_examples', 'has_tsne']].mean().to_string())
    print(f'Done. {len(coverage_df)} entities catalogued across '
          f'{coverage_df["dataset"].nunique() if not coverage_df.empty else 0} datasets. '
          f'Wrote {coverage_csv}')


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', default='test')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out_dir', default='./result/DS_0')
    parser.add_argument('--exp0_dir', default='./result/Experiment_0_Exploratory',
                         help='Where organize_experiment0_exploratory.py relocated smd/smap/msl.')
    parser.add_argument('--core_clustering_outputs_dir', default='../Core-Clustering/outputs',
                         help='Sibling repo outputs/ dir, containing self/{entity}/{seed}/plots/...')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    build_gallery(args.run_name, args.seed, args.out_dir, args.exp0_dir, args.core_clustering_outputs_dir)


if __name__ == '__main__':
    run()
