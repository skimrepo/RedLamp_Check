"""
Large-scale continuous-domain pretraining vs unseen-domain holdout (KPI/UCR),
with progressive scaling.

Builds a candidate pool of univariate CONTINUOUS time series from SMD (22
safe continuous features x 28 machines), SMAP (54 channels, telemetry column
0 only), MSL (27 channels, column 0 only), and additional UCR entities (up to
247, excluding the 3 held-out kpi_1..3/ucr_1..3 used for zero-shot eval) — up
to 944 total. Trains one model per requested pool size (--n_series, processed
sequentially, from scratch each time) on a deterministic prefix of that pool,
then immediately evaluates it — zero-shot — against kpi_1..3/ucr_1..3.

Reuses cross_inference.py (discover_entity/build_dataparams/build_model_args/
plot_tsne_multi_source/ANOMALY_TYPES) and domain_generalization.py
(ENTITY_ALIASES/CFG/DATA_ALIASES/extract_embeddings_and_accuracy) directly.
Does not modify any existing file.
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import main
import datautils
import utils
from loaders.load import load_smd, load_smap, load_msl, load_anomaly_archive
from loaders.dataset import Dataset, Entity
from loaders.loader_aug import Loader_aug

import cross_inference as ci
import domain_generalization as dg


SMD_MACHINES = [f'machine-{g}-{n}' for g, count in [(1, 8), (2, 9), (3, 11)] for n in range(1, count + 1)]  # 28
SMD_CONTINUOUS_COLS = [0, 1, 2, 3, 5, 6, 11, 13, 14, 15, 18, 19, 20, 21, 22, 23, 25, 27, 30, 33, 34, 35]  # 22
UCR_HOLDOUT = {'001', '002', '003'}
WINDOW_SIZE = 100
WINDOW_STEP = 10
MIN_ENTITY_LENGTH = 100


def smap_channel_ids():
    meta = pd.read_csv('./dataset/NASA/labeled_anomalies.csv')
    return sorted(meta.loc[meta['spacecraft'] == 'SMAP']['chan_id'].unique().tolist())


def msl_channel_ids():
    meta = pd.read_csv('./dataset/NASA/labeled_anomalies.csv')
    return sorted(meta.loc[meta['spacecraft'] == 'MSL']['chan_id'].unique().tolist())


def ucr_entities_excluding_holdout():
    return [str(i).zfill(3) for i in range(1, 251) if str(i).zfill(3) not in UCR_HOLDOUT]


def build_candidate_pool():
    """Round-robin interleave across the 4 sources (one from SMD, one from
    SMAP, one from MSL, one from UCR, repeat — skipping any source once it's
    exhausted) instead of grouping all of SMD (616) first. Grouping would mean
    n < 616 is 100% SMD-only, which defeats the point of a diverse scaling
    pool. Interleaving keeps every prefix candidates[:n] a genuine mix of all
    4 sources (until the smaller sources run out — MSL exhausts at n~108,
    SMAP at n~189, UCR at n~575 — after which only SMD remains for the tail).
    Still fully deterministic, so candidates[:n] is always a strict
    prefix-subset of candidates[:m] for n < m."""
    smd_candidates = [('smd', machine, col) for machine in SMD_MACHINES for col in SMD_CONTINUOUS_COLS]
    smap_candidates = [('smap', channel, 0) for channel in smap_channel_ids()]
    msl_candidates = [('msl', channel, 0) for channel in msl_channel_ids()]
    ucr_candidates = [('ucr', entity, 0) for entity in ucr_entities_excluding_holdout()]

    sources = [smd_candidates, smap_candidates, msl_candidates, ucr_candidates]
    candidates = []
    idx = 0
    while any(idx < len(source) for source in sources):
        for source in sources:
            if idx < len(source):
                candidates.append(source[idx])
        idx += 1
    return candidates


def load_cached(cache, source_tag, entity_id):
    key = (source_tag, entity_id)
    if key in cache:
        return cache[key]
    if source_tag == 'smd':
        train_ds, val_ds = load_smd(group='train', machines=entity_id, downsampling=1,
                                     root_dir='./dataset', validation=True, verbose=False)
    elif source_tag == 'smap':
        train_ds, val_ds = load_smap(group='train', channels=entity_id, downsampling=1,
                                      root_dir='./dataset', validation=True, verbose=False)
    elif source_tag == 'msl':
        train_ds, val_ds = load_msl(group='train', channels=entity_id, downsampling=1,
                                     root_dir='./dataset', validation=True, verbose=False)
    elif source_tag == 'ucr':
        train_ds, val_ds = load_anomaly_archive(group='train', datasets=entity_id, downsampling=1,
                                                 min_length=None, root_dir='./dataset', validation=True, verbose=False)
    else:
        raise ValueError(f'unknown source_tag {source_tag!r}')
    cache[key] = (train_ds.entities[0], val_ds.entities[0])
    return cache[key]


def assemble(selected_candidates):
    cache = {}
    train_entities, val_entities = [], []
    dropped = 0
    for source_tag, entity_id, col in selected_candidates:
        base_train, base_val = load_cached(cache, source_tag, entity_id)
        if source_tag == 'ucr':
            train_e, val_e = base_train, base_val  # already univariate, no slicing needed
        else:
            name = f'{source_tag}_{entity_id}_f{col}'
            train_e = Entity(Y=base_train.Y[[col], :], name=name)
            val_e = Entity(Y=base_val.Y[[col], :], name=name)
        if train_e.Y.shape[1] < MIN_ENTITY_LENGTH or val_e.Y.shape[1] < MIN_ENTITY_LENGTH:
            dropped += 1
            continue
        train_entities.append(train_e)
        val_entities.append(val_e)
    return train_entities, val_entities, dropped


def wrap_loader(dataset_obj, batch_size, shuffle):
    return Loader_aug(dataset=dataset_obj, batch_size=batch_size, window_size=WINDOW_SIZE,
                       window_step=WINDOW_STEP, anomaly_types=ci.ANOMALY_TYPES, min_range=1,
                       min_features=1, max_features=1, fast_sampling=False, shuffle=shuffle, verbose=True)


def run_stage(candidates, n_series, run_name, seed, device, batch_size):
    selected = candidates[:n_series]
    train_entities, val_entities, dropped = assemble(selected)
    actual_n = len(train_entities)
    model_dir = f'./result/{run_name}/_pooled/continuous_n{n_series}/{seed}'

    if os.path.isfile(f'{model_dir}/bestmodel.pkl'):
        print(f'[skip] {model_dir}/bestmodel.pkl exists — reusing (requested={n_series}, actual={actual_n}, dropped={dropped})')
        return model_dir, actual_n, dropped, None

    os.makedirs(model_dir, exist_ok=True)
    train_dl = wrap_loader(Dataset(entities=train_entities, name=f'n{n_series}-train'), batch_size, shuffle=True)
    val_dl = wrap_loader(Dataset(entities=val_entities, name=f'n{n_series}-val'), batch_size, shuffle=True)
    print(f'n_series={n_series}: requested={n_series}, actual={actual_n}, dropped={dropped}, '
          f'train windows={len(train_dl)}, val windows={len(val_dl)}')

    model_args = ci.build_model_args(dg.CFG, WINDOW_SIZE)
    params = utils.AttrDict(batch_size=batch_size, lr=0.001, epoch=100, max_grad_norm=1.0, seed=seed)
    params.override(main.model_parameters(model_args))

    start = time.time()
    main.REDLAMP(model_dir=model_dir, params=params, device=device).train(train_dl, val_dl)
    elapsed = time.time() - start
    print(f'n_series={n_series}: training took {elapsed:.1f}s')
    return model_dir, actual_n, dropped, elapsed


def evaluate_stage(model_dir, n_series, actual_n, dropped, elapsed, holdout_val_dls, anomaly_dict, device, args_cli):
    tag = f'continuous_n{n_series}'
    out_dir = f'./result/{args_cli.run_name}/_cross_domain_holdout/n{n_series}'
    plots_dir = f'{out_dir}/plots'
    os.makedirs(plots_dir, exist_ok=True)

    model_args = ci.build_model_args(dg.CFG, WINDOW_SIZE)
    infer_params = utils.AttrDict(seed=args_cli.seed)
    infer_params.override(main.model_parameters(model_args))

    embeddings_cache = {}
    accuracy = {}
    for data_alias in dg.DATA_ALIASES:
        embeddings, class_idx, acc = dg.extract_embeddings_and_accuracy(
            model_dir, infer_params, device, holdout_val_dls[data_alias], max_samples=args_cli.tsne_max_samples)
        embeddings_cache[data_alias] = (embeddings, class_idx)
        accuracy[data_alias] = acc

        save_path = f'{plots_dir}/model_{tag}__data_{data_alias}.png'
        main.plot_tsne_embeddings(
            embeddings, class_idx, anomaly_dict, save_path,
            title=f'model={tag} / data={data_alias} (val, n={len(embeddings)}, acc={acc:.2f})',
            perplexity=args_cli.tsne_perplexity, seed=args_cli.seed)

    pooled_embeddings, pooled_class_idx, pooled_source_idx = [], [], []
    for source_idx, data_alias in enumerate(dg.DATA_ALIASES):
        embeddings, class_idx = embeddings_cache[data_alias]
        pooled_embeddings.append(embeddings)
        pooled_class_idx.append(class_idx)
        pooled_source_idx.append(np.full(len(embeddings), source_idx))
    pooled_embeddings = np.concatenate(pooled_embeddings, axis=0)
    pooled_class_idx = np.concatenate(pooled_class_idx, axis=0)
    pooled_source_idx = np.concatenate(pooled_source_idx, axis=0)
    ci.plot_tsne_multi_source(
        pooled_embeddings, pooled_class_idx, pooled_source_idx, anomaly_dict, dg.DATA_ALIASES,
        f'{plots_dir}/combined_model_{tag}.png',
        title=f'model={tag} across kpi/ucr holdout (val, n={len(pooled_embeddings)})',
        perplexity=args_cli.tsne_perplexity, seed=args_cli.seed)

    df = pd.DataFrame([accuracy], index=[tag])
    df.index.name = 'model'
    df.to_csv(f'{out_dir}/accuracy.csv')

    summary = dict(n_requested=n_series, actual_n=actual_n, dropped_count=dropped,
                   train_seconds=elapsed, mean_accuracy=float(np.mean(list(accuracy.values()))))
    with open(f'{out_dir}/stage_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f'n_series={n_series}: mean holdout accuracy={summary["mean_accuracy"]:.4f} '
          f'(actual_n={actual_n}, dropped={dropped}, train_seconds={elapsed})')


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', default='test')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--n_series', type=int, nargs='+', default=[50, 100, 200, 400, 800, 944])
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--tsne_max_samples', type=int, default=2000)
    parser.add_argument('--tsne_perplexity', type=float, default=30)
    args_cli = parser.parse_args()

    device = utils.init_dl_program(args_cli.gpu, seed=args_cli.seed)

    print('Building candidate pool...')
    candidates = build_candidate_pool()
    print(f'Total candidate pool size: {len(candidates)}')

    print('Resolving the 6 holdout entities (kpi_1..3, ucr_1..3)...')
    holdout_val_dls = {}
    for alias, (dataset, real_name) in dg.ENTITY_ALIASES.items():
        model_dir, disk_cfg = ci.discover_entity(args_cli.run_name, dataset, real_name, args_cli.seed)
        dataparams = ci.build_dataparams(dataset, real_name, dg.CFG, disk_cfg)
        _, val_dl = datautils.load_dataloader_aug(dataparams, group='train')
        holdout_val_dls[alias] = val_dl
        print(f'  {alias}: val windows={len(val_dl)}')
    anomaly_dict = holdout_val_dls[dg.DATA_ALIASES[0]].anomaly_dict

    for n_series in args_cli.n_series:
        print(f'=== Stage n_series={n_series} ===')
        model_dir, actual_n, dropped, elapsed = run_stage(
            candidates, n_series, args_cli.run_name, args_cli.seed, device, args_cli.batch_size)
        evaluate_stage(model_dir, n_series, actual_n, dropped, elapsed,
                       holdout_val_dls, anomaly_dict, device, args_cli)

    print('Done.')


if __name__ == '__main__':
    run()
