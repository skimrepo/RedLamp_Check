"""
Domain-generalization experiment: KPI (iops) vs UCR (anomaly_archive).

Builds a 9 (model) x 6 (data) matrix:
  - 6 models: the already-trained per-entity models (kpi_1..3, ucr_1..3) —
    reused as-is via cross_inference.discover_entity(), never retrained.
  - 3 models: newly trained pooled models (kpi_all, ucr_all, kpi_ucr_all),
    each trained on its constituent entities' data merged into one Dataset.
  - 6 data columns: always the 6 individual entities' own validation splits.

For every (model, data) pair: one t-SNE plot (main.plot_tsne_embeddings,
same format as tsne_embeddings.png) + one classification accuracy number.
For every model (9): one combined t-SNE pooling all 6 data sources
(cross_inference.plot_tsne_multi_source, color=anomaly type, marker=data
source). Outputs to result/{run_name}/_cross_domain/.

Does not modify main.py or cross_inference.py — only imports from them.
"""
import argparse
import csv
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import main
import datautils
import utils
from loaders.load import load_iops, load_anomaly_archive
from loaders.dataset import Dataset
from loaders.loader_aug import Loader_aug

import cross_inference as ci


ENTITY_ALIASES = {
    'kpi_1': ('iops', 'KPI-05f10d3a-239c-3bef-9bdc-a2feeb0037aa'),
    'kpi_2': ('iops', 'KPI-0efb375b-b902-3661-ab23-9a0bb799f4e3'),
    'kpi_3': ('iops', 'KPI-1c6d7a26-1f1a-3321-bb4d-7a9d969ec8f0'),
    'ucr_1': ('anomaly_archive', '001'),
    'ucr_2': ('anomaly_archive', '002'),
    'ucr_3': ('anomaly_archive', '003'),
}
POOLED_MODELS = {
    'kpi_all':     ['kpi_1', 'kpi_2', 'kpi_3'],
    'ucr_all':     ['ucr_1', 'ucr_2', 'ucr_3'],
    'kpi_ucr_all': ['kpi_1', 'kpi_2', 'kpi_3', 'ucr_1', 'ucr_2', 'ucr_3'],
}
MODEL_ALIASES = list(ENTITY_ALIASES) + list(POOLED_MODELS)  # 9
DATA_ALIASES = list(ENTITY_ALIASES)  # 6, always fixed

# Both KPI and UCR entities already share these — single shared config for everyone.
CFG = dict(n_features=1, min_features=1, max_features=1)
WINDOW_SIZE = 100
WINDOW_STEP = 10
BATCH_SIZE = 128
DOWNSAMPLING = 1


def load_single_entity_train_val(dataset, real_name):
    """Load one entity's own train+val Entity objects. Single-string calls only
    (see plan notes: load_anomaly_archive's list-matching is broken for bare
    zero-padded names, so entities are always pooled by hand afterward)."""
    if dataset == 'iops':
        train_ds, val_ds = load_iops(group='train', filename=real_name, downsampling=DOWNSAMPLING,
                                      root_dir='./dataset', validation=True, verbose=False)
    elif dataset == 'anomaly_archive':
        train_ds, val_ds = load_anomaly_archive(group='train', datasets=real_name, downsampling=DOWNSAMPLING,
                                                 min_length=None, root_dir='./dataset', validation=True, verbose=False)
    else:
        raise ValueError(f'Unsupported dataset for pooling: {dataset!r}')
    return train_ds.entities[0], val_ds.entities[0]


def wrap_loader(dataset_obj, shuffle):
    return Loader_aug(dataset=dataset_obj, batch_size=BATCH_SIZE, window_size=WINDOW_SIZE,
                       window_step=WINDOW_STEP, anomaly_types=ci.ANOMALY_TYPES, min_range=1,
                       min_features=CFG['min_features'], max_features=CFG['max_features'],
                       fast_sampling=False, shuffle=shuffle, verbose=True)


def ensure_pooled_model(pooled_alias, run_name, seed, device):
    model_dir = f'./result/{run_name}/_pooled/{pooled_alias}/{seed}'
    if os.path.isfile(f'{model_dir}/bestmodel.pkl'):
        print(f'[skip] {model_dir}/bestmodel.pkl exists — reusing')
        return model_dir

    os.makedirs(model_dir, exist_ok=True)
    train_entities, val_entities = [], []
    for constituent in POOLED_MODELS[pooled_alias]:
        dataset, real_name = ENTITY_ALIASES[constituent]
        train_entity, val_entity = load_single_entity_train_val(dataset, real_name)
        train_entities.append(train_entity)
        val_entities.append(val_entity)

    train_dataset = Dataset(entities=train_entities, name=f'{pooled_alias}-train')
    val_dataset = Dataset(entities=val_entities, name=f'{pooled_alias}-val')
    train_dl = wrap_loader(train_dataset, shuffle=True)
    val_dl = wrap_loader(val_dataset, shuffle=True)
    print(f'Training {pooled_alias}: {len(train_dl)} train windows / {len(val_dl)} val windows '
          f'(pooled from {POOLED_MODELS[pooled_alias]})')

    model_args = ci.build_model_args(CFG, WINDOW_SIZE)
    params = utils.AttrDict(batch_size=BATCH_SIZE, lr=0.001, epoch=100, max_grad_norm=1.0, seed=seed)
    params.override(main.model_parameters(model_args))

    model = main.REDLAMP(model_dir=model_dir, params=params, device=device)
    model.train(train_dl, val_dl)
    return model_dir


def extract_embeddings_and_accuracy(model_dir, params, device, val_dataloader, max_samples=2000):
    """Combines main.extract_embeddings' job (t-SNE embeddings, subsampled) with
    classification accuracy (over the FULL val set, not subsampled) in a single
    model load + single pass over val_dataloader, instead of doing both
    separately (which would load the model and iterate the data twice)."""
    model = main.ConvAEC(params).to(device)
    model.load_state_dict(torch.load(f'{model_dir}/bestmodel.pkl'))
    model.eval()

    embeddings_list, class_idx_list = [], []
    correct, total = 0, 0
    with torch.no_grad():
        for batch in val_dataloader:
            inputs = batch['Y'].transpose(2, 1).to(device)
            true = batch['label'].argmax(dim=1)

            _, x_out, x_enc = model(inputs)
            embeddings_list.append(x_enc.squeeze(-1).to('cpu').detach().numpy())
            class_idx_list.append(true.numpy())

            pred = x_out.argmax(dim=1).cpu()
            correct += (pred == true).sum().item()
            total += len(true)

    embeddings = np.concatenate(embeddings_list, axis=0)
    class_idx = np.concatenate(class_idx_list, axis=0)
    accuracy = correct / total

    if len(embeddings) > max_samples:
        rng = np.random.RandomState(params.seed)
        sample_idx = rng.choice(len(embeddings), max_samples, replace=False)
        embeddings = embeddings[sample_idx]
        class_idx = class_idx[sample_idx]

    return embeddings, class_idx, accuracy


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', default='test')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--tsne_max_samples', type=int, default=2000)
    parser.add_argument('--tsne_perplexity', type=float, default=30)
    parser.add_argument('--out_dir', default=None)
    args_cli = parser.parse_args()

    out_dir = args_cli.out_dir or f'./result/{args_cli.run_name}/_cross_domain'
    plots_dir = f'{out_dir}/plots'
    os.makedirs(plots_dir, exist_ok=True)

    device = utils.init_dl_program(args_cli.gpu, seed=args_cli.seed)

    with open(f'{out_dir}/alias_map.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['alias', 'dataset', 'real_entity_name_or_constituents'])
        for alias, (dataset, real_name) in ENTITY_ALIASES.items():
            writer.writerow([alias, dataset, real_name])
        for alias, constituents in POOLED_MODELS.items():
            writer.writerow([alias, 'pooled', '+'.join(constituents)])

    # Resolve the 6 base entities — reuse already-trained models, never retrain.
    model_dirs = {}
    val_dls = {}
    for alias, (dataset, real_name) in ENTITY_ALIASES.items():
        model_dir, disk_cfg = ci.discover_entity(args_cli.run_name, dataset, real_name, args_cli.seed)
        model_dirs[alias] = model_dir
        dataparams = ci.build_dataparams(dataset, real_name, CFG, disk_cfg)
        _, val_dl = datautils.load_dataloader_aug(dataparams, group='train')
        val_dls[alias] = val_dl
        print(f'{alias}: model_dir={model_dir}, val windows={len(val_dl)}')

    # Train (or reuse, if already trained) the 3 pooled models.
    for pooled_alias in POOLED_MODELS:
        model_dirs[pooled_alias] = ensure_pooled_model(pooled_alias, args_cli.run_name, args_cli.seed, device)

    # Shared inference-time params (architecture is identical for everyone).
    model_args = ci.build_model_args(CFG, WINDOW_SIZE)
    params = utils.AttrDict(seed=args_cli.seed)
    params.override(main.model_parameters(model_args))

    anomaly_dict = val_dls[DATA_ALIASES[0]].anomaly_dict

    embeddings_cache = {}
    accuracy = {m: {} for m in MODEL_ALIASES}
    for model_alias in MODEL_ALIASES:
        for data_alias in DATA_ALIASES:
            print(f'Inferring: model={model_alias} data={data_alias}')
            embeddings, class_idx, acc = extract_embeddings_and_accuracy(
                model_dirs[model_alias], params, device, val_dls[data_alias],
                max_samples=args_cli.tsne_max_samples)
            embeddings_cache[(model_alias, data_alias)] = (embeddings, class_idx)
            accuracy[model_alias][data_alias] = acc

    print('Saving 54 individual plots...')
    for model_alias in MODEL_ALIASES:
        for data_alias in DATA_ALIASES:
            embeddings, class_idx = embeddings_cache[(model_alias, data_alias)]
            tag = 'self' if model_alias == data_alias else 'cross'
            save_path = f'{plots_dir}/model_{model_alias}__data_{data_alias}__{tag}.png'
            acc = accuracy[model_alias][data_alias]
            main.plot_tsne_embeddings(
                embeddings, class_idx, anomaly_dict, save_path,
                title=f'model={model_alias} / data={data_alias} (val, n={len(embeddings)}, acc={acc:.2f})',
                perplexity=args_cli.tsne_perplexity, seed=args_cli.seed)

    print('Saving 9 combined per-model plots...')
    for model_alias in MODEL_ALIASES:
        pooled_embeddings, pooled_class_idx, pooled_source_idx = [], [], []
        for source_idx, data_alias in enumerate(DATA_ALIASES):
            embeddings, class_idx = embeddings_cache[(model_alias, data_alias)]
            pooled_embeddings.append(embeddings)
            pooled_class_idx.append(class_idx)
            pooled_source_idx.append(np.full(len(embeddings), source_idx))
        pooled_embeddings = np.concatenate(pooled_embeddings, axis=0)
        pooled_class_idx = np.concatenate(pooled_class_idx, axis=0)
        pooled_source_idx = np.concatenate(pooled_source_idx, axis=0)

        save_path = f'{plots_dir}/combined_model_{model_alias}.png'
        ci.plot_tsne_multi_source(
            pooled_embeddings, pooled_class_idx, pooled_source_idx, anomaly_dict, DATA_ALIASES,
            save_path, title=f'model={model_alias} across all 6 datasets (val, n={len(pooled_embeddings)})',
            perplexity=args_cli.tsne_perplexity, seed=args_cli.seed)

    df = pd.DataFrame({data_alias: [accuracy[m][data_alias] for m in MODEL_ALIASES] for data_alias in DATA_ALIASES},
                       index=MODEL_ALIASES)
    df.index.name = 'model'
    df.to_csv(f'{out_dir}/classification_accuracy.csv')

    print('Done. Outputs in', out_dir)


if __name__ == '__main__':
    run()
