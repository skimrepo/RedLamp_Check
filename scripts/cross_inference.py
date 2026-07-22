"""
Cross-inference experiment: for a dataset where multiple entities each got
their own trained model (e.g. iops's 3 KPI models), run every (model, data)
combination on the validation split and visualize the resulting embeddings.

Produces, under result/{run_name}/{dataset}/_cross_inference/:
  - model_{M}__data_{D}__self.png / __cross.png   (9 = 3x3 individual t-SNE
    plots, same format as main.py's tsne_embeddings.png: color = anomaly type)
  - combined_model_{M}.png   (one per model: that model's embeddings across
    ALL entities' data pooled into a single t-SNE; color = anomaly type,
    marker = which entity's data the point came from)

Reuses main.py's extract_embeddings/plot_tsne_embeddings directly so the
9 individual plots are pixel-for-pixel the same format as the existing
per-entity tsne_embeddings.png. Does not modify main.py.
"""
import argparse
import os
import sys

import numpy as np
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import main
import datautils
import utils


ANOMALY_TYPES = ['normal', 'spike', 'flip', 'speedup', 'noise', 'cutoff',
                 'average', 'scale', 'wander', 'contextual', 'upsidedown', 'mixture']

# Add datasets here as needed; must match main.py's per-dataset branch exactly.
DATASET_CONFIGS = {
    'iops': dict(n_features=1, min_features=1, max_features=1, batch_size=128,
                 window_size=100, window_step=10, downsampling=1),
}

# Diagnostic-only palette: 8 shared hues from main.py + 4 more, so all 12
# anomaly classes get a distinct color with marker free to encode data source.
EXT_COLORS = main.CHART_COLORS + ['#8b4513', '#20b2aa', '#6b8e23', '#2f4f4f']
SOURCE_MARKERS = ['o', '^', 's', 'D', 'P', 'X']


def build_model_args(cfg):
    return utils.AttrDict(
        model='ConvAEC',
        n_features=cfg['n_features'],
        window_size=cfg['window_size'],
        embedding_dim=128,
        anomaly_types=ANOMALY_TYPES,
        c_loss_ratio=0.1,
        apply_anomaly_mask=True,
        label_smoothing=True,
    )


def build_dataparams(dataset, entity, cfg):
    return utils.AttrDict(
        dataset=dataset,
        entities=entity,
        downsampling=cfg['downsampling'],
        batch_size=cfg['batch_size'],
        window_size=cfg['window_size'],
        window_step=cfg['window_step'],
        anomaly_types=ANOMALY_TYPES,
        min_range=1,
        min_features=cfg['min_features'],
        max_features=cfg['max_features'],
    )


def model_dir_for(run_name, dataset, entity, cfg, seed):
    data_dir = (f"{dataset}/{entity}/d{cfg['downsampling']}_b{cfg['batch_size']}"
                f"_w{cfg['window_size']}_s{cfg['window_step']}")
    return f'./result/{run_name}/{data_dir}/{seed}'


def plot_tsne_multi_source(pooled_embeddings, pooled_class_idx, pooled_source_idx,
                            anomaly_dict, source_names, save_path, title=None,
                            perplexity=30, seed=0):
    import matplotlib.pyplot as plt

    n_samples = pooled_embeddings.shape[0]
    eff_perplexity = max(5, min(perplexity, n_samples - 1))
    reduced = TSNE(n_components=2, perplexity=eff_perplexity, random_state=seed,
                   init='pca').fit_transform(pooled_embeddings)

    inverse_dict = {v: k for k, v in anomaly_dict.items()}
    SURFACE, GRID, AXIS, MUTED, INK, INK2 = (main.SURFACE, main.GRID, main.AXIS,
                                              main.MUTED, main.INK, main.INK2)

    fig, ax = plt.subplots(figsize=(10, 8), facecolor=SURFACE)
    fig.subplots_adjust(right=0.72)
    ax.set_facecolor(SURFACE)
    for class_value in sorted(inverse_dict.keys()):
        for source_idx in range(len(source_names)):
            mask = (pooled_class_idx == class_value) & (pooled_source_idx == source_idx)
            if not mask.any():
                continue
            color = EXT_COLORS[class_value % len(EXT_COLORS)]
            marker = SOURCE_MARKERS[source_idx % len(SOURCE_MARKERS)]
            ax.scatter(reduced[mask, 0], reduced[mask, 1], s=18, c=color, marker=marker,
                       alpha=0.8, edgecolors='none')

    ax.tick_params(colors=MUTED, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(AXIS)
    ax.grid(True, color=GRID, linewidth=0.6)
    if title:
        ax.set_title(title, color=INK, fontsize=11)

    type_handles = [Line2D([0], [0], marker='o', color='none',
                            markerfacecolor=EXT_COLORS[c % len(EXT_COLORS)], markersize=7,
                            label=inverse_dict[c]) for c in sorted(inverse_dict.keys())]
    type_legend = ax.legend(handles=type_handles, title='anomaly type', fontsize=7,
                             title_fontsize=7, frameon=True, facecolor=SURFACE,
                             edgecolor='none', framealpha=0.85, loc='upper left',
                             bbox_to_anchor=(1.02, 1.0))
    plt.setp(type_legend.get_texts(), color=INK2)
    plt.setp(type_legend.get_title(), color=INK2)
    ax.add_artist(type_legend)

    source_handles = [Line2D([0], [0], marker=SOURCE_MARKERS[i % len(SOURCE_MARKERS)],
                              color=INK2, linestyle='none', markersize=7, label=name)
                       for i, name in enumerate(source_names)]
    source_legend = ax.legend(handles=source_handles, title='data source', fontsize=7,
                               title_fontsize=7, frameon=True, facecolor=SURFACE,
                               edgecolor='none', framealpha=0.85, loc='lower left',
                               bbox_to_anchor=(1.02, 0.0))
    plt.setp(source_legend.get_texts(), color=INK2)
    plt.setp(source_legend.get_title(), color=INK2)

    fig.savefig(save_path, dpi=150, facecolor=SURFACE, bbox_inches='tight',
                bbox_extra_artists=(type_legend, source_legend))
    plt.close(fig)


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='iops')
    parser.add_argument('--run_name', default='test')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--entities', nargs='*', default=None,
                         help='Entity names to cross-infer. Default: auto-discover '
                              'every already-trained entity under result/{run_name}/{dataset}/')
    parser.add_argument('--tsne_max_samples', type=int, default=2000)
    parser.add_argument('--tsne_perplexity', type=float, default=30)
    parser.add_argument('--out_dir', default=None)
    args_cli = parser.parse_args()

    if args_cli.dataset not in DATASET_CONFIGS:
        raise ValueError(f"No config for dataset {args_cli.dataset!r} — add one to "
                          f"DATASET_CONFIGS in scripts/cross_inference.py first.")
    cfg = DATASET_CONFIGS[args_cli.dataset]

    if args_cli.entities:
        entities = args_cli.entities
    else:
        base = f'./result/{args_cli.run_name}/{args_cli.dataset}'
        entities = sorted(d for d in os.listdir(base)
                           if os.path.isdir(os.path.join(base, d)) and not d.startswith('_'))
    print('Entities:', entities)
    if len(entities) < 2:
        raise ValueError(f'Need at least 2 trained entities to cross-infer, found {entities}')

    out_dir = args_cli.out_dir or f'./result/{args_cli.run_name}/{args_cli.dataset}/_cross_inference'
    os.makedirs(out_dir, exist_ok=True)

    device = utils.init_dl_program(args_cli.gpu, seed=args_cli.seed)

    model_args = build_model_args(cfg)
    params = utils.AttrDict(seed=args_cli.seed)
    params.override(main.model_parameters(model_args))

    val_dataloaders = {}
    model_dirs = {}
    for entity in entities:
        dataparams = build_dataparams(args_cli.dataset, entity, cfg)
        _, val_dl = datautils.load_dataloader_aug(dataparams, group='train')
        val_dataloaders[entity] = val_dl
        model_dirs[entity] = model_dir_for(args_cli.run_name, args_cli.dataset, entity, cfg, args_cli.seed)
        print(f'  {entity}: model_dir={model_dirs[entity]}, val windows={len(val_dl)}')

    anomaly_dict = val_dataloaders[entities[0]].anomaly_dict

    results = {}
    for model_entity in entities:
        for data_entity in entities:
            print(f'Inferring: model={model_entity} data={data_entity}')
            embeddings, class_idx = main.extract_embeddings(
                model_dirs[model_entity], params, device, val_dataloaders[data_entity],
                max_samples=args_cli.tsne_max_samples)
            results[(model_entity, data_entity)] = (embeddings, class_idx)

    print('Saving 9 individual plots...')
    for model_entity in entities:
        for data_entity in entities:
            embeddings, class_idx = results[(model_entity, data_entity)]
            tag = 'self' if model_entity == data_entity else 'cross'
            save_path = f'{out_dir}/model_{model_entity}__data_{data_entity}__{tag}.png'
            main.plot_tsne_embeddings(
                embeddings, class_idx, anomaly_dict, save_path,
                title=f'model={model_entity} / data={data_entity} (val, n={len(embeddings)})',
                perplexity=args_cli.tsne_perplexity, seed=args_cli.seed)

    print('Saving 3 combined per-model plots...')
    for model_entity in entities:
        pooled_embeddings, pooled_class_idx, pooled_source_idx = [], [], []
        for source_idx, data_entity in enumerate(entities):
            embeddings, class_idx = results[(model_entity, data_entity)]
            pooled_embeddings.append(embeddings)
            pooled_class_idx.append(class_idx)
            pooled_source_idx.append(np.full(len(embeddings), source_idx))
        pooled_embeddings = np.concatenate(pooled_embeddings, axis=0)
        pooled_class_idx = np.concatenate(pooled_class_idx, axis=0)
        pooled_source_idx = np.concatenate(pooled_source_idx, axis=0)

        save_path = f'{out_dir}/combined_model_{model_entity}.png'
        plot_tsne_multi_source(
            pooled_embeddings, pooled_class_idx, pooled_source_idx, anomaly_dict, entities,
            save_path, title=f'model={model_entity} across all datasets (val, n={len(pooled_embeddings)})',
            perplexity=args_cli.tsne_perplexity, seed=args_cli.seed)

    print('Done. Outputs in', out_dir)


if __name__ == '__main__':
    run()
