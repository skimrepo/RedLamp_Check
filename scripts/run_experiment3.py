"""
Experiment 3 results aggregation: combines every seed's
experiment3_score_anomsim_domain.py output (one CSV per seed under
--results_dir) into one comparison across the three models (Self_A,
Cross_without_E, Cross_without_D), computes similar_domain_gain =
performance(Cross_without_D) - performance(Cross_without_E) paired per seed
(both share the same seed's model-init/data-shuffle randomness and differ
only in the one domain swapped between them -- E present vs D present), and
renders comparison figures.

Also writes SOURCE.txt pointer files under result/Experiment_3/Models/, in
the same style as Experiment_1's Cross-AnomSim/SOURCE.txt: the actual
checkpoints live in Core-Clustering's outputs/ directory (a sibling repo),
never copied into RedLamp_Check.

Does not modify experiment3_score_anomsim_domain.py -- only reads its output.
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

METRIC_KEYS = ['VUS_ROC', 'VUS_PR', 'R_AUC_ROC', 'R_AUC_PR', 'RF']
ALL_METRICS = METRIC_KEYS + ['classification_accuracy']
MODEL_ORDER = ['Self_A', 'Cross_without_E', 'Cross_without_D']
# Fixed categorical order (never re-cycled per chart) -- validated for CVD
# separation as a 3-slot set; see the dataviz skill's palette reference.
MODEL_COLORS = {'Self_A': '#2a78d6', 'Cross_without_E': '#eb6834', 'Cross_without_D': '#1baf7a'}


def load_all_scores(results_dir: str, pattern: str = 'experiment3_scores_seed*.csv') -> pd.DataFrame:
    paths = sorted(glob.glob(os.path.join(results_dir, pattern)))
    if not paths:
        raise FileNotFoundError(f'no per-seed score CSVs found under {results_dir} matching {pattern}')
    return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)


def aggregate(raw_df: pd.DataFrame):
    """Two-level average: first across the domain's fixed test instances
    within a (model, seed), then mean+std across seeds within a model --
    same two-level pattern as full_reproduction_metrics.aggregate()."""
    per_seed = raw_df.groupby(['model', 'seed'], as_index=False)[ALL_METRICS].mean()
    summary = per_seed.groupby('model')[ALL_METRICS].agg(['mean', 'std'])
    summary.columns = [f'{metric}_{stat}' for metric, stat in summary.columns]
    return per_seed, summary.reset_index()


def paired_similar_domain_gain(per_seed: pd.DataFrame) -> pd.DataFrame:
    """similar_domain_gain = performance(Cross_without_D) - performance(Cross_without_E),
    paired by seed."""
    pivoted = per_seed.pivot(index='seed', columns='model', values=ALL_METRICS)
    rows = []
    for seed in pivoted.index:
        row = {'seed': seed}
        for metric in ALL_METRICS:
            without_e = pivoted.loc[seed, (metric, 'Cross_without_E')]
            without_d = pivoted.loc[seed, (metric, 'Cross_without_D')]
            row[f'{metric}_gain'] = without_d - without_e
        rows.append(row)
    return pd.DataFrame(rows)


def _strip_axes(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def plot_model_comparison(summary: pd.DataFrame, out_path: str,
                           metrics=('VUS_ROC', 'RF', 'classification_accuracy')) -> None:
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 4))
    if len(metrics) == 1:
        axes = [axes]
    models = [m for m in MODEL_ORDER if m in summary['model'].values]
    for ax, metric in zip(axes, metrics):
        means = [summary.loc[summary['model'] == m, f'{metric}_mean'].iloc[0] for m in models]
        stds = [summary.loc[summary['model'] == m, f'{metric}_std'].iloc[0] for m in models]
        colors = [MODEL_COLORS[m] for m in models]
        x = np.arange(len(models))
        ax.bar(x, means, yerr=stds, color=colors, width=0.6, capsize=4)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=20, ha='right')
        ax.set_title(metric)
        _strip_axes(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_paired_gain(per_seed: pd.DataFrame, out_path: str, metric: str = 'VUS_ROC') -> None:
    pivoted = per_seed.pivot(index='seed', columns='model', values=metric)
    fig, ax = plt.subplots(figsize=(5, 4))
    for seed in pivoted.index:
        without_e = pivoted.loc[seed, 'Cross_without_E']
        without_d = pivoted.loc[seed, 'Cross_without_D']
        ax.plot([0, 1], [without_e, without_d], color='#9aa0a6', linewidth=1, zorder=1)
    ax.scatter(np.zeros(len(pivoted)), pivoted['Cross_without_E'],
               color=MODEL_COLORS['Cross_without_E'], zorder=2, label='Cross_without_E', s=48)
    ax.scatter(np.ones(len(pivoted)), pivoted['Cross_without_D'],
               color=MODEL_COLORS['Cross_without_D'], zorder=2, label='Cross_without_D', s=48)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['without_E\n(D in pool)', 'without_D\n(E in pool)'])
    ax.set_ylabel(metric)
    ax.set_title(f'Per-seed paired comparison: {metric}')
    ax.legend(frameon=False)
    _strip_axes(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def write_source_pointers(results_root: str, core_clustering_outputs_dir: str, seeds) -> None:
    mapping = {
        'Self_A': 'experiment3_self_A',
        'Cross_without_E': 'experiment3_without_E',
        'Cross_without_D': 'experiment3_without_D',
    }
    descriptions = {
        'Self_A': "Trained ONLY on domain A (square)'s own fixed train/val instances "
                  "(every other domain held out) -- Core-Clustering online_cli.py, "
                  "--held_out_domains set to every domain except square.",
        'Cross_without_E': "Trained on every domain EXCEPT A (square) and E (smoothed_pulse) -- "
                            "never saw A or E.",
        'Cross_without_D': "Trained on every domain EXCEPT A (square) and D (white_noise) -- "
                            "never saw A or D (includes E/smoothed_pulse instead).",
    }
    for model_name, run_name in mapping.items():
        for seed in seeds:
            out_dir = os.path.join(results_root, 'Models', model_name, str(seed))
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, 'SOURCE.txt'), 'w') as f:
                f.write(
                    f'Originally: {os.path.join(core_clustering_outputs_dir, run_name, str(seed))} '
                    f'(Core-Clustering repo, not RedLamp_Check)\n'
                    f'{descriptions[model_name]}\n'
                )


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results_root', default='./result/Experiment_3',
                         help='Root under which --results_dir/--figures_dir/Models/ all default -- '
                              'also where SOURCE.txt pointers are written.')
    parser.add_argument('--results_dir', default=None, help='Default: {results_root}/Results')
    parser.add_argument('--figures_dir', default=None, help='Default: {results_root}/figures')
    parser.add_argument('--core_clustering_outputs_dir', default='../Core-Clustering/outputs')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    args = parser.parse_args()
    args.results_dir = args.results_dir or os.path.join(args.results_root, 'Results')
    args.figures_dir = args.figures_dir or os.path.join(args.results_root, 'figures')

    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.figures_dir, exist_ok=True)

    raw_df = load_all_scores(args.results_dir)
    raw_df.to_csv(os.path.join(args.results_dir, 'experiment3_results_raw.csv'), index=False)

    per_seed, summary = aggregate(raw_df)
    per_seed.to_csv(os.path.join(args.results_dir, 'experiment3_results_per_seed.csv'), index=False)
    summary.to_csv(os.path.join(args.results_dir, 'experiment3_results_summary.csv'), index=False)

    if {'Cross_without_E', 'Cross_without_D'} <= set(per_seed['model'].unique()):
        gain_df = paired_similar_domain_gain(per_seed)
        gain_df.to_csv(os.path.join(args.results_dir, 'experiment3_similar_domain_gain.csv'), index=False)
        print('similar_domain_gain (Cross_without_D - Cross_without_E), mean over seeds:')
        print(gain_df.drop(columns='seed').mean())

        for metric in ('VUS_ROC', 'RF', 'classification_accuracy'):
            plot_paired_gain(per_seed, os.path.join(args.figures_dir, f'paired_gain_{metric}.png'), metric=metric)

    plot_model_comparison(summary, os.path.join(args.figures_dir, 'model_comparison.png'))
    write_source_pointers(args.results_root, args.core_clustering_outputs_dir, args.seeds)

    print(summary.to_string(index=False))
    print(f'Done. Wrote results to {args.results_dir}, figures to {args.figures_dir}, '
          f'and SOURCE.txt pointers under {os.path.join(args.results_root, "Models")}')


if __name__ == '__main__':
    run()
