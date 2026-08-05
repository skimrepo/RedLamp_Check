"""
DS_3 step 1 (local half): turns self_convergence.csv (from
analyze_training_convergence.py, run on the server) plus Cross-AnomSim's
own run_summary.json (already local -- Core-Clustering's Trainer writes
final per-epoch loss values directly into it, no extraction needed) into
the convergence comparison plots.

Plot A: distribution of train_mse/val_mse/train_ce/val_ce across all UCR
Self entities, with Cross-AnomSim's single value marked as a vertical line
-- shows whether Cross-AnomSim's convergence sits inside or outside the
typical range Self entities converge to.

Plot B: per-entity train-vs-val scatter (MSE and CE separately) with a y=x
reference line -- points far above the line are overfitting (val loss much
worse than train loss).

Also writes a small summary CSV (mean/median/std for Self's 4 metrics vs
Cross-AnomSim's single values).
"""
import json
import os

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SELF_CSV = os.path.join(REPO_ROOT, 'result', 'DS_3', 'convergence', 'self_convergence.csv')
CROSS_ANOMSIM_SUMMARY = os.path.join(REPO_ROOT, 'result', 'result', 'Experiment_1', 'Models',
                                      'Cross-AnomSim', '0', 'run_summary.json')
OUT_DIR = os.path.join(REPO_ROOT, 'result', 'DS_3', 'convergence')


def load_cross_anomsim_point():
    with open(CROSS_ANOMSIM_SUMMARY) as f:
        summary = json.load(f)
    return dict(
        train_mse=summary['epochs'][summary['best_epoch']]['train_loss_ae'],
        val_mse=summary['best_val_loss_ae'],
        train_ce=summary['epochs'][summary['best_epoch']]['train_loss_c'],
        val_ce=summary['best_val_loss_c'],
    )


def plot_distributions(df, cross_point, save_path):
    metrics = ['train_mse', 'val_mse', 'train_ce', 'val_ce']
    titles = ['Train MSE', 'Val MSE', 'Train Classification Loss', 'Val Classification Loss']

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, metric, title in zip(axes.flat, metrics, titles):
        ax.hist(df[metric], bins=40, color='#3f7fbf', alpha=0.75, edgecolor='black', linewidth=0.3)
        ax.axvline(cross_point[metric], color='#d1495b', linewidth=2, linestyle='--',
                   label=f'Cross-AnomSim ({cross_point[metric]:.4f})')
        ax.set_title(f'{title} (Self, n={len(df)})')
        ax.legend(fontsize=8)
    fig.suptitle('Self (all UCR entities) vs Cross-AnomSim -- convergence at best epoch')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f'Wrote {save_path}')


def plot_train_vs_val(df, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, train_col, val_col, title in [(axes[0], 'train_mse', 'val_mse', 'MSE'),
                                           (axes[1], 'train_ce', 'val_ce', 'Classification Loss')]:
        ax.scatter(df[train_col], df[val_col], s=18, alpha=0.6, color='#3f7fbf', edgecolors='none')
        lims = [min(df[train_col].min(), df[val_col].min()), max(df[train_col].max(), df[val_col].max())]
        ax.plot(lims, lims, linestyle='--', color='#888888', linewidth=1, label='y = x (no overfit)')
        ax.set_xlabel(f'train {title}')
        ax.set_ylabel(f'val {title}')
        ax.set_title(f'{title}: train vs val (Self, n={len(df)})')
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f'Wrote {save_path}')


def run():
    df = pd.read_csv(SELF_CSV)
    df['entity'] = df['entity'].astype(str).str.zfill(3)
    cross_point = load_cross_anomsim_point()

    os.makedirs(OUT_DIR, exist_ok=True)
    plot_distributions(df, cross_point, os.path.join(OUT_DIR, 'convergence_distributions.png'))
    plot_train_vs_val(df, os.path.join(OUT_DIR, 'train_vs_val_scatter.png'))

    summary = df[['train_mse', 'val_mse', 'train_ce', 'val_ce']].agg(['mean', 'median', 'std'])
    for metric, value in cross_point.items():
        summary.loc['cross_anomsim', metric] = value
    summary_path = os.path.join(OUT_DIR, 'convergence_summary.csv')
    summary.round(4).to_csv(summary_path)
    print(f'Wrote {summary_path}')
    print(summary)


if __name__ == '__main__':
    run()
