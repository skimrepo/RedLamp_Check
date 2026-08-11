"""
Reads score_self_data_efficiency.py's self_data_efficiency_overall_summary.csv
(one row per n_pct, mean/std of each metric across ALL scored UCR entities)
and plots one line per metric: x=n_pct (log scale, since values span
1-100), y=score, with a shaded +/-1 std band across entities. One combined
figure (2x3 grid: 5 metrics + peak_in_range) by default; --separate writes
one PNG per metric instead.
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

METRIC_KEYS = ['VUS_ROC', 'VUS_PR', 'R_AUC_ROC', 'R_AUC_PR', 'RF']
LINE_COLOR = '#4C78A8'
PEAK_COLOR = '#E45756'


def fmt_pct(v):
    return str(int(v)) if float(v).is_integer() else str(v)


def plot_axis(ax, x, y, std, ylabel, color):
    ax.plot(x, y, marker='o', color=color)
    ax.fill_between(x, (y - std).clip(lower=0), (y + std).clip(upper=1), alpha=0.2, color=color)
    ax.set_xscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels([fmt_pct(v) for v in x])
    ax.set_xlabel('Training data used (%)')
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1)
    ax.set_title(ylabel)
    ax.grid(True, alpha=0.3)


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--summary_csv', default='./result/DS_2/achievability/self_data_efficiency_overall_summary.csv')
    parser.add_argument('--out_dir', default='./result/DS_2/achievability')
    parser.add_argument('--separate', action='store_true',
                         help='Write one PNG per metric instead of one combined figure')
    args = parser.parse_args()

    df = pd.read_csv(args.summary_csv).sort_values('n_pct')
    x = df['n_pct']

    if args.separate:
        for metric in METRIC_KEYS:
            fig, ax = plt.subplots(figsize=(6, 4.5))
            plot_axis(ax, x, df[f'{metric}_mean'], df[f'{metric}_std'].fillna(0), metric, LINE_COLOR)
            fig.tight_layout()
            out_path = os.path.join(args.out_dir, f'self_data_efficiency_{metric}.png')
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            print(f'Wrote {out_path}')
        fig, ax = plt.subplots(figsize=(6, 4.5))
        plot_axis(ax, x, df['peak_in_range_mean'], pd.Series([0] * len(x)), 'peak_in_range', PEAK_COLOR)
        fig.tight_layout()
        out_path = os.path.join(args.out_dir, 'self_data_efficiency_peak_in_range.png')
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f'Wrote {out_path}')
    else:
        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        for ax, metric in zip(axes.flat, METRIC_KEYS):
            plot_axis(ax, x, df[f'{metric}_mean'], df[f'{metric}_std'].fillna(0), metric, LINE_COLOR)
        plot_axis(axes.flat[5], x, df['peak_in_range_mean'], pd.Series([0] * len(x)), 'peak_in_range', PEAK_COLOR)
        fig.suptitle('Self data-efficiency: score vs. % of training data used (UCR anomaly_archive, all scored entities)')
        fig.tight_layout()
        out_path = os.path.join(args.out_dir, 'self_data_efficiency_summary.png')
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f'Wrote {out_path}')


if __name__ == '__main__':
    run()
