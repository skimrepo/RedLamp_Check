"""
Self vs Cross-AnomSim VUS_ROC scatter across ALL UCR entities -- directly
visualizes the "achievability" finding: the Self/Cross-AnomSim gap only
opens up wide when Self itself is near-perfect, not because Cross-AnomSim
is uniformly worse everywhere.

Points below the y=x line: Cross-AnomSim underperforms Self (positive gap).
Points above: Cross-AnomSim keeps up with or beats Self.

bad/good labeling is computed directly here from Experiment_2's own
VUS_ROC gap (gap = VUS_ROC_self - VUS_ROC_cross_anomsim), same threshold
convention as analyze_ds1_gap_entities.py (--gap_threshold, default 0.5;
good = gap <= 0) but with NO top-n cap and NO dependency on
Experiment_1/DS_1's accuracy-based grouping -- every one of the ~247 UCR
entities gets labeled bad/good/other on its own Experiment_2 merits.

Runs entirely from data already on disk locally -- no server, no model,
just Experiment_2/Results/ucr_results.xlsx's Per-Entity Comparison sheet
(already downloaded).
"""
import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UCR_XLSX = os.path.join(REPO_ROOT, 'result', 'Experiment_2', 'Results', 'ucr_results.xlsx')
OUT_PATH = os.path.join(REPO_ROOT, 'result', 'DS_2', 'achievability', 'self_vs_cross_anomsim_vus_roc_scatter.png')


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gap_threshold', type=float, default=0.5,
                         help='"bad": gap (VUS_ROC_self - VUS_ROC_cross_anomsim) >= this. '
                              '"good": gap <= 0. Everything in between is "other".')
    args = parser.parse_args()

    per_entity = pd.read_excel(UCR_XLSX, sheet_name='Per-Entity Comparison')
    per_entity['entity'] = per_entity['entity'].astype(str).str.zfill(3)
    per_entity = per_entity.dropna(subset=['VUS_ROC_self', 'VUS_ROC_cross_anomsim'])
    per_entity['gap'] = per_entity['VUS_ROC_self'] - per_entity['VUS_ROC_cross_anomsim']

    def highlight(gap):
        if gap >= args.gap_threshold:
            return 'exp2_bad'
        if gap <= 0:
            return 'exp2_good'
        return 'other'

    per_entity['highlight'] = per_entity['gap'].apply(highlight)

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    colors = {'other': '#c9c9c9', 'exp2_bad': '#d1495b', 'exp2_good': '#3f8f5f'}
    order = ['other', 'exp2_bad', 'exp2_good']
    for key in order:
        sub = per_entity[per_entity['highlight'] == key]
        ax.scatter(sub['VUS_ROC_self'], sub['VUS_ROC_cross_anomsim'],
                   s=22 if key == 'other' else 40, alpha=0.55 if key == 'other' else 0.9,
                   color=colors[key], label=f'{key} (n={len(sub)})',
                   edgecolors='none' if key == 'other' else 'black', linewidths=0.5)

    ax.plot([0, 1], [0, 1], linestyle='--', color='#888888', linewidth=1, label='y = x (no gap)')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Self VUS_ROC')
    ax.set_ylabel('Cross-AnomSim VUS_ROC')
    ax.set_title(f'Self vs Cross-AnomSim VUS_ROC, all UCR entities (n={len(per_entity)})')
    ax.legend(fontsize=8, loc='lower right')
    fig.tight_layout()

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150)
    plt.close(fig)
    print(f'Wrote {OUT_PATH} ({len(per_entity)} entities)')


if __name__ == '__main__':
    run()
