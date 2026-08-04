"""
Self vs Cross-AnomSim VUS_ROC scatter across ALL UCR entities (not just
DS_1's gap-selected 46) -- directly visualizes the "achievability" finding:
the Self/Cross-AnomSim gap only opens up wide when Self itself is
near-perfect, not because Cross-AnomSim is uniformly worse everywhere.

Points below the y=x line: Cross-AnomSim underperforms Self (positive gap).
Points above: Cross-AnomSim keeps up with or beats Self.

Runs entirely from data already on disk locally -- no server, no model,
just Experiment_2/Results/ucr_results.xlsx's Per-Entity Comparison sheet
(already downloaded) and DS_1/entity_metadata.csv (for bad/good highlighting).
"""
import os

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UCR_XLSX = os.path.join(REPO_ROOT, 'result', 'Experiment_2', 'Results', 'ucr_results.xlsx')
DS1_METADATA = os.path.join(REPO_ROOT, 'result', 'DS_1', 'entity_metadata.csv')
OUT_PATH = os.path.join(REPO_ROOT, 'result', 'DS_2', 'achievability', 'self_vs_cross_anomsim_vus_roc_scatter.png')


def run():
    per_entity = pd.read_excel(UCR_XLSX, sheet_name='Per-Entity Comparison')
    per_entity['entity'] = per_entity['entity'].astype(str).str.zfill(3)
    per_entity = per_entity.dropna(subset=['VUS_ROC_self', 'VUS_ROC_cross_anomsim'])

    meta = pd.read_csv(DS1_METADATA)
    meta['entity'] = meta['entity'].astype(str).str.zfill(3)
    groups = dict(zip(meta['entity'], meta['groups']))

    def highlight(entity):
        g = groups.get(entity, '')
        if 'exp2_bad' in g:
            return 'exp2_bad'
        if 'exp2_good' in g:
            return 'exp2_good'
        return 'other'

    per_entity['highlight'] = per_entity['entity'].apply(highlight)

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
