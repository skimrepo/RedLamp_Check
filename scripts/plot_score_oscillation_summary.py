"""
Turns result/DS_2/oscillation/oscillation_metrics.csv (from
analyze_score_oscillation.py, run on the server) into the two plots that
test whether Cross-AnomSim's imprecision is a universal limitation (present
similarly across both exp2_bad and exp2_good entities) rather than something
specific to "bad" entities -- see the plan's prediction.

Plot A (oscillation_by_group.png): per-group (exp2_bad/exp2_good) box plots
of self_score_std_normal vs cross_anomsim_score_std_normal -- if
cross_anomsim's normal-period score oscillation is similarly elevated in
BOTH groups (not uniquely worse in "bad"), that supports it being a
universal property rather than the thing that differentiates bad from good.

Plot B (correlation_with_local_activity.png): paired per-entity comparison
of self_corr_with_activity vs cross_anomsim_corr_with_activity, colored by
group -- tests whether cross_anomsim's score tracks raw-signal local
activity more than Self's does, and whether that gap is consistent across
both groups.

Runs entirely locally once oscillation_metrics.csv is downloaded -- no
server/model needed, just pandas + matplotlib.
"""
import os

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OSCILLATION_CSV = os.path.join(REPO_ROOT, 'result', 'DS_2', 'oscillation', 'oscillation_metrics.csv')
DS1_METADATA = os.path.join(REPO_ROOT, 'result', 'DS_1', 'entity_metadata.csv')
OUT_DIR = os.path.join(REPO_ROOT, 'result', 'DS_2', 'oscillation')


def _load_merged():
    osc = pd.read_csv(OSCILLATION_CSV)
    osc['entity'] = osc['entity'].astype(str).str.zfill(3)

    meta = pd.read_csv(DS1_METADATA)
    meta['entity'] = meta['entity'].astype(str).str.zfill(3)

    merged = osc.merge(meta[['entity', 'groups']], on='entity', how='left')

    def group_of(g):
        g = g or ''
        if 'exp2_bad' in g:
            return 'exp2_bad'
        if 'exp2_good' in g:
            return 'exp2_good'
        return 'other'

    merged['group'] = merged['groups'].apply(group_of)
    return merged[merged['group'] != 'other'].copy()


def plot_oscillation_by_group(df, save_path):
    fig, ax = plt.subplots(figsize=(7, 5))
    groups = ['exp2_bad', 'exp2_good']
    positions, labels, data = [], [], []
    for i, group in enumerate(groups):
        sub = df[df['group'] == group]
        for j, model in enumerate(['self', 'cross_anomsim']):
            data.append(sub[f'{model}_score_std_normal'].dropna().values)
            positions.append(i * 3 + j)
            labels.append(f'{group}\n{model}')

    bp = ax.boxplot(data, positions=positions, widths=0.7, patch_artist=True)
    colors = ['#3f7fbf', '#3fae59'] * 2
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('score std during normal-labeled periods\n("oscillation")')
    ax.set_title('Normal-period score oscillation: Self vs Cross-AnomSim, by group')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f'Wrote {save_path}')


def plot_correlation_with_activity(df, save_path):
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    colors = {'exp2_bad': '#d1495b', 'exp2_good': '#3f8f5f'}
    for group, color in colors.items():
        sub = df[df['group'] == group].dropna(subset=['self_corr_with_activity', 'cross_anomsim_corr_with_activity'])
        ax.scatter(sub['self_corr_with_activity'], sub['cross_anomsim_corr_with_activity'],
                   color=color, label=f'{group} (n={len(sub)})', s=45, alpha=0.85, edgecolors='black', linewidths=0.5)

    lims = [-1, 1]
    ax.plot(lims, lims, linestyle='--', color='#888888', linewidth=1, label='y = x (same for both models)')
    ax.axhline(0, color='#dddddd', linewidth=0.8, zorder=0)
    ax.axvline(0, color='#dddddd', linewidth=0.8, zorder=0)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Self score's correlation with local raw-signal activity")
    ax.set_ylabel("Cross-AnomSim score's correlation with local raw-signal activity")
    ax.set_title('Does each model\'s score just react to local signal activity?\n(points above y=x: Cross-AnomSim reacts more than Self)')
    ax.legend(fontsize=8, loc='lower right')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f'Wrote {save_path}')


def run():
    df = _load_merged()
    os.makedirs(OUT_DIR, exist_ok=True)
    plot_oscillation_by_group(df, os.path.join(OUT_DIR, 'oscillation_by_group.png'))
    plot_correlation_with_activity(df, os.path.join(OUT_DIR, 'correlation_with_local_activity.png'))

    summary = df.groupby('group')[['self_score_std_normal', 'cross_anomsim_score_std_normal',
                                    'self_corr_with_activity', 'cross_anomsim_corr_with_activity']].mean()
    print(summary)


if __name__ == '__main__':
    run()
