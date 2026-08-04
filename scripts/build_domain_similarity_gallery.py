"""
Compact visual gallery for eyeballing domain similarity: "did the
model that's supposed to generalize to this real entity ever see anything
that looks like it?" Composites already-generated normal/anomaly sample
PNGs (no new inference) into a small number of dense, labeled-row pages:

  Page 1: UCR entities where Cross-AnomSim badly underperforms Self
          (exp2_bad, all of them) + a representative handful of UCR
          entities where the gap is small/reversed (exp2_good) -- both
          legs' images come from DS_1's own examples/{normal,spike}_1.png
          (scripts/analyze_ds1_gap_entities.py).
  Page 2: what Cross-OpenSource actually trained on (SMD/SMAP/MSL, one
          entity per dataset -- images from DS_0's {dataset}/{entity}/
          {waveform,anomaly_example}.png) and what Cross-AnomSim actually
          trained on (all 9 AnomSim_v1 waveform types -- images from the
          sibling Core-Clustering repo's scripts/
          plot_anomsim_waveform_samples.py output).

Every source image already exists locally (no server/model needed) --
this script only arranges pre-rendered PNGs into a grid via imshow, it
does not re-plot any underlying data.
"""
import os

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DS1_DIR = os.path.join(REPO_ROOT, 'result', 'DS_1')
DS0_DIR = os.path.join(REPO_ROOT, 'result', 'DS_0')
ANOMSIM_SAMPLES_DIR = os.path.join(REPO_ROOT, '..', 'Core-Clustering', 'outputs', 'anomsim_domain_samples')

BAD_ENTITIES = ['034', '040', '042', '044', '046', '052', '101', '127',
                '134', '148', '150', '154', '160', '202', '205']
GOOD_ENTITIES = ['214', '165', '061', '210', '227']

CROSS_OPENSOURCE = [('SMD', 'machine-1-1'), ('SMAP', 'smap'), ('MSL', 'msl')]


def _entity_categories():
    meta = pd.read_csv(os.path.join(DS1_DIR, 'entity_metadata.csv'))
    meta['entity'] = meta['entity'].astype(str).str.zfill(3)
    return dict(zip(meta['entity'], meta['category']))


def _row(label, normal_path, anomaly_path):
    return dict(label=label, normal_path=normal_path, anomaly_path=anomaly_path)


def _ucr_rows(entities, categories, prefix):
    rows = []
    for e in entities:
        cat = categories.get(e, '?')
        rows.append(_row(
            f'{prefix} {e}\n{cat}',
            os.path.join(DS1_DIR, 'plots', e, 'examples', 'normal_1.png'),
            os.path.join(DS1_DIR, 'plots', e, 'examples', 'spike_1.png'),
        ))
    return rows


def _cross_opensource_rows():
    rows = []
    for dataset_label, entity in CROSS_OPENSOURCE:
        dataset_dir = {'SMD': 'smd', 'SMAP': 'smap', 'MSL': 'msl'}[dataset_label]
        rows.append(_row(
            f'Cross-OpenSource\n{dataset_label}: {entity}',
            os.path.join(DS0_DIR, dataset_dir, entity, 'waveform.png'),
            os.path.join(DS0_DIR, dataset_dir, entity, 'anomaly_example.png'),
        ))
    return rows


def _cross_anomsim_rows():
    rows = []
    for wf_type in sorted(os.listdir(ANOMSIM_SAMPLES_DIR)):
        type_dir = os.path.join(ANOMSIM_SAMPLES_DIR, wf_type)
        if not os.path.isdir(type_dir):
            continue
        rows.append(_row(
            f'Cross-AnomSim\n{wf_type}',
            os.path.join(type_dir, 'normal.png'),
            os.path.join(type_dir, 'spike.png'),
        ))
    return rows


def render_page(rows, save_path, title):
    n = len(rows)
    fig, axes = plt.subplots(n, 2, figsize=(9, 1.15 * n))
    if n == 1:
        axes = axes.reshape(1, 2)

    for i, row in enumerate(rows):
        for j, (col_title, path) in enumerate([('normal', row['normal_path']), ('anomaly (spike)', row['anomaly_path'])]):
            ax = axes[i, j]
            if os.path.isfile(path):
                ax.imshow(mpimg.imread(path), aspect='auto')
            else:
                ax.text(0.5, 0.5, 'missing', ha='center', va='center', fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if i == 0:
                ax.set_title(col_title, fontsize=9)
        axes[i, 0].set_ylabel(row['label'], fontsize=7, rotation=0, ha='right', va='center', labelpad=40)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(save_path, dpi=130)
    plt.close(fig)
    print(f'Wrote {save_path} ({n} rows)')


def run():
    categories = _entity_categories()
    out_dir = os.path.join(REPO_ROOT, 'result', 'DS_2', 'plots')
    os.makedirs(out_dir, exist_ok=True)

    page1_rows = (_ucr_rows(BAD_ENTITIES, categories, 'BAD (exp2_bad)')
                  + _ucr_rows(GOOD_ENTITIES, categories, 'GOOD (exp2_good)'))
    render_page(page1_rows, os.path.join(out_dir, 'domain_similarity_page1_ucr.png'),
                'UCR entities: normal vs. injected-spike sample (bad Cross-AnomSim cases first, then good)')

    page2_rows = _cross_opensource_rows() + _cross_anomsim_rows()
    render_page(page2_rows, os.path.join(out_dir, 'domain_similarity_page2_cross_models.png'),
                'What Cross-OpenSource / Cross-AnomSim actually trained on')


if __name__ == '__main__':
    run()
