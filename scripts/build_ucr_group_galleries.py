"""
DS_3: 3 PDFs, one per Self-vs-Cross-AnomSim performance group -- Red(BAD),
Green(GOOD), Other -- same grouping convention as
plot_self_vs_cross_anomsim_scatter.py: gap = VUS_ROC_self -
VUS_ROC_cross_anomsim (from result/Experiment_2/Results/ucr_results.xlsx's
'Per-Entity Comparison' sheet), Red = gap >= --gap_threshold (Cross-AnomSim
struggles relative to Self), Green = gap <= 0 (Cross-AnomSim keeps up with
or beats Self), Other = everything in between.

Each PDF has one page per entity in that group: the entity's real UCR TEST
signal with the real ground-truth anomaly region shaded. Test-only, not
train -- UCR's real anomaly is always confined to the test portion
(loaders/load.py's load_anomaly_archive never builds a label array for
group='train'), so a train-portion plot would have nothing to highlight
and add little value.

No model, no caching, no sharding needed -- pure data loading + plotting,
cheap enough for a single process even across ~247 entities.
"""
import argparse
import os
import sys

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from loaders.load import load_anomaly_archive
import local_diagnostic_curves as ldc

DATASET = 'anomaly_archive'
GROUP_PREDICATES = [
    ('Red_BAD', lambda gap, thr: gap >= thr),
    ('Green_GOOD', lambda gap, thr: gap <= 0),
    ('Other', lambda gap, thr: (gap > 0) & (gap < thr)),
]


def classify_entities(ucr_xlsx, gap_threshold):
    """Same gap/threshold convention as plot_self_vs_cross_anomsim_scatter.py
    -- returns dict group_name -> [(entity, gap), ...]."""
    per_entity = pd.read_excel(ucr_xlsx, sheet_name='Per-Entity Comparison')
    per_entity['entity'] = per_entity['entity'].astype(str).str.zfill(3)
    per_entity = per_entity.dropna(subset=['VUS_ROC_self', 'VUS_ROC_cross_anomsim'])
    per_entity['gap'] = per_entity['VUS_ROC_self'] - per_entity['VUS_ROC_cross_anomsim']

    groups = {}
    for name, predicate in GROUP_PREDICATES:
        sub = per_entity[predicate(per_entity['gap'], gap_threshold)]
        groups[name] = list(zip(sub['entity'], sub['gap']))
    return groups


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ucr_xlsx', default='./result/Experiment_2/Results/ucr_results.xlsx')
    parser.add_argument('--gap_threshold', type=float, default=0.5,
                         help='"Red/BAD": gap >= this. "Green/GOOD": gap <= 0. Everything in between is "Other".')
    parser.add_argument('--out_dir', default='./result/DS_3/entity_galleries')
    args = parser.parse_args()

    groups = classify_entities(args.ucr_xlsx, args.gap_threshold)
    for name, entities in groups.items():
        print(f'{name}: {len(entities)} entities')

    os.makedirs(args.out_dir, exist_ok=True)
    for name, entities in groups.items():
        out_path = os.path.join(args.out_dir, f'UCR_{name}_test.pdf')
        with PdfPages(out_path) as pdf:
            n_written = 0
            for entity, gap in entities:
                try:
                    test_ds = load_anomaly_archive(group='test', datasets=entity, downsampling=1,
                                                    root_dir='./dataset', validation=False, verbose=False)
                except Exception as e:
                    print(f'[skip] {entity}: failed to load ({e})')
                    continue
                raw = test_ds.entities[0].Y[0]
                labels = test_ds.entities[0].labels.reshape(-1)
                spans = ldc.find_anomaly_segments(labels, max_segments=5)

                fig, ax = plt.subplots(figsize=(11, 3))
                ax.plot(raw, color='#333333', linewidth=0.8)
                for span_start, span_end in spans:
                    ax.axvspan(span_start, span_end, color='#e34948', alpha=0.15)
                ax.set_title(f'{entity} (gap={gap:.3f})', fontsize=10)
                ax.set_xlabel('timestep')
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)
                n_written += 1
        print(f'Wrote {out_path} ({n_written} pages)')

    print('Done.')


if __name__ == '__main__':
    run()
