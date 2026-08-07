"""
DS_3: one PDF with EVERY AnomSim_v1 entity's raw waveform, one page each --
a quick "what does this dataset actually look like" overview, distinct from
build_anomsim_train_val_diagnostics.py's per-entity injected-anomaly
diagnostic PDFs.

Each entity's page is a 5x2 grid (local_diagnostic_curves.
plot_entity_gallery_page): the top row spans both columns and shows the
WHOLE entity waveform; the 4 rows below show 8 evenly-spaced example
windows sampled from within that same series, so you can see what an
actual window_size-length model input from this entity looks like without
having to squint at the full series.

AnomSim_v1 entities have no ground-truth anomaly labels at all (just
Y.npy, no companion labels/mask file -- confirmed by inspecting the
dataset directory), so there's nothing to highlight here, just the raw
signal. No model, no caching, no sharding -- 144 entities of plain data
loading + plotting is cheap enough for a single process.
"""
import argparse
import os
import sys

import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORE_CLUSTERING_DEFAULT = os.path.join(REPO_ROOT, '..', 'Core-Clustering')
WINDOW_SIZE = 100


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', default=os.path.join(REPO_ROOT, '..', 'AnomSim', 'data', 'AnomSim_v1'))
    parser.add_argument('--core_clustering_dir', default=CORE_CLUSTERING_DEFAULT)
    parser.add_argument('--out_pdf', default='./result/DS_3/entity_galleries/AnomSim_all_entities.pdf')
    args = parser.parse_args()

    sys.path.insert(0, args.core_clustering_dir)
    sys.path.insert(0, REPO_ROOT)
    from core_clustering.single_entity import list_entities
    import local_diagnostic_curves as ldc

    entities = list_entities(args.dataset_dir)
    print(f'{len(entities)} AnomSim_v1 entities to plot')

    os.makedirs(os.path.dirname(args.out_pdf), exist_ok=True)
    with PdfPages(args.out_pdf) as pdf:
        for entity_dir in entities:
            y_path = os.path.join(args.dataset_dir, entity_dir, 'Y.npy')
            if not os.path.isfile(y_path):
                print(f'[skip] {entity_dir}: no Y.npy found')
                continue
            Y = np.load(y_path)
            ldc.plot_entity_gallery_page(pdf, Y[0], WINDOW_SIZE, title=entity_dir)

    print(f'Wrote {args.out_pdf} ({len(entities)} pages)')


if __name__ == '__main__':
    run()
