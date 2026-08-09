"""
Experiment 3: visual comparison of the AnomSim_v2 domains + a look at the
actual A(square) fixed test entities that get scored, reusing the entity
gallery template (see docs/plot_templates_guide.md, template 1 --
local_diagnostic_curves.plot_entity_gallery_page) exactly as
build_anomsim_entity_gallery.py/build_ucr_group_galleries.py already do.
Pure data visualization -- no model, no scores.

Writes two PDFs:

1. --out_domains (default .../experiment3_domain_gallery.pdf): ONE page per
   AnomSim domain (10 pages for AnomSim_v2: the 9 original + smoothed_pulse),
   picking a single representative instance per domain (lowest
   base_instance_id) from --dataset_dir's manifest -- lets you compare
   sine/square/smoothed_pulse/white_noise/etc.'s whole-series shape and an
   example window side by side, page by page. No anomaly highlight (plain
   structural comparison, same as build_anomsim_entity_gallery.py).

2. --out_a_test (default .../experiment3_A_test_entities_gallery.pdf): ONE
   page per A(square)'s FIXED test instance -- the exact entities Experiment
   3 scores all three models against -- sourced from the already-cached
   labeled .npz files experiment3_score_anomsim_domain.py wrote
   (--cache_dir, `{domain}_b{id}_labeled_seed{injection_seed}_w{window_size}.npz`,
   self-contained: y_injected + real_labels, no need for the raw AnomSim_v2
   directory). The injected ground-truth anomaly spans are highlighted in
   red (find_anomaly_segments), so you can see exactly where/how large each
   test entity's anomaly region is -- this is what
   scripts/experiment3_score_anomsim_domain.py builds and every model is
   scored against, and only entities with the CURRENT chunked-injection fix
   (window_size-tagged cache files) are included -- an old whole-entity-
   injection cache file (no _w{size} suffix) is skipped with a warning.

Usage:
    python scripts/build_experiment3_domain_gallery.py \
        --dataset_dir ../AnomSim/data/AnomSim_v2 \
        --cache_dir ./result/Experiment_3/cache --domain square
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
import local_diagnostic_curves as ldc

WINDOW_SIZE = 100


def pick_representative_instances(dataset_dir, manifest_name='_manifest.jsonl'):
    """domain -> lowest-base_instance_id entity_dir, in first-seen domain order."""
    manifest_path = os.path.join(dataset_dir, manifest_name)
    best = {}
    order = []
    with open(manifest_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            meta = json.loads(line)
            domain = meta['type']
            if domain not in best or meta['base_instance_id'] < best[domain]['base_instance_id']:
                if domain not in best:
                    order.append(domain)
                best[domain] = meta
    return [(domain, best[domain]['entity_dir']) for domain in order]


def build_domain_gallery(dataset_dir, out_pdf, window_size):
    representatives = pick_representative_instances(dataset_dir)
    print(f'{len(representatives)} domains found: {[d for d, _ in representatives]}')
    os.makedirs(os.path.dirname(out_pdf), exist_ok=True)
    with PdfPages(out_pdf) as pdf:
        for domain, entity_dir in representatives:
            y_path = os.path.join(dataset_dir, entity_dir, 'Y.npy')
            if not os.path.isfile(y_path):
                print(f'[skip] {entity_dir}: no Y.npy found')
                continue
            Y = np.load(y_path)
            ldc.plot_entity_gallery_page(pdf, Y[0], window_size, title=f'{domain}  ({entity_dir})')
    print(f'Wrote {out_pdf} ({len(representatives)} pages)')


def build_a_test_gallery(cache_dir, domain, injection_seed, window_size, out_pdf):
    pattern = os.path.join(cache_dir, f'{domain}_b*_labeled_seed{injection_seed}_w{window_size}.npz')
    paths = sorted(glob.glob(pattern))
    if not paths:
        old_pattern = os.path.join(cache_dir, f'{domain}_b*_labeled_seed{injection_seed}.npz')
        if glob.glob(old_pattern):
            print(f'[warn] found cache files matching {old_pattern} but none matching the current '
                  f'(window_size-tagged) {pattern} -- those are from the old whole-entity injection '
                  f'bug; rerun experiment3_score_anomsim_domain.py first to regenerate them.')
        else:
            print(f'[warn] no cache files found matching {pattern}')
        return
    print(f'{len(paths)} fixed test entities found for domain {domain!r}')

    os.makedirs(os.path.dirname(out_pdf), exist_ok=True)
    with PdfPages(out_pdf) as pdf:
        for path in paths:
            data = np.load(path, allow_pickle=True)
            y_injected = np.asarray(data['y_injected'])[0]
            real_labels = np.asarray(data['real_labels'])
            anomaly_types = str(data['anomaly_types']).split(',') if 'anomaly_types' in data else []
            spans = ldc.find_anomaly_segments(real_labels, max_segments=10)
            base_id = os.path.basename(path).split('_')[1]  # 'b{id}'
            frac = real_labels.sum() / len(real_labels)
            title = f'{domain}_{base_id}  types={anomaly_types}  anomalous_frac={frac:.1%}'
            ldc.plot_entity_gallery_page(pdf, y_injected, window_size, title=title, real_anomaly_spans=spans)
    print(f'Wrote {out_pdf} ({len(paths)} pages)')


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset_dir', required=True, help='AnomSim_v2 base-pool directory')
    parser.add_argument('--cache_dir', default='./result/Experiment_3/cache')
    parser.add_argument('--domain', default='square', help='Target domain (A) for the test-entity gallery')
    parser.add_argument('--injection_seed', type=int, default=20260807)
    parser.add_argument('--window_size', type=int, default=WINDOW_SIZE)
    parser.add_argument('--out_domains', default='./result/Experiment_3/figures/experiment3_domain_gallery.pdf')
    parser.add_argument('--out_a_test', default='./result/Experiment_3/figures/experiment3_A_test_entities_gallery.pdf')
    args = parser.parse_args()

    build_domain_gallery(args.dataset_dir, args.out_domains, args.window_size)
    build_a_test_gallery(args.cache_dir, args.domain, args.injection_seed, args.window_size, args.out_a_test)


if __name__ == '__main__':
    run()
