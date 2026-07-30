"""
Organize Experiment 0 - Exploratory: moves the earlier, smaller-scale
experiments (built before the full 250-UCR/29-AIOps reproduction) out of
`result/{run_name}/...` into a separate `Experiment 0 - Exploratory/` folder,
to keep them clearly distinct from Experiment 1's paper-reproduction check.

Pure directory/file relocation — no data processing, no CSV regeneration.
Safe to rerun: entries whose source no longer exists (already moved, or never
produced) are skipped.

Note: continuous_n697_excl_ucr and continuous_n944 are NOT moved here even
though they're part of continuous_pool_scaling.py's own n-scaling curve —
run organize_experiment1.py FIRST, which claims those two specifically for
Experiment 1's Cross-OpenSource models.
"""
import argparse
import os
import shutil

# (source path relative to result/{run_name}, destination path relative to exp_dir)
MOVES = [
    ('_cross_domain', 'domain_generalization'),
    ('_pooled', 'continuous_pool_scaling/models'),  # continuous_n697_excl_ucr/n944 already claimed by Experiment 1
    ('_cross_domain_holdout', 'continuous_pool_scaling/results'),
    ('test_set_metrics.csv', 'test_set_anomaly_metrics/test_set_metrics.csv'),
    ('self_accuracy_all_datasets.csv', 'self_accuracy_report/self_accuracy_all_datasets.csv'),
]
CROSS_INFERENCE_DATASETS = ['anomaly_archive', 'iops', 'smd', 'smap', 'msl']


def move_if_exists(src, dest):
    if not os.path.exists(src):
        print(f'[skip] {src} does not exist (already moved, or never produced)')
        return
    if os.path.exists(dest):
        print(f'[skip] {dest} already exists — not overwriting')
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    shutil.move(src, dest)
    print(f'[moved] {src} -> {dest}')


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', default='test')
    parser.add_argument('--exp_dir', default='./Experiment 0 - Exploratory')
    args = parser.parse_args()

    base = f'./result/{args.run_name}'
    for rel_src, rel_dest in MOVES:
        move_if_exists(os.path.join(base, rel_src), os.path.join(args.exp_dir, rel_dest))

    for dataset in CROSS_INFERENCE_DATASETS:
        move_if_exists(os.path.join(base, dataset, '_cross_inference'),
                        os.path.join(args.exp_dir, 'cross_inference', dataset))

    print(f'Done. Experiment 0 - Exploratory organized at {args.exp_dir}')


if __name__ == '__main__':
    run()
