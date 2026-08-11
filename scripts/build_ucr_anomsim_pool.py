"""
Builds a Core-Clustering-compatible "base pool" directory that merges
AnomSim_v1's 144 entities (symlinked, untouched -- AnomSim_v1 itself is
never modified) with a small explicit group of real UCR entities (only
their TRAIN portion -- the test portion holds the real anomaly and must
never enter training), so core_clustering.online_cli can be pointed at
ONE --dataset_dir covering both, completely unmodified. Each UCR entity
becomes its own new "domain" (--ucr_domain_name) in the pool with one
base instance per entity.

Also writes one --exclude_entity_dirs_file JSON per UCR entity (containing
just that one entity's dir name) under --exclude_files_dir, ready to hand
to online_cli.py's --exclude_entity_dirs_file for a leave-one-out fold --
core_clustering.online_dataset.load_base_pool already supports excluding
specific entity_dirs, so no Core-Clustering code needs to change for this.

Needs the real UCR dataset (./dataset/AnomalyArchive) and a local
AnomSim_v1 checkout -- meant to run on the machine that has both (the
server), not this development machine.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from loaders.load import load_anomaly_archive

DATASET = 'anomaly_archive'
DEFAULT_ENTITIES = ['044', '045', '046', '047', '152', '153', '154', '155']  # UCR PowerDemand1-4, DISTORTED + plain


def symlink_anomsim_entities(anomsim_dir, out_dir):
    manifest_path = os.path.join(anomsim_dir, '_manifest.jsonl')
    lines = []
    with open(manifest_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            meta = json.loads(line)
            entity_dir = meta['entity_dir']
            src = os.path.abspath(os.path.join(anomsim_dir, entity_dir))
            dst = os.path.join(out_dir, entity_dir)
            if not os.path.islink(dst) and not os.path.exists(dst):
                os.symlink(src, dst)
            lines.append(line)
    return lines


def add_ucr_entities(entities, ucr_domain_name, out_dir):
    manifest_lines = []
    entity_dirs = []
    for i, entity in enumerate(entities):
        train_ds = load_anomaly_archive(group='train', datasets=entity, downsampling=1,
                                         root_dir='./dataset', validation=False, verbose=False)
        Y = train_ds.entities[0].Y.astype(np.float64)  # (1, n_time), train portion only

        entity_dir = f'{ucr_domain_name}_{entity}'
        entity_path = os.path.join(out_dir, entity_dir)
        os.makedirs(entity_path, exist_ok=True)
        np.save(os.path.join(entity_path, 'Y.npy'), Y)

        meta = dict(type=ucr_domain_name, base_instance_id=i, base_seed=0,
                    n_time=int(Y.shape[1]), entity_dir=entity_dir, ucr_entity=entity)
        manifest_lines.append(json.dumps(meta))
        # core_clustering.single_entity.load_single_entity_split (used by
        # online_cli.py --single_entity) reads a per-entity meta.json, not
        # just the pool-level _manifest.jsonl -- AnomSim_v1's own entities
        # already ship one each, UCR entities need the same file written
        # here or --single_entity fails with FileNotFoundError.
        with open(os.path.join(entity_path, 'meta.json'), 'w') as f:
            json.dump(meta, f)
        entity_dirs.append(entity_dir)
        print(f'  added {entity_dir}: n_time={Y.shape[1]} (UCR entity {entity}, train portion)')
    return manifest_lines, entity_dirs


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--anomsim_dir', default='../AnomSim/data/AnomSim_v1')
    parser.add_argument('--entities', nargs='+', default=DEFAULT_ENTITIES)
    parser.add_argument('--ucr_domain_name', default='ucr_PowerDemand')
    parser.add_argument('--out_dir', default='./result/DS_2/achievability/anomsim_plus_ucr_powerdemand')
    parser.add_argument('--exclude_files_dir', default=None, help='Default: {out_dir}/exclude_files')
    args = parser.parse_args()
    args.exclude_files_dir = args.exclude_files_dir or os.path.join(args.out_dir, 'exclude_files')

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.exclude_files_dir, exist_ok=True)

    anomsim_lines = symlink_anomsim_entities(args.anomsim_dir, args.out_dir)
    print(f'Symlinked {len(anomsim_lines)} AnomSim_v1 entities into {args.out_dir}')

    ucr_lines, ucr_entity_dirs = add_ucr_entities(args.entities, args.ucr_domain_name, args.out_dir)

    manifest_path = os.path.join(args.out_dir, '_manifest.jsonl')
    with open(manifest_path, 'w') as f:
        f.write('\n'.join(anomsim_lines + ucr_lines) + '\n')
    print(f'Wrote {manifest_path} ({len(anomsim_lines) + len(ucr_lines)} total entities)')

    for entity_dir in ucr_entity_dirs:
        exclude_path = os.path.join(args.exclude_files_dir, f'exclude_{entity_dir}.json')
        with open(exclude_path, 'w') as f:
            json.dump([entity_dir], f)
    print(f'Wrote {len(ucr_entity_dirs)} exclude-file(s) under {args.exclude_files_dir}')
    print(f'UCR entity dirs added: {ucr_entity_dirs}')


if __name__ == '__main__':
    run()
