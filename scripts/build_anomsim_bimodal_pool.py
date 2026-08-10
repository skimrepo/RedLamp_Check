"""
Builds a Core-Clustering-compatible "base pool" directory that merges
AnomSim_v1's 144 entities with AnomSim's separately-tracked bimodal_cycle
domain (16 entities, data/AnomSim_v1_bimodal_part/ -- added later, inspired
by / fit to UCR PowerDemand's real daily double-peak shape, see
anomsim.waveforms.basic.BimodalCycleWaveform's docstring), so
core_clustering.online_cli can be pointed at ONE --dataset_dir covering
both and train a single model (no leave-one-out here -- bimodal_cycle is
purely synthetic, so it can never leak any of UCR PowerDemand's real test
entities; every one of them stays eligible for evaluation against the
resulting model).

Both source directories already share the exact same manifest schema
(type/params/n_time/base_instance_id/base_seed/entity_dir), so this is
pure symlinking + manifest concatenation -- no data is copied or
regenerated, and neither source directory is ever modified. Unlike
build_ucr_anomsim_pool.py, this needs no raw UCR dataset access, so it
runs the same on this development machine as on the server.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def symlink_pool_entities(src_dir, out_dir):
    manifest_path = os.path.join(src_dir, '_manifest.jsonl')
    lines = []
    with open(manifest_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            meta = json.loads(line)
            entity_dir = meta['entity_dir']
            src = os.path.abspath(os.path.join(src_dir, entity_dir))
            dst = os.path.join(out_dir, entity_dir)
            if not os.path.islink(dst) and not os.path.exists(dst):
                os.symlink(src, dst)
            lines.append(line)
    return lines


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--anomsim_dir', default='../AnomSim/data/AnomSim_v1')
    parser.add_argument('--bimodal_dir', default='../AnomSim/data/AnomSim_v1_bimodal_part')
    parser.add_argument('--out_dir', default='./result/DS_2/achievability/anomsim_plus_bimodal')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    anomsim_lines = symlink_pool_entities(args.anomsim_dir, args.out_dir)
    print(f'Symlinked {len(anomsim_lines)} AnomSim_v1 entities into {args.out_dir}')

    bimodal_lines = symlink_pool_entities(args.bimodal_dir, args.out_dir)
    print(f'Symlinked {len(bimodal_lines)} bimodal_cycle entities into {args.out_dir}')

    manifest_path = os.path.join(args.out_dir, '_manifest.jsonl')
    with open(manifest_path, 'w') as f:
        f.write('\n'.join(anomsim_lines + bimodal_lines) + '\n')
    print(f'Wrote {manifest_path} ({len(anomsim_lines) + len(bimodal_lines)} total entities)')


if __name__ == '__main__':
    run()
