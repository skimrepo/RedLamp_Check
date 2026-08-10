"""
Trains one leave-one-out model per UCR entity in --entities (default: the
8 PowerDemand entities), using core_clustering.online_cli UNMODIFIED --
each run points at the merged pool build_ucr_anomsim_pool.py already built
(AnomSim_v1's 144 entities + all 8 UCR entities) and excludes just ONE UCR
entity via --exclude_entity_dirs_file, so the resulting model is trained on
AnomSim_v1 + the OTHER 7 UCR entities, having never seen the held-out one.

All other online_cli.py args are left at their defaults, matching exactly
how the real production Cross-AnomSim was trained (no --held_out_domains,
class_list=redlamp, epochs=100/patience=10/batch_size=128/window_size=100/
window_step=10/val_fraction=0.2) -- see result/result/Experiment_1/Models/
Cross-AnomSim/SOURCE.txt and run_summary.json for confirmation of those
defaults.

Sequential by default, not parallel -- unlike this repo's other parallel
orchestrators (which fan out lightweight CPU-only inference), this is real
GPU training (~20 min/run per the original Cross-AnomSim run's own
timing), and the GPU here is shared with other jobs -- see
run_ucr_test_diagnostics_parallel.py's own history of what happens when
many processes fight over one GPU's memory.8 runs x ~20 min ~= 2.5-3
hours total; --entities lets you split this across multiple manual
invocations (e.g. one shard per terminal) if you want to risk running a
couple concurrently on a GPU with enough free memory margin.

Resumable: online_cli.py itself skips retraining if output_dir/run_id/
bestmodel.pkl already exists (pass --force to override).
"""
import argparse
import os
import subprocess
import sys

DEFAULT_ENTITIES = ['044', '045', '046', '047', '152', '153', '154', '155']


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pool_dir', default='./result/DS_2/achievability/anomsim_plus_ucr_powerdemand')
    parser.add_argument('--exclude_files_dir', default=None, help='Default: {pool_dir}/exclude_files')
    parser.add_argument('--ucr_domain_name', default='ucr_PowerDemand')
    parser.add_argument('--entities', nargs='+', default=DEFAULT_ENTITIES)
    parser.add_argument('--core_clustering_dir', default='../Core-Clustering')
    parser.add_argument('--output_dir', default='./result/DS_2/achievability/loo_powerdemand_models')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    exclude_files_dir = args.exclude_files_dir or os.path.join(args.pool_dir, 'exclude_files')

    pool_dir_abs = os.path.abspath(args.pool_dir)
    output_dir_abs = os.path.abspath(args.output_dir)
    core_clustering_dir_abs = os.path.abspath(args.core_clustering_dir)

    results = []
    for entity in args.entities:
        entity_dir_name = f'{args.ucr_domain_name}_{entity}'
        exclude_file = os.path.abspath(os.path.join(exclude_files_dir, f'exclude_{entity_dir_name}.json'))
        if not os.path.isfile(exclude_file):
            print(f'[skip] {entity}: no exclude file at {exclude_file} -- run build_ucr_anomsim_pool.py first')
            continue
        run_id = f'without_{entity_dir_name}'

        cmd = [sys.executable, '-m', 'core_clustering.online_cli',
               '--dataset_dir', pool_dir_abs,
               '--exclude_entity_dirs_file', exclude_file,
               '--output_dir', output_dir_abs,
               '--run_id', run_id,
               '--seed', str(args.seed),
               '--gpu', str(args.gpu),
               '--epochs', str(args.epochs)]
        if args.force:
            cmd.append('--force')

        print(f'[launch] {run_id}')
        print('  ' + ' '.join(cmd))
        ret = subprocess.run(cmd, cwd=core_clustering_dir_abs).returncode
        model_path = os.path.join(output_dir_abs, run_id, 'bestmodel.pkl')
        ok = ret == 0 and os.path.isfile(model_path)
        results.append((entity, ok, model_path))
        print(f'  {"OK" if ok else f"FAILED (exit {ret})"} -> {model_path}')

    print('\nSummary:')
    for entity, ok, model_path in results:
        print(f'  {entity}: {"OK" if ok else "FAILED"} {model_path}')
    failed = [e for e, ok, _ in results if not ok]
    if failed:
        print(f'\nWARNING: {len(failed)} fold(s) failed: {failed}')


if __name__ == '__main__':
    run()
