"""
Runs the ENTIRE Self data-efficiency experiment end-to-end in one command --
meant to be launched right before stepping away (e.g. overnight) and
checked on later:
  1. train_self_data_efficiency.py --orchestrate (combined loss, the n_pct sweep)
  2. train_self_data_efficiency.py --orchestrate --n_pcts 100 --c_loss_ratio 0
     (reconstruction-only @100%, see that script's own docstring for why)
  3. score_self_data_efficiency.py (combined loss sweep)
  4. score_self_data_efficiency.py --score_mode mse_only --out_prefix ... (reconstruction-only)
  5. plot_self_data_efficiency.py (the n_pct sweep only -- reconstruction-only
     is a single n_pct value, not a sweep, so there's no curve to plot; its
     result is printed directly instead, for comparison against the n=100
     combined-loss point in the sweep)

Each stage is a plain function call into the existing standalone script
(same code/CLI contract as running it directly) -- this file just wires
arguments through and sequences them; nothing here duplicates their logic.
Every stage is independently resumable (same as running them by hand), so
if this whole thing is killed partway through, just rerun the same command
-- finished (entity, n_pct, seed) jobs and already-scored rows are skipped,
not redone.

--skip_train / --skip_score / --skip_plot / --skip_recon_only let you
re-run just part of it later (e.g. --skip_train --skip_recon_only to just
re-plot after inspecting logs).
"""
import argparse
import importlib
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, 'scripts'))

DEFAULT_ENTITIES = ['044', '045', '046', '047', '152', '153', '154', '155']
DEFAULT_N_PCTS = [1, 3, 5, 10, 25, 50, 75, 100]
DEFAULT_SEEDS = [0, 1, 2]


def call(module_name, argv):
    mod = importlib.import_module(module_name)
    old_argv = sys.argv
    sys.argv = [module_name] + [str(a) for a in argv]
    try:
        mod.run()
    finally:
        sys.argv = old_argv


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--entities', nargs='+', default=DEFAULT_ENTITIES)
    parser.add_argument('--n_pcts', type=float, nargs='+', default=DEFAULT_N_PCTS)
    parser.add_argument('--seeds', type=int, nargs='+', default=DEFAULT_SEEDS)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--max_parallel', type=int, default=8)
    parser.add_argument('--output_dir', default='./result/DS_2/achievability/self_data_efficiency_models')
    parser.add_argument('--recon_output_dir', default='./result/DS_2/achievability/self_reconstruction_only_models')
    parser.add_argument('--out_dir', default='./result/DS_2/achievability')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--skip_train', action='store_true')
    parser.add_argument('--skip_score', action='store_true')
    parser.add_argument('--skip_plot', action='store_true')
    parser.add_argument('--skip_recon_only', action='store_true')
    args = parser.parse_args()

    n_pcts_str = [str(n) for n in args.n_pcts]
    seeds_str = [str(s) for s in args.seeds]

    if not args.skip_train:
        print('=== [train] combined-loss n_pct sweep ===', flush=True)
        argv = ['--orchestrate', '--entities', *args.entities, '--n_pcts', *n_pcts_str,
                '--seeds', *seeds_str, '--max_parallel', args.max_parallel,
                '--output_dir', args.output_dir, '--gpu', args.gpu, '--epochs', args.epochs]
        if args.force:
            argv.append('--force')
        call('train_self_data_efficiency', argv)

        if not args.skip_recon_only:
            print('=== [train] reconstruction-only @ n_pct=100 ===', flush=True)
            argv = ['--orchestrate', '--entities', *args.entities, '--n_pcts', 100,
                    '--seeds', *seeds_str, '--max_parallel', args.max_parallel,
                    '--output_dir', args.recon_output_dir, '--gpu', args.gpu, '--epochs', args.epochs,
                    '--c_loss_ratio', 0]
            if args.force:
                argv.append('--force')
            call('train_self_data_efficiency', argv)

    if not args.skip_score:
        print('=== [score] combined-loss n_pct sweep ===', flush=True)
        argv = ['--entities', *args.entities, '--n_pcts', *n_pcts_str, '--seeds', *seeds_str,
                '--models_dir', args.output_dir, '--out_dir', args.out_dir, '--gpu', args.gpu]
        if args.force:
            argv.append('--force')
        call('score_self_data_efficiency', argv)

        if not args.skip_recon_only:
            print('=== [score] reconstruction-only @ n_pct=100 ===', flush=True)
            argv = ['--entities', *args.entities, '--n_pcts', 100, '--seeds', *seeds_str,
                    '--models_dir', args.recon_output_dir, '--out_dir', args.out_dir,
                    '--out_prefix', 'self_reconstruction_only', '--score_mode', 'mse_only', '--gpu', args.gpu]
            if args.force:
                argv.append('--force')
            call('score_self_data_efficiency', argv)

    if not args.skip_plot:
        print('=== [plot] combined-loss n_pct sweep ===', flush=True)
        call('plot_self_data_efficiency', [
            '--summary_csv', os.path.join(args.out_dir, 'self_data_efficiency_overall_summary.csv'),
            '--out_dir', args.out_dir])

    if not args.skip_recon_only:
        recon_csv = os.path.join(args.out_dir, 'self_reconstruction_only_overall_summary.csv')
        combined_csv = os.path.join(args.out_dir, 'self_data_efficiency_overall_summary.csv')
        if os.path.isfile(recon_csv) and os.path.isfile(combined_csv):
            import pandas as pd
            recon = pd.read_csv(recon_csv)
            combined = pd.read_csv(combined_csv)
            combined_100 = combined[combined['n_pct'] == 100]
            print('\n=== reconstruction-only vs. combined-loss, both @ n_pct=100 ===')
            if not recon.empty and not combined_100.empty:
                for col in ['VUS_ROC_mean', 'VUS_PR_mean', 'R_AUC_ROC_mean', 'R_AUC_PR_mean', 'RF_mean']:
                    print(f'  {col}: combined={combined_100.iloc[0][col]:.4f}, '
                          f'reconstruction_only={recon.iloc[0][col]:.4f}')

    print('\nDone.')


if __name__ == '__main__':
    run()
