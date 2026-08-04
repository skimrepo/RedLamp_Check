"""
Adds a 4th curve -- the model-free Reference-Distance baseline (see
reference_distance_baseline.py) -- on top of DS_1's existing Self /
Cross-OpenSource / Cross-AnomSim score_comparison plots, for a specific list
of entities. Visual counterpart to the aggregate numbers in
result/DS_2/reference_distance_metrics.csv, which showed Reference-Distance
closing 90%+ of the VUS_ROC/R_AUC_ROC gap on the exp2_bad entities DS_1
flagged.

Reuses analyze_ds1_gap_entities.plot_score_comparison unchanged (its color
palette already has a 'reference_distance' entry) and
reference_distance_baseline.score_entity_reference_distance's include_curves
option. Does not touch DS_1's own output -- writes to
result/DS_2/plots/{entity}/score_comparison_with_reference.png instead.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import main
import utils
import cross_inference as ci
import domain_generalization as dg
import continuous_pool_scaling as cps
import full_reproduction_metrics as frm
import full_cross_domain_metrics as fcdm

from analyze_ds1_gap_entities import plot_score_comparison, DATASET
from reference_distance_baseline import score_entity_reference_distance


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', default='test')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--entities', nargs='+', default=['040', '042'])
    parser.add_argument('--cross_anomsim_model_dir', default=None,
                         help='Defaults to ./result/Experiment_1/Models/Cross-AnomSim/{seed}')
    parser.add_argument('--max_reference_windows', type=int, default=300)
    parser.add_argument('--rng_seed', type=int, default=0)
    parser.add_argument('--out_dir', default='./result/DS_2/plots')
    args = parser.parse_args()

    cross_anomsim_model_dir = args.cross_anomsim_model_dir or f'./result/Experiment_1/Models/Cross-AnomSim/{args.seed}'
    device = utils.init_dl_program(args.gpu, seed=args.seed)

    model_args = ci.build_model_args(dg.CFG, cps.WINDOW_SIZE)
    params = utils.AttrDict(seed=args.seed)
    params.override(main.model_parameters(model_args))

    rng = np.random.default_rng(args.rng_seed)

    for entity in args.entities:
        entity = str(entity).zfill(3)
        try:
            self_model_dir, _ = ci.discover_entity(args.run_name, DATASET, entity, args.seed)
        except FileNotFoundError:
            print(f'[skip] {entity}: no self-model found')
            continue
        cross_opensource_dir = fcdm.cross_model_dir(args.run_name, DATASET, args.seed)

        curves = {}
        for model_name, model_dir in [('self', self_model_dir), ('cross_opensource', cross_opensource_dir),
                                       ('cross_anomsim', cross_anomsim_model_dir)]:
            curves[model_name] = frm.score_entity(args.run_name, DATASET, entity, args.seed, params, device,
                                                    model_dir=model_dir, include_curves=True)
        curves['reference_distance'] = score_entity_reference_distance(
            args.run_name, entity, args.seed, args.max_reference_windows, rng, include_curves=True)

        ucr_meta = main.get_meta_data(entity)
        entity_plot_dir = os.path.join(args.out_dir, entity)
        os.makedirs(entity_plot_dir, exist_ok=True)
        ok = plot_score_comparison(entity, ucr_meta, curves,
                                    os.path.join(entity_plot_dir, 'score_comparison_with_reference.png'))
        print(f'{entity}: {"wrote" if ok else "skipped"} score_comparison_with_reference.png')


if __name__ == '__main__':
    run()
