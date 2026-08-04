"""
Quantifies the mechanism behind Cross-AnomSim's gap vs Self (complementing
DS_1's "achievability" finding -- see plot_self_vs_cross_anomsim_scatter.py):
does Cross-AnomSim's anomaly score oscillate with the raw signal's own local
activity even during NORMAL (non-anomalous) periods, unlike Self's?

For each of DS_1's 46 gap-selected entities, computes (restricted to
timesteps where real_labels == 0, i.e. excluding the real anomaly window):
  - self_score_std_normal / cross_anomsim_score_std_normal: how much each
    model's score oscillates during normal periods (should be near-flat/low
    if the model has a clean, calibrated baseline for this entity)
  - self_corr_with_activity / cross_anomsim_corr_with_activity: Pearson
    correlation between each model's score and a "local activity" signal
    (rolling std of the raw series) -- tests whether the score is just
    reacting to how much the raw signal happens to be changing right now,
    regardless of whether that change is actually anomalous.

Measured result (already run once): score_std_normal turned out to be
similar for both models AND similar across bad/good groups -- ruling out
"whoever has less baseline noise wins" as too-simple an explanation. But a
closer look at DS_1's score_comparison.png plots showed WHY that comparison
was incomplete: Cross-AnomSim's score sits in a compressed band (e.g.
0.2-0.68 for entity 040) that never approaches 0 or 1, while Self's spans
close to the full range -- despite both being min-max normalized. This adds
the diagnostics needed to explain that and pin down what actually drives
VUS_ROC, still restricted to the same curves already being computed here:
  - self_score_min_normal / _max_normal: the actual normal-period operating
    band (directly shows the compression std alone couldn't reveal)
  - self_numerator / cross_anomsim_numerator: anomaly-period score mean
    minus this SAME model's own normal-period mean -- a within-model
    comparison, so it isn't distorted by which band a model's score happens
    to occupy
  - self_vus_roc / cross_anomsim_vus_roc: pulled straight from
    score_entity's own metrics dict, to correlate directly against the
    numerator above
  - self_mse_ce_corr / cross_anomsim_mse_ce_corr: correlation between the
    two raw score components (mse_score, ce_score -- each independently
    min-max'd before being averaged into the final score, see
    main.anomaly_scoreing). If they don't rise/fall together, their average
    gets compressed away from 0/1 regardless of true precision -- this
    tests whether THAT desynchronization is what compresses Cross-AnomSim's
    band on hard entities specifically.

Prediction: cross_anomsim_numerator should vary a lot across entities (and
correlate strongly with cross_anomsim_vus_roc), driven by whether the real
anomaly happens to coincide with a local-activity spike its score reacts
to -- while self_numerator should be reliably large regardless. And
cross_anomsim_mse_ce_corr should be lower (more desynchronized) on
exp2_bad entities than exp2_good ones, explaining the band compression.

Reuses full_reproduction_metrics.score_entity(include_curves=True) exactly
as DS_1's analyze_ds1_gap_entities.py does for score_comparison.png -- that
script is not modified; this one just calls the same function again (a
modest amount of redundant inference, but keeps DS_1 untouched).

Resumable: entities already in the output CSV are skipped unless --force.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils
import cross_inference as ci
import domain_generalization as dg
import continuous_pool_scaling as cps
import full_reproduction_metrics as frm
import main

DATASET = 'anomaly_archive'


def local_activity(raw_series, window_size):
    """Rolling std of the raw series -- how much the signal is locally
    changing at each point, regardless of whether that change is real
    anomaly or ordinary periodic transition."""
    return pd.Series(raw_series).rolling(window_size, center=True, min_periods=1).std().values


def safe_corr(a, b):
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def analyze_entity(run_name, entity, seed, params, device, cross_anomsim_model_dir):
    self_model_dir, disk_cfg = ci.discover_entity(run_name, DATASET, entity, seed)
    window_size = disk_cfg['window_size']

    curves = {}
    for model_name, model_dir in [('self', self_model_dir), ('cross_anomsim', cross_anomsim_model_dir)]:
        curves[model_name] = frm.score_entity(run_name, DATASET, entity, seed, params, device,
                                               model_dir=model_dir, include_curves=True)
    if curves['self'] is None or curves['cross_anomsim'] is None:
        return None

    real_labels = curves['self']['real_labels']
    normal_mask = real_labels == 0
    anomaly_mask = ~normal_mask
    raw_series = curves['self']['raw_series']

    activity = local_activity(raw_series, window_size)

    row = dict(entity=entity)
    for model_name in ['self', 'cross_anomsim']:
        score = curves[model_name]['score']
        mse_score = curves[model_name]['mse_score']
        ce_score = curves[model_name]['ce_score']

        normal_mean = float(np.mean(score[normal_mask]))
        anomaly_mean = float(np.mean(score[anomaly_mask]))

        row[f'{model_name}_score_std_normal'] = float(np.std(score[normal_mask]))
        row[f'{model_name}_score_min_normal'] = float(np.min(score[normal_mask]))
        row[f'{model_name}_score_max_normal'] = float(np.max(score[normal_mask]))
        row[f'{model_name}_corr_with_activity'] = safe_corr(score[normal_mask], activity[normal_mask])
        row[f'{model_name}_normal_score_mean'] = normal_mean
        row[f'{model_name}_anomaly_score_mean'] = anomaly_mean
        # The "numerator": how much the score actually rises during the true
        # anomaly relative to this SAME model's own normal-period baseline --
        # a within-model comparison, so it's not distorted by one model's
        # score sitting in a compressed band away from 0/1 (see mse_ce_corr).
        row[f'{model_name}_numerator'] = anomaly_mean - normal_mean
        row[f'{model_name}_vus_roc'] = curves[model_name]['metrics']['VUS_ROC']
        # Synchronization: do the two raw components (reconstruction-error-
        # based vs classifier-based) rise/fall at the same timesteps? Each is
        # independently min-max'd to touch 0 and 1 somewhere (see
        # main.anomaly_scoreing) -- if they're desynchronized, their average
        # (the final score) never gets close to 0 or 1, compressing the
        # entire operating band regardless of "true" precision.
        row[f'{model_name}_mse_ce_corr'] = safe_corr(mse_score, ce_score)
    return row


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', default='test')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--entity_metadata_csv', default='./result/DS_1/entity_metadata.csv')
    parser.add_argument('--cross_anomsim_model_dir', default=None,
                         help='Defaults to ./result/Experiment_1/Models/Cross-AnomSim/{seed}')
    parser.add_argument('--out_csv', default='./result/DS_2/oscillation/oscillation_metrics.csv')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    cross_anomsim_model_dir = args.cross_anomsim_model_dir or f'./result/Experiment_1/Models/Cross-AnomSim/{args.seed}'
    device = utils.init_dl_program(args.gpu, seed=args.seed)

    model_args = ci.build_model_args(dg.CFG, cps.WINDOW_SIZE)
    params = utils.AttrDict(seed=args.seed)
    params.override(main.model_parameters(model_args))

    entities = pd.read_csv(args.entity_metadata_csv)['entity'].astype(str).str.zfill(3).tolist()
    print(f'{len(entities)} entities loaded from {args.entity_metadata_csv}')

    out_dir = os.path.dirname(args.out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    rows = []
    already_done = set()
    if not args.force and os.path.isfile(args.out_csv):
        prior = pd.read_csv(args.out_csv)
        rows = prior.to_dict('records')
        already_done = set(prior['entity'].astype(str).str.zfill(3))
        print(f'Resuming from {args.out_csv}: {len(already_done)} entities already done.')

    for entity in entities:
        if entity in already_done:
            continue
        try:
            row = analyze_entity(args.run_name, entity, args.seed, params, device, cross_anomsim_model_dir)
        except FileNotFoundError:
            print(f'[skip] {entity}: no trained model found')
            continue
        if row is None:
            print(f'[skip] {entity}: no score curve available')
            continue
        rows.append(row)
        print(f'  {entity}: {row}')
        pd.DataFrame(rows).to_csv(args.out_csv, index=False)

    print(f'Done. {len(rows)} entities analyzed. Wrote {args.out_csv}')


if __name__ == '__main__':
    run()
