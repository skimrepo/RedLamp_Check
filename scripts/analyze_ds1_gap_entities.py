"""
DS_1: comprehensive good-vs-bad case analysis of Self vs Cross-AnomSim
performance gaps on UCR (anomaly_archive) entities, across BOTH
Experiment_1 (injected-anomaly 12-class classification accuracy) and
Experiment_2 (real-anomaly VUS-ROC).

Entities are NOT hardcoded -- selected automatically from
Experiment_1/Results/ucr_results.xlsx and Experiment_2/Results/ucr_results.xlsx's
"Per-Entity Comparison" sheets, by gap = self - cross_anomsim, in each of two
directions: "bad" (gap >= --gap_threshold, Cross-AnomSim struggles) and
"good" (gap <= 0, Cross-AnomSim keeps up with or beats Self), giving 4 groups:
exp1_bad, exp1_good, exp2_bad, exp2_good.

For every selected entity, generates (all under --out_dir, default
./result/DS_1 -- an analysis folder, not another Experiment):
  - entity_metadata.csv: real UCR category name (parsed from main.
    get_meta_data's filename convention), series length, which group(s) it's in
  - plots/{entity}/examples/: that entity's own waveform + all 12 injected
    pseudo-anomaly types (reuses main.save_anomaly_type_examples, scoped to
    a single-entity val_dataloader so it can't sample any other entity)
  - plots/{entity}/score_comparison.png: Self / Cross-OpenSource /
    Cross-AnomSim anomaly-score curves overlaid against the real anomaly
    window
  - plots/{entity}/tsne_self.png, tsne_cross_anomsim.png: t-SNE embedding-
    space separation of that entity's own validation windows by injected
    pseudo-anomaly type, under each model
  - type_confusion.csv: for Experiment_1-flagged entities only, per-type
    classification accuracy breakdown (Self vs Cross-AnomSim)
  - group_summary.csv: bad-vs-good aggregate comparison (category
    distribution, length stats) per group

Reuses (no changes needed there): main.get_meta_data, main.
save_anomaly_type_examples, main.extract_embeddings, main.plot_tsne_embeddings,
cross_inference.discover_entity/build_dataparams/build_model_args,
full_cross_domain_metrics.cross_model_dir, full_reproduction_metrics.
real_ground_truth_labels. The one small addition elsewhere is
full_reproduction_metrics.score_entity's new optional `include_curves` kwarg
(default False, existing callers unaffected) so this script can pull the
actual score/label arrays instead of just the 5 scalar metrics.
"""
import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import main
import datautils
import utils
import cross_inference as ci
import domain_generalization as dg
import continuous_pool_scaling as cps
import full_reproduction_metrics as frm
import full_cross_domain_metrics as fcdm

DATASET = 'anomaly_archive'
GROUP_NAMES = ['exp1_bad', 'exp1_good', 'exp2_bad', 'exp2_good']


def _zfill_entity(x):
    return str(int(x)).zfill(3)


def select_entities(exp1_xlsx, exp2_xlsx, gap_threshold, top_n):
    exp1 = pd.read_excel(exp1_xlsx, sheet_name='Per-Entity Comparison')
    exp2 = pd.read_excel(exp2_xlsx, sheet_name='Per-Entity Comparison')
    exp1['entity'] = exp1['entity'].apply(_zfill_entity)
    exp2['entity'] = exp2['entity'].apply(_zfill_entity)

    exp1['gap'] = exp1['accuracy_self'] - exp1['accuracy_cross_anomsim']
    exp2['gap'] = exp2['VUS_ROC_self'] - exp2['VUS_ROC_cross_anomsim']

    def pick(df, ascending, keep):
        d = df.dropna(subset=['gap']).copy()
        d = d[keep(d['gap'])].sort_values('gap', ascending=ascending)
        return dict(list(zip(d['entity'], d['gap']))[:top_n])

    return {
        'exp1_bad': pick(exp1, ascending=False, keep=lambda g: g >= gap_threshold),
        'exp1_good': pick(exp1, ascending=True, keep=lambda g: g <= 0),
        'exp2_bad': pick(exp2, ascending=False, keep=lambda g: g >= gap_threshold),
        'exp2_good': pick(exp2, ascending=True, keep=lambda g: g <= 0),
    }


def build_val_dataloader(run_name, entity, seed):
    _, disk_cfg = ci.discover_entity(run_name, DATASET, entity, seed)
    dataparams = ci.build_dataparams(DATASET, entity, dg.CFG, disk_cfg)
    _, val_dl = datautils.load_dataloader_aug(dataparams, group='train')
    return val_dl


def entity_metadata(entity):
    meta = main.get_meta_data(entity)
    if meta is None:
        return None
    name_parts = meta['name'].split('_')
    category = name_parts[3] if len(name_parts) > 3 else None
    real_labels = frm.real_ground_truth_labels(DATASET, entity)
    test_length = len(real_labels)
    return dict(
        entity=entity, category=category, train_end=meta['train_end'],
        test_length=test_length, total_length=meta['train_end'] + test_length,
        anomaly_length=meta['anomaly_end_in_test'] - meta['anomaly_start_in_test'],
    )


def save_entity_examples(val_dl, save_dir, seed):
    os.makedirs(save_dir, exist_ok=True)
    main.save_anomaly_type_examples(val_dl, save_dir, n_examples=1, seed=seed)


def plot_score_comparison(entity, ucr_meta, curves, save_path, context=1000):
    """curves: dict model_name -> score_entity(include_curves=True) result or None."""
    anomaly_start = ucr_meta['anomaly_start_in_test']
    anomaly_end = ucr_meta['anomaly_end_in_test']
    if anomaly_start == anomaly_end:
        anomaly_end += 1

    any_result = next((r for r in curves.values() if r is not None), None)
    if any_result is None:
        print(f'[skip] {entity}: no score curve available from any model')
        return False

    total_len = len(any_result['score'])
    start = max(0, anomaly_start - context)
    end = min(total_len, anomaly_end + context)
    x = np.arange(start, end)

    colors = {'self': '#3f7fbf', 'cross_opensource': '#e0883f', 'cross_anomsim': '#3fae59',
              'reference_distance': '#9b59b6'}

    fig, (ax_raw, ax_score) = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    ax_raw.plot(x, any_result['raw_series'][start:end], color='#888888', linewidth=1.0)
    ax_raw.axvspan(anomaly_start, anomaly_end, color='#e34948', alpha=0.15)
    ax_raw.set_ylabel('raw series')

    for model_name, result in curves.items():
        if result is None:
            continue
        ax_score.plot(x, result['score'][start:end], label=model_name, color=colors.get(model_name), linewidth=1.2)
    ax_score.axvspan(anomaly_start, anomaly_end, color='#e34948', alpha=0.15, label='real anomaly')
    ax_score.set_ylabel('anomaly score')
    ax_score.set_xlabel('timestep')
    ax_score.legend(fontsize=8)

    fig.suptitle(f'{entity} ({ucr_meta["name"]})', fontsize=10)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return True


def plot_tsne_for_entity(model_dir, val_dl, params, device, save_path, title, seed):
    if not os.path.isfile(f'{model_dir}/bestmodel.pkl'):
        print(f'[skip] no bestmodel.pkl at {model_dir}')
        return
    embeddings, class_idx = main.extract_embeddings(model_dir, params, device, val_dl, max_samples=2000)
    main.plot_tsne_embeddings(embeddings, class_idx, val_dl.anomaly_dict, save_path, title=title, seed=seed)


def compute_type_confusion(model_dir, params, device, val_dataloader):
    if not os.path.isfile(f'{model_dir}/bestmodel.pkl'):
        return []
    model = main.ConvAEC(params).to(device)
    model.load_state_dict(torch.load(f'{model_dir}/bestmodel.pkl'))
    model.eval()

    true_all, pred_all = [], []
    with torch.no_grad():
        for batch in val_dataloader:
            inputs = batch['Y'].transpose(2, 1).to(device)
            true = batch['label'].argmax(dim=1)
            _, x_out, _ = model(inputs)
            pred = x_out.argmax(dim=1).cpu()
            true_all.append(true.numpy())
            pred_all.append(pred.numpy())
    true_all = np.concatenate(true_all)
    pred_all = np.concatenate(pred_all)

    inverse_dict = {v: k for k, v in val_dataloader.anomaly_dict.items()}
    rows = []
    for type_idx, type_name in inverse_dict.items():
        mask = true_all == type_idx
        n = int(mask.sum())
        if n == 0:
            continue
        rows.append(dict(anomaly_type=type_name, n=n, accuracy=float((pred_all[mask] == type_idx).mean())))
    return rows


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', default='test')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--exp1_xlsx', default='./result/Experiment_1/Results/ucr_results.xlsx')
    parser.add_argument('--exp2_xlsx', default='./result/Experiment_2/Results/ucr_results.xlsx')
    parser.add_argument('--cross_anomsim_model_dir', default=None,
                         help='Defaults to ./result/Experiment_1/Models/Cross-AnomSim/{seed}')
    parser.add_argument('--gap_threshold', type=float, default=0.5,
                         help='"Bad" cases: gap (self - cross_anomsim) >= this. "Good" cases are always gap <= 0.')
    parser.add_argument('--top_n', type=int, default=15, help='Cap each of the 4 groups to this many entities.')
    parser.add_argument('--out_dir', default='./result/DS_1')
    args = parser.parse_args()

    cross_anomsim_model_dir = args.cross_anomsim_model_dir or f'./result/Experiment_1/Models/Cross-AnomSim/{args.seed}'
    device = utils.init_dl_program(args.gpu, seed=args.seed)

    groups = select_entities(args.exp1_xlsx, args.exp2_xlsx, args.gap_threshold, args.top_n)
    for name in GROUP_NAMES:
        print(f'{name}: {len(groups[name])} entities -> {list(groups[name].keys())}')

    entity_groups, entity_gaps = {}, {}
    for group_name in GROUP_NAMES:
        for entity, gap in groups[group_name].items():
            entity_groups.setdefault(entity, set()).add(group_name)
            entity_gaps.setdefault(entity, {})[group_name] = gap

    exp1_entities = set(groups['exp1_bad']) | set(groups['exp1_good'])

    plots_dir = os.path.join(args.out_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    model_args = ci.build_model_args(dg.CFG, cps.WINDOW_SIZE)
    params = utils.AttrDict(seed=args.seed)
    params.override(main.model_parameters(model_args))

    metadata_rows, confusion_rows = [], []

    for entity in sorted(entity_groups):
        member_groups = entity_groups[entity]
        print(f'--- {entity} ({sorted(member_groups)}) ---')

        meta = entity_metadata(entity)
        if meta is None:
            print(f'[skip] {entity}: no UCR metadata found')
            continue
        meta['groups'] = ','.join(sorted(member_groups))
        for g in GROUP_NAMES:
            meta[f'gap_{g}'] = entity_gaps[entity].get(g)
        metadata_rows.append(meta)

        try:
            self_model_dir, _ = ci.discover_entity(args.run_name, DATASET, entity, args.seed)
            val_dl = build_val_dataloader(args.run_name, entity, args.seed)
        except FileNotFoundError:
            print(f'[skip] {entity}: no self-model/val-split found')
            continue

        cross_opensource_dir = fcdm.cross_model_dir(args.run_name, DATASET, args.seed)
        entity_plot_dir = os.path.join(plots_dir, entity)
        os.makedirs(entity_plot_dir, exist_ok=True)

        save_entity_examples(val_dl, os.path.join(entity_plot_dir, 'examples'), args.seed)

        curves = {}
        for model_name, model_dir in [('self', self_model_dir), ('cross_opensource', cross_opensource_dir),
                                       ('cross_anomsim', cross_anomsim_model_dir)]:
            curves[model_name] = frm.score_entity(args.run_name, DATASET, entity, args.seed, params, device,
                                                    model_dir=model_dir, include_curves=True)
        ucr_meta = main.get_meta_data(entity)
        plot_score_comparison(entity, ucr_meta, curves, os.path.join(entity_plot_dir, 'score_comparison.png'))

        plot_tsne_for_entity(self_model_dir, val_dl, params, device,
                              os.path.join(entity_plot_dir, 'tsne_self.png'), f'{entity} - Self', args.seed)
        plot_tsne_for_entity(cross_anomsim_model_dir, val_dl, params, device,
                              os.path.join(entity_plot_dir, 'tsne_cross_anomsim.png'),
                              f'{entity} - Cross-AnomSim', args.seed)

        if entity in exp1_entities:
            for model_name, model_dir in [('self', self_model_dir), ('cross_anomsim', cross_anomsim_model_dir)]:
                for row in compute_type_confusion(model_dir, params, device, val_dl):
                    row.update(entity=entity, model=model_name, groups=meta['groups'])
                    confusion_rows.append(row)

    os.makedirs(args.out_dir, exist_ok=True)
    metadata_df = pd.DataFrame(metadata_rows)
    metadata_df.to_csv(os.path.join(args.out_dir, 'entity_metadata.csv'), index=False)

    if confusion_rows:
        pd.DataFrame(confusion_rows).to_csv(os.path.join(args.out_dir, 'type_confusion.csv'), index=False)

    summary_rows = []
    if not metadata_df.empty:
        for group_name in GROUP_NAMES:
            sub = metadata_df[metadata_df['groups'].str.contains(group_name)]
            if sub.empty:
                continue
            summary_rows.append(dict(
                group=group_name, n=len(sub),
                mean_total_length=sub['total_length'].mean(),
                median_total_length=sub['total_length'].median(),
                mean_anomaly_length=sub['anomaly_length'].mean(),
                top_categories=', '.join(f'{k}({v})' for k, v in sub['category'].value_counts().head(3).items()),
            ))
    pd.DataFrame(summary_rows).to_csv(os.path.join(args.out_dir, 'group_summary.csv'), index=False)

    print(f'Done. {len(metadata_rows)} entities analyzed. Wrote {args.out_dir}')


if __name__ == '__main__':
    run()
