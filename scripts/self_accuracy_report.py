"""
Self-only validation classification accuracy across every trained model in
every dataset (anomaly_archive, iops, smd, smap, msl).

Unlike domain_generalization.py's 9x6 cross-domain matrix (only valid between
anomaly_archive/iops, which both share n_features=1), the other datasets have
different n_features (smd=38, smap=25, msl=55) — feeding one model's weights
a differently-shaped input is architecturally impossible (Conv1d channel
mismatch), so this script only ever evaluates a model against its own
validation split. Does not modify main.py or cross_inference.py.
"""
import argparse
import os
import sys

import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import main
import datautils
import utils

import cross_inference as ci


DATASET_CFGS = {
    'anomaly_archive': dict(n_features=1, min_features=1, max_features=1),
    'iops': dict(n_features=1, min_features=1, max_features=1),
    'smd': dict(n_features=38, min_features=1, max_features=38),
    'smap': dict(n_features=25, min_features=1, max_features=25),
    'msl': dict(n_features=55, min_features=1, max_features=55),
}


def discover_dataset_entities(run_name, dataset):
    base = f'./result/{run_name}/{dataset}'
    if not os.path.isdir(base):
        return []
    return sorted(d for d in os.listdir(base)
                  if os.path.isdir(os.path.join(base, d)) and not d.startswith('_'))


def compute_self_accuracy(model_dir, params, device, val_dataloader):
    model = main.ConvAEC(params).to(device)
    model.load_state_dict(torch.load(f'{model_dir}/bestmodel.pkl'))
    model.eval()

    correct, total = 0, 0
    with torch.no_grad():
        for batch in val_dataloader:
            inputs = batch['Y'].transpose(2, 1).to(device)
            true = batch['label'].argmax(dim=1)
            _, x_out, _ = model(inputs)
            pred = x_out.argmax(dim=1).cpu()
            correct += (pred == true).sum().item()
            total += len(true)
    return correct / total, total


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', default='test')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--out_csv', default=None)
    args_cli = parser.parse_args()

    device = utils.init_dl_program(args_cli.gpu, seed=args_cli.seed)
    out_csv = args_cli.out_csv or f'./result/{args_cli.run_name}/self_accuracy_all_datasets.csv'

    rows = []
    for dataset, cfg in DATASET_CFGS.items():
        entities = discover_dataset_entities(args_cli.run_name, dataset)
        if not entities:
            print(f'[skip] no trained entities found for {dataset}')
            continue
        for entity in entities:
            print(f'Evaluating: dataset={dataset} entity={entity}')
            model_dir, disk_cfg = ci.discover_entity(args_cli.run_name, dataset, entity, args_cli.seed)
            dataparams = ci.build_dataparams(dataset, entity, cfg, disk_cfg)
            _, val_dl = datautils.load_dataloader_aug(dataparams, group='train')

            model_args = ci.build_model_args(cfg, disk_cfg['window_size'])
            params = utils.AttrDict(seed=args_cli.seed)
            params.override(main.model_parameters(model_args))

            accuracy, n_windows = compute_self_accuracy(model_dir, params, device, val_dl)
            rows.append(dict(dataset=dataset, entity=entity, model_dir=model_dir,
                              val_windows=n_windows, accuracy=accuracy))
            print(f'  -> accuracy={accuracy:.4f} (n={n_windows})')

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print('Done. Wrote', out_csv)


if __name__ == '__main__':
    run()
