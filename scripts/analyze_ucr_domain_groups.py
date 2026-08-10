"""
DS_2: how the Self-vs-Cross-AnomSim VUS_ROC gap grouping (Red/BAD, Green/GOOD,
Other -- same convention as plot_self_vs_cross_anomsim_scatter.py) breaks
down by UCR domain and by distortion variant (DISTORTED/NOISE/plain).

Entity -> real UCR category name is NOT available from a clean source
locally: the raw dataset (./dataset/AnomalyArchive/*.txt, whose filenames
encode the category) isn't present on this machine, and main.get_meta_data
would try to download it. Instead this scrapes 'Entity meta-data: {...}'
lines that main.py/loaders/load.py already print (verbose=True by
default) and that happen to survive in old run logs
(result/DS_2/oscillation/shards/*.log, result/DS_3/test_diagnostics/logs/*.log).
This is a fallback, not a designed pipeline -- if those log files are ever
cleaned up, category names need to come from the server's actual dataset
via main.get_meta_data instead. --log_glob lets you point at wherever
comparable logs exist.

UCR's Anomaly Archive registers the SAME source recording under up to
three separate entities -- DISTORTED<name><n>, NOISE<name><n>, and plain
<name><n> -- confirmed here by matching identical train_end/
anomaly_start_in_test/anomaly_end_in_test across variants (e.g. entities
027/098/135 are all "InternalBleeding16", just DISTORTED/NOISE/plain).
`domain` strips that prefix AND the trailing instance number, so all three
variants of the same recording land in one domain row. This is a regex
heuristic, not an authoritative UCR taxonomy -- entities that differ only
by a LEADING digit (e.g. '1sddb40'/'2sddb40'/'3sddb40'/'sddb40', likely
different channels of the same recording) are deliberately NOT merged
further since that would be guessing without confirmation.
"""
import argparse
import glob
import os
import re

import pandas as pd

GAP_GROUPS = [
    ('Red', lambda gap, thr: gap >= thr),
    ('Green', lambda gap, thr: gap <= 0),
    ('Other', lambda gap, thr: (gap > 0) & (gap < thr)),
]


def scrape_entity_domains(log_glob):
    """Returns DataFrame(entity, category, variant, domain) by parsing
    every unique "Entity meta-data: {'name': '...', ...}" line found across
    log_glob -- see module docstring for why this (rather than
    main.get_meta_data) is the source."""
    pattern = re.compile(r"'name': '(\d+)_UCR_Anomaly_([A-Za-z0-9]+)'")
    rows = {}
    for path in glob.glob(log_glob, recursive=True):
        with open(path, errors='ignore') as f:
            for line in f:
                m = pattern.search(line)
                if not m:
                    continue
                entity, category = m.group(1), m.group(2)
                if entity in rows:
                    continue
                if category.startswith('DISTORTED'):
                    variant, base = 'DISTORTED', category[len('DISTORTED'):]
                elif category.startswith('NOISE'):
                    variant, base = 'NOISE', category[len('NOISE'):]
                else:
                    variant, base = 'plain', category
                domain = re.sub(r'\d+$', '', base)
                rows[entity] = dict(entity=entity, category=category, variant=variant, domain=domain)
    return pd.DataFrame(rows.values())


def classify_gap(ucr_xlsx, gap_threshold):
    per_entity = pd.read_excel(ucr_xlsx, sheet_name='Per-Entity Comparison')
    per_entity['entity'] = per_entity['entity'].astype(str).str.zfill(3)
    per_entity = per_entity.dropna(subset=['VUS_ROC_self', 'VUS_ROC_cross_anomsim']).copy()
    per_entity['gap'] = per_entity['VUS_ROC_self'] - per_entity['VUS_ROC_cross_anomsim']

    def group_of(gap):
        for name, predicate in GAP_GROUPS:
            if predicate(gap, gap_threshold):
                return name
        return 'Other'
    per_entity['group'] = per_entity['gap'].apply(group_of)
    return per_entity[['entity', 'gap', 'group']]


def build_group_table(merged, index_col, with_entity_lists=False):
    pivot = merged.pivot_table(index=index_col, columns='group', values='entity', aggfunc='count', fill_value=0)
    for c in ['Red', 'Green', 'Other']:
        if c not in pivot.columns:
            pivot[c] = 0
    pivot['Total'] = pivot[['Red', 'Green', 'Other']].sum(axis=1)
    pivot['Red_Gap_Mean'] = merged[merged['group'] == 'Red'].groupby(index_col)['gap'].mean()
    pivot['Green_Gap_Mean'] = merged[merged['group'] == 'Green'].groupby(index_col)['gap'].mean()
    columns = ['Red', 'Green', 'Other', 'Total', 'Red_Gap_Mean', 'Green_Gap_Mean']

    if with_entity_lists:
        def entity_list(group_name):
            sub = merged[merged['group'] == group_name]
            return sub.sort_values('entity').groupby(index_col)['entity'].apply(', '.join)
        pivot['Red_Entities'] = entity_list('Red')
        pivot['Green_Entities'] = entity_list('Green')
        pivot['Other_Entities'] = entity_list('Other')
        for c in ['Red_Entities', 'Green_Entities', 'Other_Entities']:
            pivot[c] = pivot[c].fillna('')
        columns += ['Red_Entities', 'Green_Entities', 'Other_Entities']

    return pivot[columns].sort_values('Total', ascending=False)


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ucr_xlsx', default='./result/Experiment_2/Results/ucr_results.xlsx')
    parser.add_argument('--gap_threshold', type=float, default=0.5)
    parser.add_argument('--log_glob', default='./result/DS_*/**/*.log',
                         help='Glob (recursive) over old run logs to scrape "Entity meta-data" lines from.')
    parser.add_argument('--out_xlsx', default='./result/DS_2/achievability/ucr_domain_group_analysis.xlsx')
    args = parser.parse_args()

    domains = scrape_entity_domains(args.log_glob)
    print(f'{len(domains)} entities with a scraped domain/category')
    gaps = classify_gap(args.ucr_xlsx, args.gap_threshold)
    print(f'{len(gaps)} entities with a VUS_ROC gap/group')

    merged = domains.merge(gaps, on='entity', how='inner')
    missing = set(gaps['entity']) - set(domains['entity'])
    if missing:
        print(f'[warn] {len(missing)} gap-scored entities had no scraped domain: {sorted(missing)}')

    domain_table = build_group_table(merged, 'domain', with_entity_lists=True)
    variant_table = build_group_table(merged, 'variant')

    os.makedirs(os.path.dirname(args.out_xlsx), exist_ok=True)
    with pd.ExcelWriter(args.out_xlsx) as writer:
        domain_table.to_excel(writer, sheet_name='By Domain')
        variant_table.to_excel(writer, sheet_name='By Distorted-Noise-Plain')
    print(f'Wrote {args.out_xlsx} ({len(domain_table)} domains, {len(variant_table)} variants)')


if __name__ == '__main__':
    run()
