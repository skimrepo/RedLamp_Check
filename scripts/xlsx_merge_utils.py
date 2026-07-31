"""
Shared helpers for building the Experiment_1/Experiment_2 result workbooks:
3-way merging Self/Cross-OpenSource/Cross-AnomSim frames that share
identical column sets (accuracy/val_windows for Experiment_1's classification
accuracy, or the 5 VUS metrics + peak_in_range for Experiment_2's real-test-
set metrics). Kept in its own module (not inside organize_experiment1.py) so
organize_experiment2.py doesn't need to import organize_experiment1's own
model-moving dependencies (cross_inference, full_reproduction_metrics) just
to reuse these two small functions.
"""
import pandas as pd


def suffix_cols(df, suffix, keep=('entity',)):
    """Renames every column except `keep` by appending _suffix."""
    if df.empty:
        return df
    return df.rename(columns={c: f'{c}_{suffix}' for c in df.columns if c not in keep})


def merge_3way(frames_with_suffix):
    """frames_with_suffix: list of (df, suffix) pairs. Renames non-entity
    columns per frame, then outer-merges whichever frames are non-empty on
    'entity'. Returns an empty DataFrame if all inputs are empty."""
    renamed = [r for r in (suffix_cols(df, suffix) for df, suffix in frames_with_suffix) if not r.empty]
    merged = pd.DataFrame()
    for r in renamed:
        merged = r if merged.empty else merged.merge(r, on='entity', how='outer')
    return merged
