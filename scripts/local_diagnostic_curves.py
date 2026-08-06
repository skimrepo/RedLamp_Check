"""
Shared library for DS_3's 4-panel diagnostic plot template (raw signal +
model reconstruction, MSE score, Classification score, blended Anomaly
Score), reused identically across Self train/val, Cross-AnomSim train/val,
and the UCR test set.

Not a new metric anywhere: mse_score/ce_score/score are exactly what
main.anomaly_scoreing() already computes for VUS-ROC/RF etc (same function,
reused as-is). Windows overlap with window_step=1, so a single dense pass
over an entity/split gives one mse_score/ce_score/score/reconstruction
value per timestep once the first window_size-1 (fake, zero-padded)
positions are skipped -- exactly full_reproduction_metrics.score_entity's
own existing alignment convention, just reused here for train/val splits
too (previously only used for the test split).

Plots only ever slice a small LOCAL display range out of these full-split
curves (one highlighted window + 2*window_size of context on each side);
the curves themselves are always computed over the whole entity/split and
cached to .npz so future diagnostics can reuse them without re-inference.
"""
import os

import numpy as np
import torch
from matplotlib import pyplot as plt

import main


def load_convaec_model(model_dir, params, device):
    model = main.ConvAEC(params).to(device)
    model.load_state_dict(torch.load(f'{model_dir}/bestmodel.pkl', map_location=device))
    model.eval()
    return model


def compute_dense_curves(windows, model, device):
    """windows: torch.Tensor (n_windows, window_size, n_features), CPU,
    ordered by position with window_step=1 (consecutive windows overlap by
    window_size-1). Returns dict(raw_series, reconstruction, mse_score,
    ce_score, score, mse_raw), each zero-padded at the front by
    window_size-1 so curve index t lines up with the raw signal's own
    timestep t.

    mse_raw is the per-window reconstruction MSE BEFORE
    convolve_minmax_score's smoothing+[0,1] normalization (main.mse(),
    the same call anomaly_scoreing() makes internally before smoothing) --
    exposed separately since mse_score's normalization is per-entity/split,
    so its absolute scale can't be compared across pages/entities; mse_raw
    can."""
    with torch.no_grad():
        inputs = windows.to(device)
        predicted, pred_label, pred_enc = model(inputs)
        inputs_np = inputs.cpu().numpy()
        predicted_np = predicted.cpu().numpy()
        score, mse_score, ce_score = main.anomaly_scoreing(
            inputs_np, predicted_np, pred_label.cpu().numpy(), return_components=True)
        B = inputs_np.shape[0]
        mse_raw = main.mse(inputs_np.reshape(B, -1), predicted_np.reshape(B, -1))

    window_size = windows.shape[1]
    pad = np.zeros(window_size - 1)
    return dict(
        raw_series=np.concatenate([pad, windows[:, -1, 0].numpy()]),
        reconstruction=np.concatenate([pad, predicted[:, -1, 0].cpu().numpy()]),
        mse_score=np.concatenate([pad, mse_score]),
        ce_score=np.concatenate([pad, ce_score]),
        score=np.concatenate([pad, score]),
        mse_raw=np.concatenate([pad, mse_raw]),
    )


def save_curves_npz(path, curves):
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.savez(path, **curves)


def load_curves_npz(path):
    if not os.path.isfile(path):
        return None
    data = np.load(path)
    return {k: data[k] for k in data.files}


def get_or_compute_curves(cache_path, compute_fn, force=False):
    """compute_fn: zero-arg callable returning a curves dict (only invoked
    on a cache miss or --force)."""
    if not force:
        cached = load_curves_npz(cache_path)
        if cached is not None:
            return cached
    curves = compute_fn()
    save_curves_npz(cache_path, curves)
    return curves


def window_bounds_from_end_index(end_idx, window_size):
    """Curve index `end_idx` is the LAST timestep of some window (see
    compute_dense_curves's alignment) -- returns that window's
    [start, end) span in the same timestep coordinates."""
    return end_idx - window_size + 1, end_idx + 1


def pick_sample_positions(curve_len, window_size, n=5):
    """n evenly-spaced center positions within the curve's valid (non-zero-
    padded) range -- used to show n DIFFERENT display windows for the same
    cached (entity, split, type) curve, e.g. because dense window_step=1
    injection draws an independent random instance of that anomaly type at
    every position, so different positions genuinely show different
    injected samples, not just different crops of the same one.

    Prefers positions with a full 2*window_size of real context on both
    sides (matching plot_diagnostic_page's own display_bounds) so pages
    aren't dominated by zero-padding or an abruptly-truncated end -- falls
    back to the full valid range only if the curve is too short for that."""
    valid_start = window_size - 1
    valid_end = curve_len - 1
    safe_start = valid_start + 2 * window_size
    safe_end = valid_end - 2 * window_size
    if safe_end > safe_start:
        valid_start, valid_end = safe_start, safe_end
    if valid_end <= valid_start:
        return [valid_start] * n
    return [int(round(x)) for x in np.linspace(valid_start, valid_end, n)]


def find_anomaly_segments(real_labels, max_segments=5):
    """Contiguous runs of 1s in a binary label array, longest first, capped
    at max_segments. Returns a list of (start, end) [inclusive, exclusive)."""
    labels = np.asarray(real_labels)
    if labels.sum() == 0:
        return []
    padded = np.concatenate([[0], labels, [0]])
    diff = np.diff(padded)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    segments = list(zip(starts.tolist(), ends.tolist()))
    segments.sort(key=lambda se: se[1] - se[0], reverse=True)
    return segments[:max_segments]


def pick_extreme_positions(curves, window_size):
    """argmax/argmin of mse_score and ce_score, restricted to real
    (non-zero-padded) positions -- the first window_size-1 entries are fake
    zeros from alignment padding, not real data, and must be excluded or
    argmin would trivially always land there."""
    offset = window_size - 1
    mse = curves['mse_score'][offset:]
    ce = curves['ce_score'][offset:]
    return dict(
        mse_argmax=int(np.argmax(mse)) + offset,
        mse_argmin=int(np.argmin(mse)) + offset,
        ce_argmax=int(np.argmax(ce)) + offset,
        ce_argmin=int(np.argmin(ce)) + offset,
    )


def merge_nearby_positions(labeled_positions, window_size):
    """labeled_positions: list of (timestep, criterion_label). Positions
    within window_size of each other (i.e. their highlighted windows would
    overlap/be redundant to show as separate pages) are merged into one,
    combining criterion labels into a single page title."""
    ordered = sorted(labeled_positions, key=lambda lp: lp[0])
    merged = []
    for pos, label in ordered:
        if merged and pos - merged[-1][0] < window_size:
            merged[-1] = (merged[-1][0], merged[-1][1] + [label])
        else:
            merged.append((pos, [label]))
    return merged


def display_bounds(focus_start, focus_end, window_size, total_len):
    d0 = max(0, focus_start - 2 * window_size)
    d1 = min(total_len, focus_end + 2 * window_size)
    return d0, d1


def plot_diagnostic_page(pdf, raw_series, series_list, focus_start, focus_end, window_size,
                          title='', real_anomaly_spans=None):
    """raw_series: 1D array, the real/injected input signal (shared across
    all series -- there's only ever one underlying signal per page, even
    when overlaying 2 models' curves).
    series_list: list of dict(label, color, reconstruction, mse_score,
    ce_score, score, mse_raw) -- one dict per model (1 for Self/AnomSim-only
    pages, 2 for Self+Cross-AnomSim overlay on Test pages).
    real_anomaly_spans: optional list of (start, end) to shade in every panel.

    6 panels, in order:
      1.    raw signal (context/focus) + reconstruction
      1.25  |raw - reconstruction|, pointwise -- no windowing, no smoothing,
            no normalization; the most literal "how far off is it right
            here" view, free y-axis.
      1.5   MSE (raw): main.mse()'s own per-window mean squared error,
            BEFORE convolve_minmax_score's smoothing (box-convolution,
            width window_size/2) and [0,1] normalization -- free y-axis.
            This will generally look smoother than panel 1.25 (it's a
            per-window average, not per-point) but is still the
            pre-smoothing/pre-normalization value.
      2-4.  MSE_Norm_Smooth / CE_Norm_Smooth / Anomaly_Norm_Smooth scores --
            exactly mse_score/ce_score/score from main.anomaly_scoreing,
            i.e. panel 1.5 (or its CE-side equivalent) run through
            convolve_minmax_score. Each is independently min-max normalized
            to [0,1] over the WHOLE entity/split (not just this local
            display slice), so a flat-looking curve here doesn't mean flat
            raw error -- it means this local slice sits in a narrow part of
            that entity's own full min-max range (compare against panel
            1.25/1.5 for the unnormalized picture). Smoothing also changes
            the shape, not just the scale, versus panels 1.25/1.5 -- it's a
            literal box-convolution low-pass filter, so sharp local spikes
            get attenuated/widened. Fixed y-axis [0,1] for comparability
            across pages."""
    total_len = len(raw_series)
    d0, d1 = display_bounds(focus_start, focus_end, window_size, total_len)
    x = np.arange(d0, d1)
    raw_slice = raw_series[d0:d1]
    in_focus = (x >= focus_start) & (x < focus_end)

    fig, (ax_raw, ax_abs_err, ax_mse_raw, ax_mse, ax_ce, ax_score) = plt.subplots(
        6, 1, figsize=(11, 14), sharex=True)
    all_axes = (ax_raw, ax_abs_err, ax_mse_raw, ax_mse, ax_ce, ax_score)

    # Draw the FULL context line first (one continuous segment, no gaps),
    # then overlay just the focus span in its own color on top -- masking
    # both pieces via x[~in_focus]/x[in_focus] instead would leave a straight
    # line falsely bridging the excised gap (matplotlib always connects
    # consecutive points in whatever's passed to plot(), and boolean masking
    # makes the two context sub-arrays adjacent even though there's a real
    # gap between them).
    ax_raw.plot(x, raw_slice, color='#888888', linewidth=1.0, label='context (raw)')
    ax_raw.plot(x[in_focus], raw_slice[in_focus], color='#1f77b4', linewidth=1.4, label='focus window (raw)')
    recon_colors = ['#e0883f', '#3fae59']
    for i, s in enumerate(series_list):
        color = s.get('color', recon_colors[i % len(recon_colors)])
        ax_raw.plot(x, s['reconstruction'][d0:d1], color=color, linewidth=1.0, linestyle='--',
                    alpha=0.85, label=f"{s['label']} reconstruction")
    ax_raw.set_ylabel('raw / reconstruction')

    for i, s in enumerate(series_list):
        color = s.get('color', recon_colors[i % len(recon_colors)])
        abs_err = np.abs(raw_slice - s['reconstruction'][d0:d1])
        ax_abs_err.plot(x, abs_err, color=color, linewidth=1.1, alpha=0.9, label=s['label'])
    ax_abs_err.set_ylabel('|raw - reconstruction|')

    for i, s in enumerate(series_list):
        color = s.get('color', recon_colors[i % len(recon_colors)])
        ax_mse_raw.plot(x, s['mse_raw'][d0:d1], color=color, linewidth=1.1, alpha=0.9, label=s['label'])
    ax_mse_raw.set_ylabel('MSE (raw)')

    for ax, key, ylabel in [(ax_mse, 'mse_score', 'MSE_Norm_Smooth score'),
                             (ax_ce, 'ce_score', 'CE_Norm_Smooth score'),
                             (ax_score, 'score', 'Anomaly_Norm_Smooth score')]:
        for i, s in enumerate(series_list):
            color = s.get('color', recon_colors[i % len(recon_colors)])
            ax.plot(x, s[key][d0:d1], color=color, linewidth=1.1, alpha=0.9, label=s['label'])
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 1)

    for span_start, span_end in (real_anomaly_spans or []):
        for ax in all_axes:
            ax.axvspan(span_start, span_end, color='#e34948', alpha=0.15)

    for ax in all_axes:
        ax.legend(fontsize=7, loc='upper right')
        ax.set_xlabel('timestep')
        ax.tick_params(labelbottom=True)  # sharex hides tick labels on non-bottom axes by default
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)
