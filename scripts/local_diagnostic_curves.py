"""
Shared library for DS_3's 4-panel diagnostic plot template (raw signal +
model reconstruction, MSE score, Classification score, blended Anomaly
Score), reused identically across Self train/val, Cross-AnomSim train/val,
and the UCR test set.

Not a new metric anywhere: mse_score/ce_score/score are exactly what
main.anomaly_scoreing() already computes for VUS-ROC/RF etc (same function,
reused as-is). Windows overlap with window_step=1, so a dense pass over any
raw chunk gives one mse_score/ce_score/score/reconstruction value per
timestep once the first window_size-1 (fake, zero-padded) positions are
skipped -- exactly full_reproduction_metrics.score_entity's own existing
alignment convention.

Each displayed page is its own small, self-contained local experiment, not
a slice out of one big whole-split pass: build_local_chunk carves out one
focus window (window_size) plus 2*window_size of real context on each side,
optionally injects a single anomaly instance into just the focus sub-range,
and dense_windows_from_chunk turns that small chunk into the window_step=1
window stack compute_dense_curves expects. Because mse_score/ce_score/score
are min-max normalized inside compute_dense_curves's call to
main.anomaly_scoreing, this normalization is now local to what's shown on
the page (see plot_diagnostic_page's docstring) rather than spanning the
whole entity/split -- each focus window + its context is treated as
basically its own individual little time series, run through the model on
its own. Position SELECTION (e.g. UCR test's real-anomaly-segment/argmax/
argmin picks) still needs a whole-split dense pass and stays on
full_reproduction_metrics.score_entity for that purpose; only the final
per-page rendering is local.
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


def _curves_from_model_outputs(inputs_np, predicted_np, pred_label_np, window_size):
    """Shared tail end of compute_dense_curves/compute_dense_curves_batch --
    everything after the model forward pass, given already-numpy outputs for
    ONE continuous chunk (anomaly_scoreing's smoothing/normalization is only
    correct within a single continuous window_step=1 pass)."""
    score, mse_score, ce_score = main.anomaly_scoreing(
        inputs_np, predicted_np, pred_label_np, return_components=True)
    B = inputs_np.shape[0]
    mse_raw = main.mse(inputs_np.reshape(B, -1), predicted_np.reshape(B, -1))

    pad = np.zeros(window_size - 1)
    return dict(
        raw_series=np.concatenate([pad, inputs_np[:, -1, 0]]),
        reconstruction=np.concatenate([pad, predicted_np[:, -1, 0]]),
        mse_score=np.concatenate([pad, mse_score]),
        ce_score=np.concatenate([pad, ce_score]),
        score=np.concatenate([pad, score]),
        mse_raw=np.concatenate([pad, mse_raw]),
    )


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
    exposed separately since mse_score's normalization is per-chunk (see
    plot_diagnostic_page's docstring), so its absolute scale can't be
    compared across pages/entities; mse_raw can."""
    with torch.no_grad():
        inputs = windows.to(device)
        predicted, pred_label, pred_enc = model(inputs)
    window_size = windows.shape[1]
    return _curves_from_model_outputs(inputs.cpu().numpy(), predicted.cpu().numpy(),
                                       pred_label.cpu().numpy(), window_size)


def compute_dense_curves_batch(windows_list, model, device):
    """windows_list: list of torch.Tensor (n_windows_i, window_size,
    n_features), one per independent local chunk (same window_size
    throughout; n_windows_i may differ, e.g. chunks clipped near an
    entity's own edges). Runs the model in a SINGLE forward pass over the
    concatenation of all of them -- a pure per-window computation with no
    cross-window dependency, so concatenating is safe -- to cut down on
    per-call Python/dispatch overhead when there are many small independent
    chunks (e.g. up to 12 types x N_SAMPLES per entity/split in
    build_self_train_val_diagnostics.py/build_anomsim_train_val_diagnostics.py).
    The model output is then split back apart and main.anomaly_scoreing/
    main.mse are run PER CHUNK separately (not batched) -- their
    box-convolution smoothing and [0,1] min-max normalization are only
    correct within one continuous chunk and must not blend across
    independent ones. Returns a list of curve dicts, same order/length as
    windows_list -- () if windows_list is empty."""
    if not windows_list:
        return []
    window_size = windows_list[0].shape[1]
    sizes = [w.shape[0] for w in windows_list]
    with torch.no_grad():
        inputs = torch.cat(windows_list, dim=0).to(device)
        predicted, pred_label, pred_enc = model(inputs)
    inputs_np = inputs.cpu().numpy()
    predicted_np = predicted.cpu().numpy()
    pred_label_np = pred_label.cpu().numpy()

    results = []
    offset = 0
    for size in sizes:
        sl = slice(offset, offset + size)
        offset += size
        results.append(_curves_from_model_outputs(inputs_np[sl], predicted_np[sl], pred_label_np[sl], window_size))
    return results


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


def dense_windows_from_chunk(chunk, window_size):
    """chunk: (n_features, L) -> torch.FloatTensor (L-window_size+1, window_size,
    n_features), matching compute_dense_curves's expected input shape. Used to
    turn a small LOCAL raw chunk (from build_local_chunk) into the same kind of
    window_step=1 dense window stack that used to come from a whole-split
    Loader_aug pass -- compute_dense_curves itself doesn't change."""
    windows = np.lib.stride_tricks.sliding_window_view(chunk, window_size, axis=1)  # (n_features, n_windows, window_size)
    windows = np.ascontiguousarray(windows.transpose(1, 2, 0))  # (n_windows, window_size, n_features)
    return torch.tensor(windows, dtype=torch.float32)


def build_local_chunk(Y, focus_start, focus_end, inject_fn=None):
    """Y: (n_features, n_time) the real, unmodified signal for one entity/split.
    Extracts a local chunk of up to 2*window_size real context on each side of
    the focus window [focus_start, focus_end) (clipped at the signal's own
    edges), then -- if inject_fn is given -- injects a SINGLE anomaly instance
    into ONLY the focus sub-range, leaving the context on both sides untouched
    real data. inject_fn(window) -> (injected_window, mask), same (n_features,
    window_size) shape as the window it's given; mask 0 where modified,
    matching RedLamp's anomaly_mask convention.

    Returns (chunk, local_focus_start, anomaly_spans):
      chunk: (n_features, chunk_len) -- feed through dense_windows_from_chunk.
      local_focus_start: the focus window's start index WITHIN chunk (usually
        2*window_size, less near the signal's own edges).
      anomaly_spans: [(start, end)] in chunk-local coordinates where inject_fn
        actually modified the signal -- empty if inject_fn is None (e.g. the
        'normal' section, or the real UCR test signal, which has no injection
        at all; real ground-truth anomaly spans there come from labels
        instead, computed separately by the caller)."""
    window_size = focus_end - focus_start
    n_time = Y.shape[1]
    chunk_start = max(0, focus_start - 2 * window_size)
    chunk_end = min(n_time, focus_end + 2 * window_size)
    chunk = np.array(Y[:, chunk_start:chunk_end], copy=True)
    local_focus_start = focus_start - chunk_start

    spans = []
    if inject_fn is not None:
        window = Y[:, focus_start:focus_end]
        injected, mask = inject_fn(window)
        chunk[:, local_focus_start:local_focus_start + window_size] = injected
        modified = (np.asarray(mask).min(axis=0) == 0).astype(int)
        spans = [(s + local_focus_start, e + local_focus_start)
                 for s, e in find_anomaly_segments(modified, max_segments=5)]
    return chunk, local_focus_start, spans


def display_bounds(focus_start, focus_end, window_size, total_len):
    d0 = max(0, focus_start - 2 * window_size)
    d1 = min(total_len, focus_end + 2 * window_size)
    return d0, d1


def plot_diagnostic_page(pdf, raw_series, series_list, focus_start, focus_end, window_size,
                          title='', real_anomaly_spans=None):
    """raw_series: 1D array, the real/injected input signal (shared across
    all series -- there's only ever one underlying signal per page, even
    when overlaying 2 models' curves). Pass focus_start=0, focus_end=len(raw_series)
    for a "whole series, no local zoom" page (build_ucr_test_diagnostics.py) --
    display_bounds then spans the entire series and the redundant "context"
    line/legend entry is skipped automatically (in_focus covers everything).
    series_list: list of dict(label, color, reconstruction, mse_score,
    ce_score, score, mse_raw, threshold) -- one dict per model (1 for
    Self/AnomSim-only pages, 2 for Self+Cross-AnomSim overlay on Test pages).
    threshold is optional: if given, drawn as a dotted horizontal line on the
    final anomaly-score panel only (e.g. TSB_UAD's RF threshold, mean(score)
    + 3*std(score) over that series' own score array).
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
            to [0,1] over whatever curve was actually passed in as `series`
            -- for build_self_train_val_diagnostics.py/
            build_anomsim_train_val_diagnostics.py that's the small LOCAL
            chunk built by build_local_chunk (this focus window + its
            context), so scores are only comparable WITHIN one page,  not
            across pages (compare against panel 1.25/1.5 for the
            unnormalized picture, which IS comparable across pages). For
            build_ucr_test_diagnostics.py the whole test split is passed in
            instead (matching the official RedLamp/TSB_UAD evaluation
            convention exactly, including its mean+3*std threshold), so
            those scores ARE the real published-metric values, comparable
            across entities. Smoothing also changes the shape, not just the
            scale, versus panels 1.25/1.5 -- it's a literal box-convolution
            low-pass filter, so sharp local spikes get attenuated/widened.
            Fixed y-axis [0,1] for comparability across pages."""
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
    # If the focus window covers the ENTIRE display range (e.g. focus_start=0,
    # focus_end=len(raw_series) for a whole-series page), the "focus" line
    # would draw directly on top of the "context" line and completely hide
    # it -- skip the redundant context line/legend entry in that case.
    if not in_focus.all():
        ax_raw.plot(x, raw_slice, color='#888888', linewidth=1.0, label='context (raw)')
    ax_raw.plot(x[in_focus], raw_slice[in_focus], color='#1f77b4', linewidth=1.4,
                label='focus window (raw)' if not in_focus.all() else 'raw')
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

    # Optional per-series threshold (e.g. TSB_UAD's RF threshold, mean(score)
    # + 3*std(score) over that series' own score array) drawn on the final
    # blended anomaly-score panel only.
    for i, s in enumerate(series_list):
        if s.get('threshold') is None:
            continue
        color = s.get('color', recon_colors[i % len(recon_colors)])
        ax_score.axhline(s['threshold'], color=color, linewidth=1.0, linestyle=':',
                          alpha=0.8, label=f"{s['label']} threshold")

    for span_start, span_end in (real_anomaly_spans or []):
        for ax in all_axes:
            ax.axvspan(span_start, span_end, color='#e34948', alpha=0.15)

    for ax in all_axes:
        # axvspan's rectangle is included in autoscale, so a real_anomaly_span
        # far outside [d0, d1) would otherwise stretch the x-axis to cover
        # both it and the focus window, squeezing the actual displayed curves
        # into a sliver -- pin the range back to the intended display window.
        ax.set_xlim(d0, d1)
        ax.legend(fontsize=7, loc='upper right')
        ax.set_xlabel('timestep')
        ax.tick_params(labelbottom=True)  # sharex hides tick labels on non-bottom axes by default
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def plot_window_inspector_page(pdf, curves_by_model, models_by_label, device, positions, window_size,
                                title='', n_cols=2):
    """One page, one subplot per position in `positions` (up to 10 fits a
    5x2 grid legibly) -- for each position t (the LAST timestep of some
    dense window), re-runs each model FRESH on just that single window
    (batch=1) to get its true, exact reconstruction across the WHOLE window
    (not the neighbor-window approximation panel 1's "reconstruction" curve
    gives, which only ever shows each window's own LAST-position output
    stitched together). This is the only way to see whether a model's error
    concentrates at a window's edges or is spread evenly across it -- e.g.
    to explain why panel 1.25 (pointwise, last-position-only) and panel 1.5
    (mse_raw, whole-window average) can disagree about which model does
    better at a given timestep.

    curves_by_model: dict label -> curves dict (must have raw_series,
    mse_raw, mse_score, ce_score, score -- the whole-split values already
    computed for that model, read here only to ANNOTATE each subplot, not
    recomputed). models_by_label: dict label -> loaded ConvAEC model,
    same keys as curves_by_model. All models share the same underlying
    raw_series (real signal, no injection -- built for the UCR test case).
    positions: iterable of int, each the last timestep of the window to
    inspect."""
    positions = list(positions)
    if not positions:
        return
    n_rows = int(np.ceil(len(positions) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.5 * n_cols, 3.2 * n_rows), squeeze=False)
    axes = axes.flatten()
    labels = list(curves_by_model.keys())
    colors = {labels[0]: '#e0883f', labels[1] if len(labels) > 1 else '': '#3fae59'}
    raw_series = next(iter(curves_by_model.values()))['raw_series']

    for i, (ax, t) in enumerate(zip(axes, positions)):
        start, end = t - window_size + 1, t + 1
        raw_window = raw_series[start:end]
        window_tensor = torch.tensor(raw_window.reshape(1, window_size, 1), dtype=torch.float32)
        ax.plot(np.arange(start, end), raw_window, color='#333333', linewidth=1.3, label='raw')

        title_lines = [f't={start}-{end - 1}']
        for label in labels:
            with torch.no_grad():
                predicted, _, _ = models_by_label[label](window_tensor.to(device))
            recon = predicted[0, :, 0].cpu().numpy()
            ax.plot(np.arange(start, end), recon, color=colors.get(label, '#888888'), linewidth=1.0,
                    linestyle='--', label=f'{label} recon')
            c = curves_by_model[label]
            title_lines.append(f'{label}: MSE_raw={c["mse_raw"][t]:.4f} MSE={c["mse_score"][t]:.2f} '
                                f'CE={c["ce_score"][t]:.2f} Anom={c["score"][t]:.2f}')
        ax.set_title('\n'.join(title_lines), fontsize=7)
        ax.tick_params(labelsize=6)
        if i == 0:
            ax.legend(fontsize=6, loc='upper right')

    for ax in axes[len(positions):]:
        ax.axis('off')

    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def plot_entity_gallery_page(pdf, raw, window_size, title='', real_anomaly_spans=None, n_windows=8):
    """One page per entity, 5 rows x 2 cols: the top row spans both columns
    and shows the entity's WHOLE raw series (real_anomaly_spans shaded, if
    given); the remaining 4 rows x 2 cols show n_windows (default 8)
    evenly-spaced example windows from within that same series -- reuses
    pick_sample_positions/window_bounds_from_end_index exactly as train/val
    sampling does elsewhere (those functions only assume a window must fit
    within [0, len(raw)), which holds for a plain raw array just as well as
    for a zero-padded dense curve). Used by build_anomsim_entity_gallery.py
    (no anomaly spans -- AnomSim has no ground-truth labels) and
    build_ucr_group_galleries.py (real ground-truth anomaly spans)."""
    n_rows = int(np.ceil(n_windows / 2))
    fig = plt.figure(figsize=(11, 3 + 2.3 * n_rows))
    gs = fig.add_gridspec(n_rows + 1, 2, height_ratios=[2] + [1] * n_rows)

    ax_top = fig.add_subplot(gs[0, :])
    ax_top.plot(raw, color='#333333', linewidth=0.8)
    for span_start, span_end in (real_anomaly_spans or []):
        ax_top.axvspan(span_start, span_end, color='#e34948', alpha=0.15)
    ax_top.set_title(title, fontsize=11)
    ax_top.set_xlabel('timestep')

    positions = pick_sample_positions(len(raw), window_size, n=n_windows)
    for i, t in enumerate(positions):
        start, end = window_bounds_from_end_index(t, window_size)
        ax = fig.add_subplot(gs[1 + i // 2, i % 2])
        ax.plot(np.arange(start, end), raw[start:end], color='#333333', linewidth=1.0)
        for span_start, span_end in (real_anomaly_spans or []):
            overlap_start, overlap_end = max(span_start, start), min(span_end, end)
            if overlap_end > overlap_start:
                ax.axvspan(overlap_start, overlap_end, color='#e34948', alpha=0.15)
        ax.set_title(f'window {start}-{end - 1}', fontsize=8)
        ax.tick_params(labelsize=7)

    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)
