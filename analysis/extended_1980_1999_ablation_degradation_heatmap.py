"""Design-component importance heatmap for the Extended1980_1999 Camels531 ablations.

Each cell is the % degradation of one metric for one ablation, measured
against the proposed method (the condensed 128-d daily MFFormer256 embedding
+ dual-residual adapter). Positive always means worse, whichever direction
the underlying metric runs.

Experiment: camels_531, 10-fold PUB, train 1980/10/01-1999/09/30, test
1995/10/01-1999/09/30, 3 seeds (111111/222222/333333).

Two stages, so the figure is reproducible without the scratch runs:

  1. collect  -- recompute per-basin metrics from the archived
     aggregated_predictions/targets .npy arrays, average over seeds per basin,
     and write BOTH the per-basin arrays (.npz) and a tidy summary (.csv),
     plus a provenance .json naming every run dir and seed that went in.
  2. plot     -- render the heatmap from those saved files alone.

    python extended_1980_1999_ablation_degradation_heatmap.py           # collect + plot
    python extended_1980_1999_ablation_degradation_heatmap.py --replot  # plot from cache

Metrics are recomputed from the arrays rather than read from each run's
metrics.json: collect_extended_1980_1999.py archives both, but a stored
metrics.json goes stale if a run is ever re-tested, while the arrays are the
primary artifact. The two agree to <1e-14 on runs that have not been re-tested.
"""

import argparse
import json
import os
import subprocess
import warnings
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = '${oc.env:DMG_DATA_ROOT}/HBVPaperResults/Extended1980_1999_Camels531'
SCRATCH_ROOT = '${oc.env:DMG_OUTPUT_ROOT}/Extended1980_1999_Camels531'
OUT_DIR = '${oc.env:DMG_DATA_ROOT}/HBVPaperResults/Extended1980_1999_ablation_degradation'

SEEDS = ['seed111111', 'seed222222', 'seed333333']
SPLIT = 'PUB'

# The proposed method every cell is measured against.
BASELINE = ('Daily EMB', 'Camels_531_condensed_embeddings_daily')

# Ablation variants, ordered as columns (least to most damaging).
ABLATIONS = [
    ('Scratch',    'NoPretrainingHBV'),
    ('FM Data as Inputs', 'RawFmInputs'),
    ('EMB as Statics',    'Camels_531_condensed_embeddings_daily_asstatics'),
    ('No Adapter',        'Camels_531_condensed_embeddings_daily_noadapter'),
    ('Linear Probe',      'Camels_531_condensed_embeddings_daily_linearprobe'),
]

# (key, row label, direction). 'higher' = larger is better.
METRICS = [
    ('nse',  'NSE',  'higher'),
    ('kge',  'KGE',  'higher'),
    ('corr', 'Corr', 'higher'),
    ('rmse', 'RMSE', 'lower'),
]

NPZ_PATH = os.path.join(OUT_DIR, 'ablation_metrics_perbasin.npz')
CSV_PATH = os.path.join(OUT_DIR, 'ablation_summary.csv')
PROV_PATH = os.path.join(OUT_DIR, 'provenance.json')
FIG_STEM = os.path.join(OUT_DIR, 'Extended1980_1999_ablation_degradation_heatmap')

# Times New Roman where installed; Nimbus Roman is the metric-compatible URW
# clone shipped on this machine and is what actually renders here.
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'Nimbus Roman',
                              'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'stix'


def compute_basin_metrics(pred, target):
    """Per-basin metrics from raw (time, basins) arrays.

    Mirrors dmg.core.calc.metrics.Metrics: predictions and targets are jointly
    masked on NaN per basin, a basin needs >1 jointly-valid timestep to be
    scoreable, and spread terms use population (ddof=0) moments.
    """
    if target.ndim == 3:
        target = np.squeeze(target, axis=-1)
    if pred.ndim == 3:
        pred = np.squeeze(pred, axis=-1)

    mask = np.isfinite(pred) & np.isfinite(target)
    n = mask.sum(axis=0)
    n_safe = np.maximum(n, 1)

    p = np.where(mask, pred, 0.0).astype(np.float64)
    t = np.where(mask, target, 0.0).astype(np.float64)

    mean_p = p.sum(axis=0) / n_safe
    mean_t = t.sum(axis=0) / n_safe
    dp = np.where(mask, p - mean_p, 0.0)
    dt = np.where(mask, t - mean_t, 0.0)

    with np.errstate(divide='ignore', invalid='ignore'):
        resid = np.where(mask, p - t, 0.0)
        sse = (resid ** 2).sum(axis=0)
        sst = (dt ** 2).sum(axis=0)
        nse = 1.0 - sse / sst

        std_p = np.sqrt((dp ** 2).sum(axis=0) / n_safe)
        std_t = np.sqrt((dt ** 2).sum(axis=0) / n_safe)
        corr = ((dp * dt).sum(axis=0) / n_safe) / (std_p * std_t)
        kge = 1.0 - np.sqrt(
            (corr - 1.0) ** 2
            + (std_p / std_t - 1.0) ** 2
            + (mean_p / mean_t - 1.0) ** 2
        )

        rmse = np.sqrt(sse / n_safe)
        pbias_abs = np.abs(resid.sum(axis=0) / t.sum(axis=0)) * 100.0

    out = {'nse': nse, 'kge': kge, 'corr': corr,
           'rmse': rmse, 'pbias_abs': pbias_abs}
    for v in out.values():
        v[n <= 1] = np.nan
    return out


_target_ref = [None]


def check_targets_match(target, where):
    """Warn if a run's targets differ from the first run seen.

    Every series here is evaluated on the same 531 basins over the same
    window, so target arrays must be identical. When they are not, the run was
    launched against a different dataset or basin list and its degradation is
    computed over a different population.
    """
    if target.ndim == 3:
        target = np.squeeze(target, axis=-1)
    if _target_ref[0] is None:
        _target_ref[0] = (target, where)
        return
    ref_arr, ref_where = _target_ref[0]
    ok = (target.shape == ref_arr.shape and np.array_equal(
        np.nan_to_num(target, nan=-9e9), np.nan_to_num(ref_arr, nan=-9e9)))
    if not ok:
        warnings.warn(
            'targets differ from {}: {} has shape {} -- not evaluated on the '
            'same basins, degradation not comparable'.format(
                ref_where, where, target.shape), stacklevel=2)


def summarise(v):
    """(median, mean, n) over the finite basins only.

    A few basins are degenerate rather than merely unscoreable: a constant
    target gives sst = 0 -> NSE = -inf. Those are +-inf, not NaN, so nanmean/nanmedian propagate
    them. Restricting every statistic to the finite subset keeps median, mean
    and the reported basin count describing the same population.
    """
    vf = v[np.isfinite(v)]
    if vf.size == 0:
        return np.nan, np.nan, 0
    return float(np.median(vf)), float(np.mean(vf)), int(vf.size)


def find_run_dirs(model_dir):
    """Seed dirs for a series, from the archive, else scratch. May be empty."""
    for root in (ROOT, SCRATCH_ROOT):
        split = os.path.join(root, model_dir, SPLIT)
        dirs = [os.path.join(split, s, 'spatial_aggregated_{}'.format(SPLIT))
                for s in SEEDS]
        dirs = [d for d in dirs
                if os.path.exists(os.path.join(d, 'aggregated_predictions.npy'))
                and os.path.exists(os.path.join(d, 'aggregated_targets.npy'))]
        if dirs:
            return dirs
    return []


def load_series(model_dir):
    """Per-basin metrics for one series, seed-averaged over its replicates.

    Seed-averaging per basin before summarising is valid because every model
    and seed shares identical targets, so basin ordering matches. Averaging is
    NaN-safe: a basin one seed failed to score still contributes its others.
    """
    run_dirs = find_run_dirs(model_dir)
    if not run_dirs:
        return None, [], []

    per_run = {k: [] for k, _, _ in METRICS}
    for d in run_dirs:
        pred = np.load(os.path.join(d, 'aggregated_predictions.npy'))
        target = np.load(os.path.join(d, 'aggregated_targets.npy'))
        check_targets_match(target, os.path.relpath(d, os.path.dirname(ROOT)))
        vals = compute_basin_metrics(pred, target)
        for k, _, _ in METRICS:
            per_run[k].append(vals[k])

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        out = {k: np.nanmean(np.vstack(per_run[k]), axis=0) for k in per_run}
    return out, run_dirs, [os.path.basename(os.path.dirname(d)) for d in run_dirs]


def degradation(base_med, abl_med, direction):
    """% change of a median, signed so positive is always worse."""
    if not np.isfinite(base_med) or not np.isfinite(abl_med):
        return np.nan
    delta = (base_med - abl_med) if direction == 'higher' else (abl_med - base_med)
    return 100.0 * delta / abs(base_med)


def git_commit():
    try:
        return subprocess.check_output(
            ['git', '-C', '${oc.env:DMG_REPO}',
             'rev-parse', '--short', 'HEAD'],
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return 'unknown'


def collect():
    os.makedirs(OUT_DIR, exist_ok=True)
    series = [BASELINE] + ABLATIONS
    arrays, prov, rows = {}, {}, []

    base_vals = None
    for label, model_dir in series:
        vals, run_dirs, seeds = load_series(model_dir)
        prov[label] = {'model_dir': model_dir, 'split': SPLIT,
                       'run_dirs': run_dirs, 'seeds': seeds,
                       'n_seeds': len(run_dirs)}
        if vals is None:
            print('  {:<24} NO RESULTS ({})'.format(label, model_dir))
            continue
        if label == BASELINE[0]:
            base_vals = vals
        for k, _, _ in METRICS:
            arrays['{}|{}'.format(label, k)] = vals[k]
        print('  {:<24} {} seed(s), {} scoreable basins'.format(
            label, len(run_dirs), int(np.isfinite(vals['nse']).sum())))

    if base_vals is None:
        raise SystemExit('Baseline {} has no results.'.format(BASELINE[0]))

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        for label, _ in series:
            for k, mlabel, direction in METRICS:
                key = '{}|{}'.format(label, k)
                base_med, _, _ = summarise(base_vals[k])
                if key not in arrays:
                    rows.append((label, k, mlabel, direction, 0, 0,
                                 np.nan, np.nan, np.nan))
                    continue
                med, mean, n_fin = summarise(arrays[key])
                rows.append((label, k, mlabel, direction,
                             prov[label]['n_seeds'], n_fin, med, mean,
                             degradation(base_med, med, direction)))

    np.savez_compressed(NPZ_PATH, **arrays)

    with open(CSV_PATH, 'w') as f:
        f.write('series,metric,metric_label,direction,n_seeds,'
                'n_basins_scoreable,median,mean,degradation_pct_vs_baseline\n')
        for r in rows:
            f.write('{},{},{},{},{},{},{},{},{}\n'.format(
                r[0], r[1], r[2], r[3], r[4], r[5],
                '' if not np.isfinite(r[6]) else '{:.6f}'.format(r[6]),
                '' if not np.isfinite(r[7]) else '{:.6f}'.format(r[7]),
                '' if not np.isfinite(r[8]) else '{:.6f}'.format(r[8])))

    with open(PROV_PATH, 'w') as f:
        json.dump({'generated': datetime.now().isoformat(timespec='seconds'),
                   'dmg_dev_commit': git_commit(),
                   'experiment': 'Extended1980_1999_Camels531',
                   'split': SPLIT,
                   'baseline': {'label': BASELINE[0], 'model_dir': BASELINE[1]},
                   'metrics_recomputed_from': 'aggregated_predictions/targets .npy',
                   'series': prov}, f, indent=2)

    print('\nWrote:\n  {}\n  {}\n  {}'.format(NPZ_PATH, CSV_PATH, PROV_PATH))
    return rows


def load_cached_rows():
    if not os.path.exists(CSV_PATH):
        raise SystemExit('No cache at {} -- run without --replot first.'
                         .format(CSV_PATH))
    rows = []
    with open(CSV_PATH) as f:
        next(f)
        for line in f:
            p = line.rstrip('\n').split(',')
            rows.append((p[0], p[1], p[2], p[3], int(p[4]), int(p[5]),
                         float(p[6]) if p[6] else np.nan,
                         float(p[7]) if p[7] else np.nan,
                         float(p[8]) if p[8] else np.nan))
    return rows


def plot(rows):
    os.makedirs(OUT_DIR, exist_ok=True)
    deg = {(r[0], r[1]): r[8] for r in rows}

    col_labels = [lab for lab, _ in ABLATIONS]
    row_labels = [ml for _, ml, _ in METRICS]
    D = np.array([[deg.get((c, k), np.nan) for c in col_labels]
                  for k, _, _ in METRICS], dtype=float)

    finite = D[np.isfinite(D)]
    vmax = float(np.nanmax(finite)) if finite.size else 1.0
    vmin = min(0.0, float(np.nanmin(finite))) if finite.size else 0.0

    fig, ax = plt.subplots(figsize=(1.55 * len(col_labels) + 4.4,
                                    0.82 * len(row_labels) + 3.0))

    im = ax.imshow(np.ma.masked_invalid(D), cmap='Reds', vmin=vmin, vmax=vmax,
                   aspect='auto')
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            v = D[i, j]
            if not np.isfinite(v):
                ax.add_patch(plt.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1, facecolor='#f4f4f4',
                    edgecolor='#bbbbbb', hatch='///', linewidth=0.8, zorder=2))
                ax.text(j, i, 'pending', ha='center', va='center', fontsize=11,
                        style='italic', color='#888888', zorder=3)
                continue
            ax.text(j, i, '{:.1f}%'.format(v), ha='center', va='center',
                    fontsize=15,
                    color='white' if norm(v) > 0.55 else '#1a1a1a', zorder=3)

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=13, rotation=20, ha='right')
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=13)
    ax.set_xlabel('Ablation Variant', fontsize=15, labelpad=10)
    ax.set_ylabel('Metric', fontsize=15, labelpad=10)
    ax.set_title('Design Component Importance (% degradation)',
                 fontsize=19, pad=16)

    ax.set_xticks(np.arange(-0.5, len(col_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=2.2)
    ax.tick_params(which='minor', length=0)
    ax.tick_params(which='major', length=0)
    for s in ax.spines.values():
        s.set_edgecolor('#333333')
        s.set_linewidth(1.2)

    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cbar.set_label('Degradation vs. {} (%)'.format(BASELINE[0]), fontsize=12)
    cbar.ax.tick_params(labelsize=11)

    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig('{}.{}'.format(FIG_STEM, ext), dpi=300,
                    bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('Wrote:\n  {}.png\n  {}.pdf'.format(FIG_STEM, FIG_STEM))

    print('\nDegradation vs {} (%):'.format(BASELINE[0]))
    print(' ' * 10 + ''.join('{:>16}'.format(c) for c in col_labels))
    for i, rl in enumerate(row_labels):
        cells = ''.join(
            '{:>16}'.format('pending' if not np.isfinite(D[i, j])
                            else '{:+.1f}'.format(D[i, j]))
            for j in range(len(col_labels)))
        print('{:<10}{}'.format(rl, cells))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--replot', action='store_true',
                    help='Plot from the saved csv/npz without re-reading runs.')
    args = ap.parse_args()

    if args.replot:
        plot(load_cached_rows())
    else:
        print('Collecting per-basin metrics...')
        plot(collect())
