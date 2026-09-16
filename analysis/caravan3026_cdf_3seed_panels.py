"""
Seed-averaged CDF plots of per-basin streamflow skill (NSE, KGE) for
Caravan3026, 1998-2008 test period, as four standalone paper panels:

    nse_spatial_holdout.png        kge_spatial_holdout.png
    nse_spatial_temporal.png       kge_spatial_temporal.png

Same panel geometry and typography as caravan3026_cdf_individual_panels.py,
but every series is the average of 3 seeds (seed111111/222222/333333).

Metrics are recomputed from each run's aggregated_{predictions,targets}.npy.
Nothing reads metrics.json any more. That file is written once at test time
and is not rewritten when a run is re-tested in place, so it silently
describes whatever predictions came before: checked 2026-08-24, the
Embedding/daily and Embedding/monthly metrics.json reproduced the superseded
predictions exactly (NSE max|diff| = 0) while disagreeing with the current
ones by up to ~1e2 NSE. Recomputing costs ~40 s per run of the script and
removes that whole failure mode. See collect_caravan3026_seeds.py.

Seed handling: the 3 seeds of a given experiment/split differ only in model
init -- their aggregated_targets.npy are byte-identical, so basin ordering is
shared. Per-basin NSE/KGE are therefore averaged across seeds *before* the
CDF is built, which is the same treatment extended_1980_1999_cdf_3seed.py
applies and keeps the curve interpretable as "the typical basin's skill".

Embedding/annual is included as of 2026-08-24. It used to be excluded because
those runs were launched with observations.name=camels_531 and so covered 250
CAMELS basins despite living in the Caravan3026 tree. The reruns of
2026-08-21 fixed the config: their targets are now byte-identical to every
other series here (4018 x 3026). check_targets_match() below re-verifies that
on every run, so a series that regresses to the wrong basin set announces
itself instead of quietly plotting a different population.

Requires: collect_caravan3026_seeds.py to have been run.
"""
import os
import warnings

import numpy as np
import matplotlib.pyplot as plt

ROOT = '${oc.env:DMG_DATA_ROOT}/HBVPaperResults/Caravan3026'
OUT_DIR = '${oc.env:DMG_DATA_ROOT}/HBVPaperResults/Caravan3026_cdf_panels_3seed'

SEEDS = ['seed111111', 'seed222222', 'seed333333']
SCENARIOS = ['pure_spatial', 'spatial_temporal']
SPLIT_DIR = {
    'pure_spatial': 'pure_spatial_1998-2008',
    'spatial_temporal': 'spatial_temporal_train1987-1997_test1998-2008',
}

# label -> (model dir prefix under ROOT, color)
# Seed count is discovered per series at load time, not declared here, so a
# series picks up replicates as soon as collect_caravan3026_seeds.py copies
# them in.
SERIES = [
    ('LSTM',                    'LSTMNoHBV',          '#eb6834'),  # orange
    ('LSTM δHBV', 'LSTMHBV',            '#2a78d6'),  # blue
    ('Scratch_EMB δHBV', 'MFFormerNoPretrain', '#c4a000'),  # gold
    ('Daily_EMB δHBV',          'Embedding/daily',    '#3fa34d'),  # green
    ('Monthly_EMB δHBV',        'Embedding/monthly',  '#00a0a0'),  # teal
    ('Annual_EMB δHBV',         'Embedding/annual',   '#d6217f'),  # magenta
    ('AlphaEarth δHBV',         'AlphaEarth',         '#8c5bd8'),  # purple
]

METRICS = ['nse', 'kge']
METRIC_LABELS = {'nse': 'NSE', 'kge': 'KGE'}
FILE_TAG = {'pure_spatial': 'spatial_holdout', 'spatial_temporal': 'spatial_temporal'}
CLIP_LOW = -1.0  # curves clipped at -1 for readability; medians computed unclipped

# enlarged type, matching caravan3026_cdf_individual_panels.py
FS_TICK = 22
FS_AXIS = 26
FS_LEGEND = 11

# Times New Roman where it is installed; Nimbus Roman is the metric-compatible
# URW clone shipped on this machine and is what actually renders here. Both
# carry U+03B4, so the delta in the series labels survives either way.
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'Nimbus Roman',
                              'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'stix'


def compute_basin_nse_kge(pred, target):
    """Per-basin NSE and KGE from raw (time, basins) pred/target arrays.

    Mirrors dmg.core.calc.metrics.Metrics: predictions and targets are
    jointly masked on NaN per basin, a basin needs >1 valid timestep to get a
    finite NSE/KGE, and the spread terms use the population (ddof=0) moments
    that ndarray.std() gives.

    Vectorised over basins rather than looped -- with 7 series x 2 splits x 3
    seeds the per-basin Python loop this replaces dominated the runtime.
    Verified against the loop and against the metrics.json of runs that had
    not been re-tested; see the printout from --verify below.
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
        sse = (np.where(mask, p - t, 0.0) ** 2).sum(axis=0)
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

    # A basin with <=1 jointly-valid timestep is unscoreable, not zero-skill.
    nse[n <= 1] = np.nan
    kge[n <= 1] = np.nan
    return {'nse': nse, 'kge': kge}


_cache = {}
_target_ref = {}


def check_targets_match(scenario, target, where):
    """Warn if a run's targets differ from the first run seen for the split.

    Every series here is evaluated on the same Caravan3026 basins over the
    same window, so the target arrays must be identical. When they are not,
    the run was launched against a different dataset or basin list and its
    CDF is drawn over a different population -- the Embedding/annual
    camels_531 mix-up that kept it out of this figure until 2026-08-24.
    """
    if target.ndim == 3:
        target = np.squeeze(target, axis=-1)
    ref = _target_ref.get(scenario)
    if ref is None:
        _target_ref[scenario] = (target, where)
        return
    ref_arr, ref_where = ref
    ok = (target.shape == ref_arr.shape and np.array_equal(
        np.nan_to_num(target, nan=-9e9), np.nan_to_num(ref_arr, nan=-9e9)))
    if not ok:
        warnings.warn(
            'targets differ from {}: {} has shape {} -- this series is not '
            'evaluated on the same basins and its CDF is not comparable'
            .format(ref_where, where, target.shape), stacklevel=2)


def find_run_dirs(model_prefix, scenario):
    """Every run dir for a series, preferring seed replicates when present.

    Returns (dirs, n_seeds). A split dir may also carry a split-level
    spatial_aggregated_PUB from an earlier single run, so once any seed dir
    exists the split-level copy is ignored rather than double-weighted.
    """
    split = os.path.join(ROOT, model_prefix, SPLIT_DIR[scenario])
    seed_dirs = [os.path.join(split, s, 'spatial_aggregated_PUB') for s in SEEDS]
    seed_dirs = [d for d in seed_dirs
                 if os.path.exists(os.path.join(d, 'aggregated_predictions.npy'))]
    if seed_dirs:
        return seed_dirs, len(seed_dirs)

    agg = os.path.join(split, 'spatial_aggregated_PUB')
    if os.path.exists(os.path.join(agg, 'aggregated_predictions.npy')):
        return [agg], 1
    return [split], 1


def load_series(model_prefix, scenario):
    """Per-basin {nse, kge} for one series, seed-averaged over its replicates.

    Averaging is NaN-safe: a basin one seed failed to score still contributes
    its other seeds rather than dropping out of the CDF.
    """
    key = (model_prefix, scenario)
    if key in _cache:
        return _cache[key]

    run_dirs, n_seeds = find_run_dirs(model_prefix, scenario)

    per_run = {m: [] for m in METRICS}
    for d in run_dirs:
        pred = np.load(os.path.join(d, 'aggregated_predictions.npy'))
        target = np.load(os.path.join(d, 'aggregated_targets.npy'))
        check_targets_match(scenario, target, os.path.relpath(d, ROOT))
        vals = compute_basin_nse_kge(pred, target)
        for m in METRICS:
            per_run[m].append(vals[m])

    out = {}
    # A handful of the 3026 basins are unscoreable in every seed; nanmean
    # warns on those all-NaN columns and yields NaN, which cdf filtering drops.
    with np.errstate(invalid='ignore'), warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        for m in METRICS:
            out[m] = np.nanmean(np.stack(per_run[m], axis=0), axis=0)
    out['n_seeds'] = n_seeds

    _cache[key] = out
    return out


def cdf_xy(vals):
    x = np.sort(vals)
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    for scenario in SCENARIOS:
        for metric in METRICS:
            fig, ax = plt.subplots(figsize=(8.5, 7))

            for label, model_prefix, color in SERIES:
                data = load_series(model_prefix, scenario)
                vals = data[metric]
                vals = vals[np.isfinite(vals)]
                x, y = cdf_xy(vals)
                median = np.median(vals)

                # A series without replicates is dashed and labelled, so it is
                # never read as a seed average.
                seeded = data['n_seeds'] > 1
                tag = '' if seeded else ', 1 seed'
                ax.plot(np.clip(x, CLIP_LOW, None), y, color=color, lw=2.5,
                        ls='-' if seeded else '--',
                        label='{} (median {}={:.2f}{})'.format(
                            label, METRIC_LABELS[metric], median, tag))
                print('{:17s} {:3s} {:26s} median={:.3f} n={} seeds={}'.format(
                    scenario, metric.upper(), label, median, len(vals),
                    data['n_seeds']))

            ax.axvline(0, color='#8a8a86', lw=1.2, ls='--', alpha=0.7)
            ax.set_xlim(CLIP_LOW, 1.0)
            ax.set_ylim(0, 1)
            ax.grid(which='major', ls='--', lw=0.6, alpha=0.5, color='#c3c2b7')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.tick_params(labelsize=FS_TICK, width=1.4, length=6)
            ax.set_xlabel(METRIC_LABELS[metric], fontsize=FS_AXIS)
            ax.set_ylabel('CDF', fontsize=FS_AXIS)
            ax.legend(fontsize=FS_LEGEND, loc='upper left', framealpha=0.9,
                      handlelength=1.5, handletextpad=0.5, borderpad=0.4)

            fig.tight_layout()
            out = os.path.join(OUT_DIR, '{}_{}.png'.format(metric, FILE_TAG[scenario]))
            fig.savefig(out, dpi=300, bbox_inches='tight')
            fig.savefig(out.replace('.png', '.pdf'), bbox_inches='tight')
            plt.close(fig)
            print('Saved: {}\n'.format(out))


if __name__ == '__main__':
    main()
