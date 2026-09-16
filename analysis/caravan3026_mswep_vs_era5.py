#!/usr/bin/env python
"""MSWEP vs ERA5-Land precipitation on Caravan3026 PUB, 3 seeds, both splits.

Answers the question the runs were launched for: does swapping Caravan's
ERA5-Land `total_precipitation_sum` for a catchment-averaged MSWEP product
change PUB skill, and does it change it more for the physics-constrained
models (delta-HBV) than for the pure LSTM?

Every arm is the same architecture, window, PUB fold set, seed set and
optimizer as its twin -- only the precipitation variable differs -- so the
difference is attributable to the forcing product.

Metrics come from `compute_basin_metrics` / `summarise` in
`caravan3026_ablation_degradation_heatmap.py` (imported, not re-implemented)
so the numbers are defined identically to the ablation figure: joint NaN
masking per basin, population moments, and every statistic restricted to the
finite subset (a constant target gives NSE = -inf, a zero-sum target gives
|PBIAS| = inf -- those are +-inf, not NaN, and would poison a nanmedian).

Per-basin seed averaging before summarising is valid here for the same reason
it is in the CDF scripts: every run shares a byte-identical
`aggregated_targets.npy`, so basin ordering matches. That is checked, not
assumed -- a mismatch is what the Embedding/annual CAMELS-531 mix-up looked
like.

Because the arms are paired basin-for-basin, the table also reports the
paired per-basin delta: its median, and the fraction of basins that improve.
That is a much stronger read than comparing two independent medians, and it
is what distinguishes "MSWEP is better everywhere by a little" from "MSWEP is
much better in some regions and worse in others".

Usage:
    python caravan3026_mswep_vs_era5.py            # table + CSV
    python caravan3026_mswep_vs_era5.py --plot     # + CDF panels
"""

import argparse
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from caravan3026_ablation_degradation_heatmap import (  # noqa: E402
    compute_basin_metrics,
    summarise,
)

SCRATCH = '${oc.env:DMG_OUTPUT_ROOT}/Caravan3026'
ARCHIVE = '${oc.env:DMG_DATA_ROOT}/HBVPaperResults/Caravan3026'
SEEDS = ['seed111111', 'seed222222', 'seed333333']

SPLITS = [
    ('pure spatial (train/test 1998-2008)', 'pure_spatial_1998-2008'),
    ('spatial-temporal (train 1987-1997, test 1998-2008)',
     'spatial_temporal_train1987-1997_test1998-2008'),
]

# label, ERA5 prefix, MSWEP prefix
MODELS = [
    ('daily EMB + dHBV', 'Embedding/daily', 'Embedding_MSWEP/daily'),
    ('dHBV + LSTM',      'LSTMHBV',         'LSTMHBV_MSWEP'),
    ('LSTM (no physics)', 'LSTMNoHBV',      'LSTMNoHBV_MSWEP'),
]

METRIC_KEYS = [
    ('nse', 'NSE', +1),
    ('kge', 'KGE', +1),
    ('corr', 'Corr', +1),
    ('rmse', 'RMSE', -1),
    ('pbias_abs', '|PBIAS|', -1),
]

OUT_DIR = '${oc.env:DMG_DATA_ROOT}/HBVPaperResults/Caravan3026_MSWEP_vs_ERA5'


def run_dirs(prefix, split_dir):
    """Seed dirs for one arm, preferring the archive, else scratch."""
    for root in (ARCHIVE, SCRATCH):
        base = os.path.join(root, prefix, split_dir)
        dirs = [os.path.join(base, s, 'spatial_aggregated_PUB') for s in SEEDS]
        dirs = [d for d in dirs
                if os.path.exists(os.path.join(d, 'aggregated_predictions.npy'))
                and os.path.exists(os.path.join(d, 'aggregated_targets.npy'))]
        if dirs:
            return dirs
    return []


def load_arm(prefix, split_dir, target_ref):
    """Seed-averaged per-basin metrics for one arm, plus a target-identity check."""
    dirs = run_dirs(prefix, split_dir)
    if not dirs:
        return None, [], target_ref, 'MISSING'

    per_run = {k: [] for k, _, _ in METRIC_KEYS}
    note = ''
    for d in dirs:
        pred = np.load(os.path.join(d, 'aggregated_predictions.npy'))
        tgt = np.load(os.path.join(d, 'aggregated_targets.npy'))
        t2 = np.squeeze(tgt, -1) if tgt.ndim == 3 else tgt
        if target_ref is None:
            target_ref = t2
        elif not (t2.shape == target_ref.shape and np.array_equal(
                np.nan_to_num(t2, nan=-9e9),
                np.nan_to_num(target_ref, nan=-9e9))):
            note = 'TARGET MISMATCH'
        vals = compute_basin_metrics(pred, tgt)
        for k, _, _ in METRIC_KEYS:
            per_run[k].append(vals[k])

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)  # all-NaN basins
        out = {k: np.nanmean(np.vstack(per_run[k]), axis=0) for k in per_run}
    return out, [os.path.basename(os.path.dirname(d)) for d in dirs], target_ref, note


def paired(era, msw, sign):
    """Paired per-basin delta over basins finite in BOTH arms.

    `sign` orients the delta so positive always means "MSWEP is better",
    including for RMSE and |PBIAS| where lower is better.
    """
    ok = np.isfinite(era) & np.isfinite(msw)
    if ok.sum() == 0:
        return np.nan, np.nan, 0
    d = sign * (msw[ok] - era[ok])
    return float(np.median(d)), float((d > 0).mean() * 100.0), int(ok.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plot', action='store_true', help='also write CDF panels')
    ap.add_argument('--out-dir', default=OUT_DIR)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rows = []
    store = {}

    for split_label, split_dir in SPLITS:
        print('\n' + '=' * 100)
        print(split_label)
        print('=' * 100)
        target_ref = None

        for mlabel, era_prefix, msw_prefix in MODELS:
            era, era_seeds, target_ref, n1 = load_arm(era_prefix, split_dir, target_ref)
            msw, msw_seeds, target_ref, n2 = load_arm(msw_prefix, split_dir, target_ref)
            if era is None or msw is None:
                print('\n  {:<20} SKIPPED (ERA5 {}, MSWEP {})'.format(
                    mlabel, n1 or 'ok', n2 or 'ok'))
                continue
            flag = ' [{}]'.format(n1 or n2) if (n1 or n2) else ''
            print('\n  {:<20} ERA5 {} seeds | MSWEP {} seeds{}'.format(
                mlabel, len(era_seeds), len(msw_seeds), flag))
            print('  {:<9} {:>10} {:>10} {:>10}   {:>12} {:>10}'.format(
                '', 'ERA5 med', 'MSWEP med', 'diff', 'paired med', '% basins'))
            print('  {:<9} {:>10} {:>10} {:>10}   {:>12} {:>10}'.format(
                'metric', '', '', '', 'delta', 'improved'))
            print('  ' + '-' * 68)

            store[(split_dir, mlabel)] = (era, msw)

            for key, name, sign in METRIC_KEYS:
                e_med, _, e_n = summarise(era[key])
                m_med, _, m_n = summarise(msw[key])
                pmed, pfrac, pn = paired(era[key], msw[key], sign)
                better = sign * (m_med - e_med) > 0
                print('  {:<9} {:>10.4f} {:>10.4f} {:>10} {:>12.4f} {:>9.1f}%'
                      .format(name, e_med, m_med,
                              '{}{:.4f}'.format('+' if m_med >= e_med else '',
                                                m_med - e_med),
                              pmed, pfrac)
                      + ('  <-- MSWEP better' if better else ''))
                rows.append([split_dir, mlabel, name, e_med, m_med,
                             m_med - e_med, pmed, pfrac, e_n, m_n, pn])

    csv_path = os.path.join(args.out_dir, 'mswep_vs_era5_summary.csv')
    with open(csv_path, 'w') as fh:
        fh.write('split,model,metric,era5_median,mswep_median,median_diff,'
                 'paired_median_delta_mswep_better_positive,'
                 'pct_basins_improved,n_era5,n_mswep,n_paired\n')
        for r in rows:
            fh.write(','.join(
                ['"{}"'.format(x) if isinstance(x, str) else
                 ('' if not np.isfinite(x) else '{:.6f}'.format(x))
                 for x in r]) + '\n')
    print('\nwrote', csv_path)

    npz_path = os.path.join(args.out_dir, 'mswep_vs_era5_perbasin.npz')
    flat = {}
    for (sd, ml), (era, msw) in store.items():
        tag = '{}|{}'.format(sd, ml).replace(' ', '_')
        for key, _, _ in METRIC_KEYS:
            flat['ERA5|' + tag + '|' + key] = era[key]
            flat['MSWEP|' + tag + '|' + key] = msw[key]
    np.savez_compressed(npz_path, **flat)
    print('wrote', npz_path)

    if args.plot:
        make_plot(store, args.out_dir)


def make_plot(store, out_dir):
    """NSE/KGE CDF panels, MSWEP vs ERA5, one column per split."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    colors = {'daily EMB + dHBV': '#4C72B0',
              'dHBV + LSTM': '#DD8452',
              'LSTM (no physics)': '#55A868'}

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), sharey=True)
    for col, (split_label, split_dir) in enumerate(SPLITS):
        for row, (key, name) in enumerate([('nse', 'NSE'), ('kge', 'KGE')]):
            ax = axes[row, col]
            for mlabel, _, _ in MODELS:
                pair = store.get((split_dir, mlabel))
                if pair is None:
                    continue
                era, msw = pair
                for arr, ls, tag in ((era[key], '--', 'ERA5'),
                                     (msw[key], '-', 'MSWEP')):
                    v = np.sort(arr[np.isfinite(arr)])
                    ax.plot(v, np.arange(1, v.size + 1) / v.size,
                            ls=ls, lw=1.6, color=colors[mlabel],
                            label='{} {}'.format(mlabel, tag))
            ax.set_xlim(-0.5, 1.0)
            ax.grid(alpha=0.3, lw=0.5)
            ax.set_xlabel(name)
            if col == 0:
                ax.set_ylabel('cumulative fraction of basins')
            if row == 0:
                ax.set_title(split_label, fontsize=10)
    axes[0, 0].legend(fontsize=7, loc='upper left', framealpha=0.9)
    fig.suptitle('Caravan3026 PUB: MSWEP (solid) vs ERA5-Land (dashed) '
                 'precipitation, 3-seed mean per basin', fontsize=11)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        p = os.path.join(out_dir, 'mswep_vs_era5_cdf.' + ext)
        fig.savefig(p, dpi=200, bbox_inches='tight')
        print('wrote', p)


if __name__ == '__main__':
    main()
