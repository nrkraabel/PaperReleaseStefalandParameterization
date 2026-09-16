"""
Supplementary table figure: KGE, percent bias, and flow-segment error
distributions for both Caravan3026 global protocols.

    Figure S#: KGE, percent bias, and flow-segment error distributions for
               both global protocols.

Renders one table per protocol plus a stacked two-panel version, and writes
the same numbers as CSV and LaTeX so the manuscript can pull them directly:

    suppS_flowseg_table_spatial_holdout.{png,pdf}
    suppS_flowseg_table_spatial_temporal.{png,pdf}
    suppS_flowseg_table_both_protocols.{png,pdf}   <- the figure to submit
    suppS_flowseg_stats.csv                        <- full distribution stats
    suppS_flowseg_table.tex                        <- booktabs table

Each cell is the across-basin distribution of a per-basin metric: median on
the first line, [Q1, Q3] on the second. Medians and quartiles, not means, for
the same reason the CDF panels use them -- FHV and FLV have heavy tails, and
a handful of near-zero-flow basins would otherwise set the column.

Metric definitions mirror dmg.core.calc.metrics.Metrics exactly (see
_pbias and the flow-segment block there), so the table agrees with anything
the training pipeline reports:

    PBIAS     100 * sum(pred - target) / sum(target)          all timesteps
    FLV       same, over the lowest 30% of the FDC   (+1e-4 offset, as dmg)
    PBIAS_mid same, over the 30-98% midsegment
    FHV       same, over the highest 2% of the FDC

The segments are taken from the *separately sorted* flow duration curves of
prediction and target, not from time-matched pairs -- these are FDC-segment
volume errors, so a model can score FHV = 0 while missing every peak's date.
Note PBIAS_mid is a midsegment *volume* bias; it is not Yilmaz et al. (2008)
%FMS, which is an FDC slope error. Labelled as such in the table.

Two guards beyond dmg, both reported in the figure footnote and the CSV:
basins whose target volume in a segment is zero give an undefined percent
bias (12 of 3026 have no flow at all in the test window) and are dropped from
that column rather than reported as a ~1e8 % outlier; and, as in every other
figure here, a basin needs >1 jointly-valid timestep to be scored at all.

Per-basin values are seed-averaged before the distribution is taken, matching
caravan3026_cdf_3seed_panels.py. Arrays are cached to an .npz next to this
file so re-styling the table does not re-read ~4 GB of .npy.

Usage:
    python caravan3026_supp_flowseg_table.py             # build/restyle
    python caravan3026_supp_flowseg_table.py --refresh   # re-read the .npy
    python caravan3026_supp_flowseg_table.py --verify    # vs a dmg-style loop
"""
import os
import sys
import textwrap
import warnings

import numpy as np
import matplotlib.pyplot as plt

from caravan3026_cdf_3seed_panels import (
    ROOT, SCENARIOS, SERIES, FILE_TAG,
    compute_basin_nse_kge, find_run_dirs, check_targets_match,
)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, 'Caravan3026_supp_tables')
CACHE = os.path.join(HERE, 'caravan3026_flowseg_cache.npz')

SCENARIO_TITLE = {
    'pure_spatial': 'Protocol 1 - spatial holdout (PUB)',
    'spatial_temporal': 'Protocol 2 - spatial + temporal holdout',
}

# FDC segment cut points, as fractions of a basin's valid record. dmg rounds
# these to integer indices per basin; so does the vectorised code below.
LOW_FRAC = 0.30
HIGH_FRAC = 0.98
FLV_OFFSET = 1e-4  # dmg adds this to the low-segment denominator only

# (key, header, 'high' = larger is better | 'zero' = closer to zero is better)
COLUMNS = [
    ('kge',       'KGE',                                  'high'),
    ('pbias',     'PBIAS (%)\nall flows',                 'zero'),
    ('flv',       'FLV (%)\nlowest 30% of FDC',           'zero'),
    ('pbias_mid', 'PBIAS$_{\\mathrm{mid}}$ (%)\n30-98% of FDC', 'zero'),
    ('fhv',       'FHV (%)\nhighest 2% of FDC',           'zero'),
]
METRICS = ['nse', 'kge', 'rmse', 'pbias', 'flv', 'pbias_mid', 'fhv']

FS_TICK = 15
FS_CELL = 14
FS_SUB = 11
FS_TITLE = 18

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'Nimbus Roman',
                              'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'stix'


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

def compute_basin_flow_metrics(pred, target):
    """Per-basin PBIAS and FDC-segment volume errors, in percent.

    Vectorised over basins. dmg loops, sorting each basin's masked series and
    slicing it at round(0.3n) / round(0.98n); the same thing is done here with
    one sort plus a cumulative sum, because 42 runs x 3026 basins of Python
    loop is minutes of wall clock for numbers that a cumsum gives in seconds.
    NaNs sort to the end of each column, so a basin's valid values occupy rows
    [0, n) and the cut indices are just round(frac * n). --verify checks this
    against a literal transcription of the dmg loop.
    """
    if target.ndim == 3:
        target = np.squeeze(target, axis=-1)
    if pred.ndim == 3:
        pred = np.squeeze(pred, axis=-1)

    mask = np.isfinite(pred) & np.isfinite(target)
    n = mask.sum(axis=0)

    p = np.where(mask, pred, np.nan).astype(np.float64)
    t = np.where(mask, target, np.nan).astype(np.float64)
    p_sorted = np.sort(p, axis=0)
    t_sorted = np.sort(t, axis=0)

    # cum[k, b] = sum of basin b's k smallest valid flows
    def cumulative(a):
        c = np.cumsum(np.nan_to_num(a, nan=0.0), axis=0)
        return np.concatenate([np.zeros((1, a.shape[1])), c], axis=0)

    cp, ct = cumulative(p_sorted), cumulative(t_sorted)

    i_low = np.round(LOW_FRAC * n).astype(int)
    i_high = np.round(HIGH_FRAC * n).astype(int)

    def at(cum, idx):
        return np.take_along_axis(cum, idx[None, :], axis=0)[0]

    segs = {
        'pbias':     (at(cp, n),      at(ct, n),      0.0),
        'flv':       (at(cp, i_low),  at(ct, i_low),  FLV_OFFSET),
        'pbias_mid': (at(cp, i_high) - at(cp, i_low),
                      at(ct, i_high) - at(ct, i_low), 0.0),
        'fhv':       (at(cp, n) - at(cp, i_high),
                      at(ct, n) - at(ct, i_high),     0.0),
    }

    # RMSE, mirroring dmg's _rmse: sqrt(nanmean((pred - target)^2)) over the
    # time axis. Same joint mask as everything else here, so a basin's RMSE is
    # taken over exactly the timesteps its NSE and bias metrics are.
    with np.errstate(invalid='ignore'):
        sq = np.where(mask, (p - t) ** 2, 0.0)
        rmse = np.sqrt(sq.sum(axis=0) / np.maximum(n, 1))
    rmse[n < 1] = np.nan

    out = {'rmse': rmse}
    for key, (sum_p, sum_t, offset) in segs.items():
        with np.errstate(divide='ignore', invalid='ignore'):
            v = (sum_p - sum_t) / (sum_t + offset) * 100.0
        # A segment the gauge reports as carrying no water has no percent
        # bias; with dmg's 1e-4 offset FLV would instead read ~1e8 %.
        v[sum_t <= 0] = np.nan
        v[n <= 1] = np.nan
        out[key] = v
    return out


def _dmg_reference(pred, target, basins):
    """Literal transcription of the dmg per-basin loop, for --verify."""
    if target.ndim == 3:
        target = np.squeeze(target, axis=-1)
    if pred.ndim == 3:
        pred = np.squeeze(pred, axis=-1)

    def pbias(a, b, offset=0.0):
        return np.sum(a - b) / (np.sum(b) + offset) * 100

    out = {k: np.full(len(basins), np.nan)
           for k in ('pbias', 'flv', 'pbias_mid', 'fhv')}
    for j, i in enumerate(basins):
        _pred, _target = pred[:, i], target[:, i]
        idx = np.where(np.logical_and(~np.isnan(_pred), ~np.isnan(_target)))[0]
        if idx.shape[0] == 0:
            continue
        p, t = _pred[idx].astype(np.float64), _target[idx].astype(np.float64)
        ps, ts = np.sort(p), np.sort(t)
        i_low = round(LOW_FRAC * ps.shape[0])
        i_high = round(HIGH_FRAC * ps.shape[0])
        with np.errstate(divide='ignore', invalid='ignore'):
            out['flv'][j] = pbias(ps[:i_low], ts[:i_low], offset=FLV_OFFSET)
            out['fhv'][j] = pbias(ps[i_high:], ts[i_high:])
            out['pbias'][j] = pbias(p, t)
            out['pbias_mid'][j] = pbias(ps[i_low:i_high], ts[i_low:i_high])
    return out


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------

def build_cache():
    data = {}
    for scenario in SCENARIOS:
        for _, prefix, _ in SERIES:
            run_dirs, n_seeds = find_run_dirs(prefix, scenario)
            per_seed = {m: [] for m in METRICS}
            for d in run_dirs:
                pred = np.load(os.path.join(d, 'aggregated_predictions.npy'))
                target = np.load(os.path.join(d, 'aggregated_targets.npy'))
                check_targets_match(scenario, target, os.path.relpath(d, ROOT))
                vals = compute_basin_nse_kge(pred, target)
                vals.update(compute_basin_flow_metrics(pred, target))
                for m in METRICS:
                    per_seed[m].append(vals[m])
            for m in METRICS:
                data['{}|{}|{}'.format(prefix, scenario, m)] = \
                    np.stack(per_seed[m], axis=0)
            print('cached {:20s} {:17s} seeds={} basins={}'.format(
                prefix, scenario, n_seeds, per_seed['kge'][0].size))
    np.savez_compressed(CACHE, **data)
    print('wrote {}'.format(CACHE))
    return data


def load_cache(refresh=False):
    if refresh or not os.path.exists(CACHE):
        return build_cache()
    with np.load(CACHE) as z:
        return {k: z[k] for k in z.files}


def seed_mean(data, prefix, scenario, metric):
    arr = data['{}|{}|{}'.format(prefix, scenario, metric)]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        return np.nanmean(arr, axis=0)


def n_seeds(data, prefix, scenario):
    return data['{}|{}|kge'.format(prefix, scenario)].shape[0]


def label_for(label, data, prefix, scenario):
    """Series label, marked when it is a single run rather than a seed mean."""
    return label if n_seeds(data, prefix, scenario) > 1 else label + '*'


def dist(v):
    """Distribution summary of one column, ignoring unscoreable basins."""
    v = v[np.isfinite(v)]
    q = np.percentile(v, [10, 25, 50, 75, 90])
    return dict(p10=q[0], q1=q[1], med=q[2], q3=q[3], p90=q[4],
                iqr=q[3] - q[1], mean=v.mean(), n=int(v.size))


def fmt_val(key, v):
    """KGE to 3 dp; bias columns signed to 1 dp.

    A bias median that rounds to zero is printed '0.0', not '-0.0': the sign
    of a value the table has just rounded away is noise, and a reader scanning
    the column for over- vs under-prediction should not be handed one.
    """
    if key == 'kge':
        return '{:.3f}'.format(v)
    return '0.0' if abs(round(v, 1)) == 0.0 else '{:+.1f}'.format(v)


def fmt_range(key, lo, hi):
    return '[{}, {}]'.format(fmt_val(key, lo), fmt_val(key, hi))


def n_dropped(data, prefix, scenario, metric):
    """Basins with no finite value for this metric, after seed averaging."""
    v = seed_mean(data, prefix, scenario, metric)
    return int((~np.isfinite(v)).sum())


# --------------------------------------------------------------------------
# table rendering
# --------------------------------------------------------------------------

def table_cells(data, scenario):
    """(text, subtext, rank) arrays for one protocol's table."""
    n_rows, n_cols = len(SERIES), len(COLUMNS)
    text = np.empty((n_rows, n_cols), dtype=object)
    sub = np.empty((n_rows, n_cols), dtype=object)
    score = np.zeros((n_rows, n_cols))

    for i, (_, prefix, _) in enumerate(SERIES):
        for j, (key, _, better) in enumerate(COLUMNS):
            s = dist(seed_mean(data, prefix, scenario, key))
            text[i, j] = fmt_val(key, s['med'])
            sub[i, j] = fmt_range(key, s['q1'], s['q3'])
            # Bias columns are ranked on |median|: a model 5% wet and one 5%
            # dry are equally biased, and shading them differently would
            # invent a preference for over-prediction.
            score[i, j] = s['med'] if better == 'high' else -abs(s['med'])

    rank = np.argsort(np.argsort(score, axis=0), axis=0) / (n_rows - 1)
    return text, sub, rank


def draw_table(ax, data, scenario, show_header=True):
    text, sub, rank = table_cells(data, scenario)
    n_rows, n_cols = text.shape

    # The rank is stretched over only the pale half of YlGn. Letting the best
    # cell take the colormap's dark green puts near-black text on near-black
    # background, and the [Q1, Q3] line in particular stops being readable;
    # ranking only needs the eye to order the column, not to saturate it.
    ax.imshow(rank, cmap='YlGn', vmin=-0.12, vmax=1.75, aspect='auto')
    for i in range(n_rows):
        for j in range(n_cols):
            ax.text(j, i - 0.16, text[i, j], ha='center', va='center',
                    fontsize=FS_CELL, color='#1a1a1a')
            ax.text(j, i + 0.24, sub[i, j], ha='center', va='center',
                    fontsize=FS_SUB, color='#4a4a4a')

    ax.set_xticks(range(n_cols))
    if show_header:
        ax.set_xticklabels([c[1] for c in COLUMNS], fontsize=FS_TICK - 1)
        ax.xaxis.set_label_position('top')
        ax.xaxis.tick_top()
    else:
        ax.set_xticklabels([])
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(
        [label_for(l, data, p, scenario) for l, p, _ in SERIES],
        fontsize=FS_TICK - 1)
    ax.set_xticks(np.arange(-0.5, n_cols), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows), minor=True)
    ax.grid(which='minor', color='white', lw=2)
    ax.tick_params(which='both', length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    return text


def wrap_note(text, width=155):
    """Hard-wrap a caption to `width` characters.

    matplotlib's wrap=True measures against the figure's *current* width, but
    bbox_inches='tight' then grows the bbox to fit whatever was laid out, so a
    long note renders as one line and doubles the figure width. Wrapping the
    string ourselves keeps the panel at its declared size. No token here
    contains a space inside mathtext, so $...$ spans survive the wrap.
    """
    return '\n'.join(textwrap.wrap(' '.join(text.split()), width=width))


def footnote(data, scenario):
    n = dist(seed_mean(data, SERIES[0][1], scenario, 'kge'))['n']
    worst_flv = max(n_dropped(data, p, scenario, 'flv') for _, p, _ in SERIES)
    single = [l for l, p, _ in SERIES if n_seeds(data, p, scenario) == 1]
    note = ('Cells give the across-basin median with [Q1, Q3] beneath; '
            'n = {} scoreable basins. Shading ranks each column '
            '(KGE: higher is better; bias columns: |median| closer to 0). '
            'FDC segments are taken from separately sorted flow duration '
            'curves, so they measure segment volume error, not timing. '
            'PBIAS$_{{\\mathrm{{mid}}}}$ is a midsegment volume bias, not '
            'Yilmaz %FMS slope. Up to {} further basins are undefined for '
            'FLV (zero observed low-flow volume) and excluded from that '
            'column.'.format(n, worst_flv))
    if single:
        note += ' *single seed, not a 3-seed mean ({}).'.format(
            ', '.join(single))
    return note


def figure_per_protocol(data):
    for scenario in SCENARIOS:
        fig, ax = plt.subplots(figsize=(13.5, 5.4))
        draw_table(ax, data, scenario)
        ax.set_title(SCENARIO_TITLE[scenario], fontsize=FS_TITLE, pad=46)
        fig.text(0.5, -0.04, wrap_note(footnote(data, scenario)), ha='center',
                 va='top', fontsize=FS_SUB - 1, color='#4a4a4a',
                 linespacing=1.5)
        save(fig, 'suppS_flowseg_table_{}'.format(FILE_TAG[scenario]))


def figure_both_protocols(data):
    """The submission figure: both protocols in one column of panels."""
    fig, axes = plt.subplots(2, 1, figsize=(13.5, 11.6))
    for ax, scenario in zip(axes, SCENARIOS):
        draw_table(ax, data, scenario, show_header=True)
        ax.set_title(SCENARIO_TITLE[scenario], fontsize=FS_TITLE, pad=46)

    n = dist(seed_mean(data, SERIES[0][1], SCENARIOS[0], 'kge'))['n']
    worst_flv = max(n_dropped(data, p, s, 'flv')
                    for _, p, _ in SERIES for s in SCENARIOS)
    note = ('Each cell is the across-basin distribution of a per-basin metric: '
            'median on the first line, [Q1, Q3] beneath; values are averaged '
            'over 3 seeds per basin before the distribution is taken '
            '(n = {} scoreable basins). Shading ranks each column within a '
            'protocol (KGE: higher is better; bias columns: |median| closer '
            'to zero is better). FDC segments come from separately sorted '
            'flow duration curves and so measure segment volume error, not '
            'timing; PBIAS$_{{\\mathrm{{mid}}}}$ is a midsegment volume bias '
            'rather than Yilmaz et al. (2008) %FMS slope. FLV is undefined '
            'for the {} intermittent basins whose lowest 30% of observed flow '
            'is entirely zero; those are excluded from that column '
            'only.'.format(n, worst_flv))
    fig.text(0.5, 0.028, wrap_note(note), ha='center', va='top',
             fontsize=FS_SUB, color='#4a4a4a', linespacing=1.5)
    save(fig, 'suppS_flowseg_table_both_protocols', rect=(0, 0.055, 1, 1))


# --------------------------------------------------------------------------
# machine-readable outputs
# --------------------------------------------------------------------------

def write_csv(data):
    path = os.path.join(OUT_DIR, 'suppS_flowseg_stats.csv')
    with open(path, 'w') as fh:
        fh.write('protocol,series,n_seeds,metric,n_basins,n_undefined,'
                 'median,q1,q3,iqr,p10,p90,mean\n')
        for scenario in SCENARIOS:
            for label, prefix, _ in SERIES:
                for key, _, _ in COLUMNS:
                    s = dist(seed_mean(data, prefix, scenario, key))
                    fh.write('{},{},{},{},{},{},{:.6f},{:.6f},{:.6f},'
                             '{:.6f},{:.6f},{:.6f},{:.6f}\n'.format(
                                 FILE_TAG[scenario], label,
                                 n_seeds(data, prefix, scenario), key,
                                 s['n'], n_dropped(data, prefix, scenario, key),
                                 s['med'], s['q1'], s['q3'], s['iqr'],
                                 s['p10'], s['p90'], s['mean']))
    print('saved {}'.format(path))


def write_latex(data):
    """booktabs table, median with [Q1, Q3] in a smaller second line."""
    heads = ['KGE', 'PBIAS (\\%)', 'FLV (\\%)',
             'PBIAS$_{\\mathrm{mid}}$ (\\%)', 'FHV (\\%)']
    lines = [
        '% Figure/Table S#: KGE, percent bias, and flow-segment error',
        '% distributions for both global protocols.',
        '% Generated by caravan3026_supp_flowseg_table.py -- do not hand-edit.',
        '\\begin{table}[t]', '\\centering', '\\small',
        '\\begin{tabular}{l' + 'c' * len(COLUMNS) + '}', '\\toprule',
        'Model & ' + ' & '.join(heads) + ' \\\\',
        ' & & all flows & lowest 30\\% & 30--98\\% & highest 2\\% \\\\',
        '\\midrule',
    ]
    for scenario in SCENARIOS:
        # Keep the .tex ASCII: the figure titles carry a real en dash, which
        # a pdflatex source file without inputenc would choke on.
        title = (SCENARIO_TITLE[scenario].replace('-', '--')
                 .replace('%', '\\%').replace('_', '\\_'))
        lines.append('\\multicolumn{{{}}}{{l}}{{\\textit{{{}}}}} \\\\'.format(
            len(COLUMNS) + 1, title))
        for label, prefix, _ in SERIES:
            cells = []
            for key, _, _ in COLUMNS:
                s = dist(seed_mean(data, prefix, scenario, key))
                cells.append('\\shortstack{{{} \\\\ \\scriptsize {}}}'.format(
                    fmt_val(key, s['med']),
                    fmt_range(key, s['q1'], s['q3'])))
            tex_label = label.replace('δ', '$\\delta$').replace('_', '\\_')
            lines.append('{} & {} \\\\'.format(tex_label, ' & '.join(cells)))
        if scenario != SCENARIOS[-1]:
            lines.append('\\midrule')
    lines += [
        '\\bottomrule', '\\end{tabular}',
        '\\caption{Across-basin distributions (median with [Q1, Q3]) of KGE, '
        'percent bias, and flow-duration-curve segment volume errors for both '
        'global protocols.}',
        '\\label{tab:supp-flowseg}', '\\end{table}',
    ]
    path = os.path.join(OUT_DIR, 'suppS_flowseg_table.tex')
    with open(path, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    print('saved {}'.format(path))


# --------------------------------------------------------------------------

def save(fig, stem, rect=None):
    fig.tight_layout(rect=rect) if rect else fig.tight_layout()
    png = os.path.join(OUT_DIR, stem + '.png')
    fig.savefig(png, dpi=200, bbox_inches='tight')
    fig.savefig(png.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close(fig)
    print('saved {}'.format(png))


def print_summary(data):
    for scenario in SCENARIOS:
        print('\n=== {} ==='.format(SCENARIO_TITLE[scenario]))
        print('{:26s} {:>7s} {:>8s} {:>8s} {:>10s} {:>8s} {:>6s}'.format(
            'series', 'KGE', 'PBIAS', 'FLV', 'PBIASmid', 'FHV', 'seeds'))
        for label, prefix, _ in SERIES:
            vals = [dist(seed_mean(data, prefix, scenario, k))['med']
                    for k, _, _ in COLUMNS]
            print('{:26s} {:7.3f} {:+8.1f} {:+8.1f} {:+10.1f} {:+8.1f} '
                  '{:6d}'.format(label, vals[0], vals[1], vals[2], vals[3],
                                 vals[4], n_seeds(data, prefix, scenario)))


def verify():
    """Vectorised flow-segment metrics vs a literal dmg-style per-basin loop."""
    rng = np.random.default_rng(0)
    for scenario in SCENARIOS[:1]:
        for _, prefix, _ in SERIES[:2]:
            run_dirs, _ = find_run_dirs(prefix, scenario)
            d = run_dirs[0]
            pred = np.load(os.path.join(d, 'aggregated_predictions.npy'))
            target = np.load(os.path.join(d, 'aggregated_targets.npy'))
            fast = compute_basin_flow_metrics(pred, target)
            n_basins = fast['pbias'].size
            basins = rng.choice(n_basins, size=250, replace=False)
            ref = _dmg_reference(pred, target, basins)
            for key in ref:
                a = fast[key][basins]
                b = ref[key]
                both = np.isfinite(a) & np.isfinite(b)
                # dmg reports the ~1e8 % FLV that this module drops; compare
                # only where both are defined, and count the difference.
                diff = np.abs(a[both] - b[both])
                rel = diff / np.maximum(np.abs(b[both]), 1e-9)
                print('{:20s} {:10s} n={:3d} max|diff|={:.3e} '
                      'max rel={:.3e} dropped={}'.format(
                          prefix, key, both.sum(), diff.max() if diff.size else 0,
                          rel.max() if rel.size else 0,
                          int((~np.isfinite(a) & np.isfinite(b)).sum())))


def main():
    if '--verify' in sys.argv:
        verify()
        return
    os.makedirs(OUT_DIR, exist_ok=True)
    data = load_cache(refresh='--refresh' in sys.argv)
    print_summary(data)
    figure_per_protocol(data)
    figure_both_protocols(data)
    write_csv(data)
    write_latex(data)


if __name__ == '__main__':
    main()
