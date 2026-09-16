"""
CDF plots of per-basin streamflow skill (NSE, KGE) for the
Extended1980_1999_Camels531 experiments, PUB vs PUR, each seed-averaged
across the 3 available seeds (seed111111/222222/333333).

Per-basin NSE/KGE come straight from each run's spatial_aggregated_<split>/
metrics.json (same values dmg.core.calc.metrics produced during testing).
For a given experiment/split, the 3 seeds share identical basin ordering and
targets (only model init differs), so the seed average is taken per basin
before building the CDF -- this mirrors averaging seeds the same way the
scalar summary table does, just at per-basin granularity.

Requires: HBVPaperResults/Extended1980_1999_Camels531/<model>/<split>/seed*/
spatial_aggregated_<split>/metrics.json, produced by collect_extended_1980_1999.py.
"""
import json
import numpy as np
import matplotlib.pyplot as plt

BASE = '${oc.env:DMG_DATA_ROOT}/HBVPaperResults/Extended1980_1999_Camels531'
SEEDS = ['seed111111', 'seed222222', 'seed333333']
SPLITS = ['PUB', 'PUR']

# label -> (model dir name, color, linestyle)
SERIES = {
    'LSTM (no HBV, no embedding)':               ('LSTM', '#8a8a86', '--'),
    'LSTM+HBV (no embedding)':                    ('LSTMHBV', '#52514e', '-'),
    'Embedding adapter (StefaLandGrid64)':         ('EmbeddingStefaLandGrid64', '#2a78d6', '-'),
    'Condensed embedding (MFFormer256, daily)':    ('Camels_531_condensed_embeddings_daily', '#eb6834', '-'),
    'Condensed embedding (MFFormer256, monthly)':  ('Camels_531_condensed_embeddings_monthly', '#1baf7a', '-'),
}

METRICS = ['nse', 'kge']
METRIC_LABELS = {'nse': 'NSE', 'kge': 'KGE'}
CLIP_LOW = -1.0  # curves clipped at -1 for readability; medians computed unclipped


def load_seed_averaged_metric(model_dir, split, metric):
    """Per-basin metric averaged across the 3 seeds (NaN-safe)."""
    per_seed = []
    for seed in SEEDS:
        f = f'{BASE}/{model_dir}/{split}/{seed}/spatial_aggregated_{split}/metrics.json'
        d = json.loads(json.load(open(f)))
        per_seed.append(np.array(d[metric], dtype=np.float64))
    stacked = np.stack(per_seed, axis=0)  # (3, n_basins)
    with np.errstate(invalid='ignore'):
        avg = np.nanmean(stacked, axis=0)
    return avg[np.isfinite(avg)]


def cdf_xy(vals):
    x = np.sort(vals)
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y


fig, axes = plt.subplots(len(METRICS), len(SPLITS), figsize=(11, 8), sharey=True)

for row, metric in enumerate(METRICS):
    for col, split in enumerate(SPLITS):
        ax = axes[row, col]
        for label, (model_dir, color, ls) in SERIES.items():
            vals = load_seed_averaged_metric(model_dir, split, metric)
            x, y = cdf_xy(vals)
            median = np.median(vals)
            ax.plot(np.clip(x, CLIP_LOW, None), y, color=color, lw=2, ls=ls,
                    label=f'{label} (median {METRIC_LABELS[metric]}={median:.2f})')
            ax.axvline(median, color=color, lw=1, ls=':', alpha=0.6)

        ax.axvline(0, color='#8a8a86', lw=1, ls='--', alpha=0.7)
        ax.set_xlim(CLIP_LOW, 1.0)
        ax.set_ylim(0, 1)
        ax.grid(which='major', ls='--', lw=0.5, alpha=0.5, color='#c3c2b7')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=9)

        if row == 0:
            ax.set_title(f'{split} (3-seed average, n={531})', fontsize=10.5, fontweight='bold')
        if col == 0:
            ax.set_ylabel(f'{METRIC_LABELS[metric]}\nCumulative fraction of basins', fontsize=10)
        if row == len(METRICS) - 1:
            ax.set_xlabel(METRIC_LABELS[metric], fontsize=10)

        ax.legend(fontsize=7.5, loc='upper left', framealpha=0.9,
                   handlelength=1.5, handletextpad=0.5, borderpad=0.4)

fig.suptitle('Extended 1980-1999 training: PUB vs PUR skill, averaged over 3 seeds',
             fontsize=12, fontweight='bold', y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.96])

out = '${oc.env:DMG_DATA_ROOT}/HBVPaperResults/extended_1980_1999_cdf_3seed.png'
fig.savefig(out, dpi=300, bbox_inches='tight')
print(f'Saved: {out}')
