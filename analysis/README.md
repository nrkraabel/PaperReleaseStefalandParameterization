# Analysis

Turns finished multi-seed runs into the figures and tables in the paper. Two
steps: collect, then plot.

```bash
python collect_caravan3026_seeds.py        # $DMG_OUTPUT_ROOT -> ./Caravan3026/
python caravan3026_cdf_3seed_panels.py     # ./Caravan3026/   -> figures
```

`collect_extended_1980_1999.py` and `extended_1980_1999_cdf_3seed.py` are the
CAMELS-531 equivalents. The remaining scripts read the same collected trees:

| Script | Output |
|---|---|
| `caravan3026_cdf_3seed_panels.py` | Per-basin NSE and KGE CDFs, three seeds |
| `extended_1980_1999_cdf_3seed.py` | The same for CAMELS-531 |
| `caravan3026_ablation_degradation_heatmap.py` | Ablation degradation, by component |
| `extended_1980_1999_ablation_degradation_heatmap.py` | The same for CAMELS-531 |
| `caravan3026_mswep_vs_era5.py` | MSWEP against ERA5-Land precipitation |
| `caravan3026_supp_flowseg_table.py` | Supplementary table, by flow segment |

## Why the collection scripts copy arrays, not metrics

A finished run writes both `aggregated_predictions.npy` and a `metrics.json`
computed from it. Re-testing a run overwrites the arrays but can leave the JSON
describing the previous predictions, and nothing in the file says so. That
happened here: for the re-tested Embedding daily and monthly runs the stored
JSON matched the superseded predictions exactly and disagreed with the current
ones by up to about 100 NSE.

So the arrays are the source of truth, the collection step copies them, and the
plotting scripts recompute per-basin NSE and KGE. It costs roughly 3.5 GB and
about 40 seconds per figure. It also makes the collected tree usable for
hydrographs and flow-duration curves, which the JSON never supported.

Collection uses an explicit allowlist of filenames rather than a glob, so
unrelated files sitting in a results directory are never picked up. Re-running
is safe; files already in place are skipped.
