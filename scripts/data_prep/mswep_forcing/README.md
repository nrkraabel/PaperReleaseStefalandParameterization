# MSWEP precipitation forcing for the CAMELS-only Caravan file

Adds `total_precipitation_sum_MSWEP` (station_ids, time) alongside Caravan's
ERA5-Land `total_precipitation_sum`, so the HBV/embedding runs can swap the
precipitation source without touching anything else.

Target file:
`caravan_zenodo/OfficailCaravan_camels_only_singlefile_direct_dropHighNaNTrue_stationids.nc`
(3026 basins across camels / camelsaus / camelsbr / camelscl / camelsgb).

MSWEP source: `$DMG_DATA_ROOT/Daily/P_MSWEP` -- global 0.1°, one file per day,
`YYYYDDD.nc`, `precipitation` in mm/d, 1979-01-01 to 2025-06-29, no gaps.

## Read this first: the file's `time` coordinate is wrong

`CombineCamels.py:164-165` builds the time axis from the *length* of each source
file rather than reading its `date` variable:

```python
n_days = len(ds.dimensions["date"])
dates  = make_dates_from_dim(n_days)   # 1950-01-01 + 0..n_days-1
```

The data values are copied positionally and are bit-identical to the source
(verified: r = 1.000000, max abs diff = 0 against the per-basin Caravan files).
Only the labels are wrong, and by a different amount per sub-dataset:

| sub-dataset | true axis                | label is off by |
|-------------|--------------------------|-----------------|
| camels      | 1951-01-01 .. 2023-12-30 | +365 d          |
| camelsbr    | 1951-01-01 .. 2023-12-30 | +365 d          |
| camelscl    | 1951-01-01 .. 2023-12-30 | +365 d          |
| camelsgb    | 1951-01-02 .. 2023-12-31 | +366 d          |
| camelsaus   | 1950-01-02 .. 2022-12-31 | +1 d            |

The 561 camelsaus stations also had 27027 source days truncated to the first
26662 (`CombineCamels.py:285`).

Two consequences:

1. **MSWEP is aligned to the true dates, not the labels.** Column-for-column it
   is in register with `total_precipitation_sum` and every other forcing in the
   file, which is what the model needs. Aligning to the printed `time` instead
   would have shifted MSWEP against the rest of the forcings by up to a year.
2. **Any date-based split is off by a year** for four of the five sub-datasets,
   and inconsistently so across them. Splitting by column index is unaffected.

A companion variable `time_offset_days` (station_ids) is written with the new
forcing: `true_date[s, k] = time[k] + time_offset_days[s]`.

## Method

Caravan's `total_precipitation_sum` is an area-weighted mean of ERA5-Land over
the catchment polygon, not a point sample at the gauge, so MSWEP is reduced the
same way: exact polygon/cell intersection areas over the Caravan basin
shapefiles, scaled by cos(latitude) so cells count by true surface area, then
normalised per basin. 3026 basins, 381,011 nonzero weights, 108,344 distinct
0.1° cells, MSWEP rows 314-1443.

Cross-check on the weights: cos-lat polygon area against Caravan's own `area`
attribute gives a ratio of 1.0022 at every percentile from p1 to p99, with zero
basins off by more than 2x. That flat 1.0022 is just (111.32/111.195)² -- the
equatorial vs mean Earth radius in the km-per-degree constant -- so every gauge
matched the right polygon.

Fill handling: the archive declares `_FillValue = -9999.0` but actually stores
`-239976.0` (= -9999 × 24, a fill that survived an hourly->daily unit conversion).
Anything ≤ -1 is treated as missing, weights renormalised, and a day set to NaN
if under 50% of the basin has data. In practice the only fills are a 150×150
block north of 75°N and no basin touches them (min valid area fraction = 1.000).

## Pipeline

| step | script | cost | output |
|---|---|---|---|
| 00 | `00_station_time_axis.py` | ~10 min | `station_time_axis.csv` -- true first date per station |
| 01 | `01_build_weights.py` | ~1 min | `basin_weights.npz`, `basin_weights_report.csv` |
| 02 | `02_extract_mswep.py` | ~35 min total | `chunks/mswep_YYYY.npz`, one per year |
| 03 | `03_merge_into_caravan.py` | ~20 min | `..._stationids_MSWEP.nc` |

Steps 00 and 01 have already been run; their outputs are in this directory.

```bash
cd ${oc.env:DMG_DATA_ROOT}/caravan_zenodo/mswep_forcing
sbatch run_02_extract.sh              # array 1979-2023, ~1 min/task
sbatch --dependency=afterok:<JOBID> run_03_merge.sh
```

Step 02 is checkpointed per year and skips finished ones, so resubmitting the
array is safe. Step 03 must not run until the array is complete -- a missing year
becomes a NaN block, not an error. `--inplace` on step 03 appends to the master
file instead of writing a copy.

Step 03 prints the validation summary: per-basin daily correlation against
ERA5-Land and mean-precipitation bias. Single-day spatial correlations on
spot-checked days ran 0.65-0.96 by sub-dataset once aligned (they were ~0.06
before the time-axis correction, which is how the bug surfaced).

## Using it

In the model config, swap the precipitation entry:

```yaml
  phy:
    raw_forcings: [
      total_precipitation_sum_MSWEP,     # was total_precipitation_sum
      temperature_2m_max,
      temperature_2m_min,
      surface_net_solar_radiation_mean,
      potential_evaporation_sum_FAO_PENMAN_MONTEITH,
    ]
  nn:
    forcings: [
      total_precipitation_sum_MSWEP,     # was total_precipitation_sum
      ...
    ]
```

Two things to keep in mind:

- **Record start.** MSWEP begins 1979-01-01, so columns before each station's
  1979 column are NaN -- roughly the first 10227 (camels/br/cl), 10226
  (camelsgb) or 10591 (camelsaus) columns. With `rho: 365` and `warmup: 365`
  the training window has to start after that; ERA5-Land covers 1950 onward, so
  this is a real reduction in usable record, not a cosmetic one.
- **The `nn.attributes` list is still ERA5-derived.** `p_mean`,
  `aridity_FAO_PM`, `seasonality_FAO_PM`, `frac_snow`, `high_prec_freq`,
  `high_prec_dur`, `low_prec_freq`, `low_prec_dur` and `moisture_index_FAO_PM`
  are all computed from ERA5-Land precipitation. Feeding MSWEP as forcing while
  those attributes still describe ERA5 is a mild inconsistency -- probably
  second-order next to the forcing swap itself, but worth noting if the
  comparison comes out close. Recomputing them from MSWEP is a bounded follow-up
  (Caravan's formulas are in `caravan_point_forcings/caravan_indices.py`).
