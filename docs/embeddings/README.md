# Foundation-model embeddings for streamflow modeling

Nothing here is tied to a particular dataset. It works with your own catchments.

```
 stage 0 (optional)       stage 1              stage 2 (optional)    stage 3
 generate_embeddings  ->  train_condensed_ ->  apply_condenser  ->  src/dmg/__main__.py
 raw FM embeddings        embedding            same condenser       fine-tune a
 (station, time, D)       D -> width           on other stations    streamflow model
```

| stage | script | needs streamflow? | needs GPU? |
|---|---|---|---|
| 0 | `scripts/generate_embeddings.py` | no | recommended |
| 1 | `scripts/train_condensed_embedding.py` | no | recommended |
| 2 | `scripts/apply_condenser.py` | no | no |
| 3 | `python src/dmg/__main__.py --config-name <config>` | yes | yes |

Only stage 3 sees streamflow. Editable SLURM scripts for each stage are in
`scripts/templates/`, configs in `conf/templates/` and
`conf/observations/_dataset_template.yaml`. Fill in every `<PLACEHOLDER>`.

## Setup

```bash
uv venv && source .venv/bin/activate
uv pip install -e .
uv pip install netCDF4
uv pip install -e '.[hydrodl2]'   # for conf/templates/embedding_finetune_hbv.yaml
```

## Stage 0: generate embeddings

Skip if you already have an embedding file in the layout below.

You need a pretrained encoder checkpoint and a NetCDF holding the exact input
variables it was trained on. Copy `conf/templates/_encoder_arch_template.yaml`,
set the architecture, checkpoint path and variable lists, then:

```bash
python scripts/generate_embeddings.py \
    --config conf/templates/my_encoder.yaml \
    --out_dir /path/to/embeddings \
    --dataset_name MyDataset \
    --resolutions daily monthly
```

Writes `<out_dir>/<dataset_name>/<dataset_name>_embeddings_<resolution>.nc`.
`daily` is full resolution and large; `monthly`, `seasonal` and `annual` are
mean-pooled and much smaller.

> Read the variable lists off the checkpoint, not off a training config.
> Weights are copied by name and shape only. A wrong list still loads the
> variables that match and silently leaves the rest randomly initialized.
> Check the `Loaded N parameters` log line.

## Stage 1: train a condenser

A linear map from the raw embedding width (e.g. 256) down to a smaller one
(e.g. 64), trained on reconstruction loss over the whole station set and time
range. It only ever sees the embedding, so it can then be applied to any
stations, including ungauged ones.

```bash
python scripts/train_condensed_embedding.py \
    --embedding_nc MyDataset_embeddings_daily.nc \
    --condensed_width 64 \
    --out condensers/MyDataset_condenser_daily64.pt --export_embedding
```

`--start_time` and `--end_time` subset the embedding file's own time axis.

**Memory:** the full embedding array for the selected stations and time range
is read into memory. The script prints its size first. Tens of GB at daily
resolution.

`--export_embedding` also writes `<out>_embedding.nc`, the condensed embedding
for those stations, ready for stage 3.

## Stage 2: apply a condenser

Applies a trained condenser to another embedding file of the same width, for
example a larger or ungauged station set.

```bash
python scripts/apply_condenser.py \
    --condenser condensers/MyDataset_condenser_daily64.pt \
    --embedding_nc OtherStations_embeddings_daily.nc \
    --out OtherStations_condensed_daily64.nc
```

Output keeps the input file's own stations and time axis. Stage 3 does the
alignment when it loads.

## Stage 3: fine-tune a streamflow model

1. Copy `conf/observations/_dataset_template.yaml` to
   `conf/observations/<my_dataset>.yaml` and fill it in.
2. Copy a template from `conf/templates/`, point its `defaults:` entry at
   `<my_dataset>`, and set `embedding_path`, the dates, `target`, `forcings`,
   `attributes` and `embedding_size`.
3. From the repo root:

   ```bash
   python src/dmg/__main__.py --config-name templates/my_experiment
   ```

| template | what it does | extra requirements |
|---|---|---|
| `embedding_finetune_temporal.yaml` | Start here. Predicts streamflow directly, train/test split by date. | none |
| `embedding_finetune_spatial_pub.yaml` | Same model, k-fold holdout by basin | `gage_split_file`, `station_ids`, integer station ids |
| `embedding_finetune_hbv.yaml` | Network predicts HBV parameters, HBV produces streamflow | `hydrodl2` |

`model.nn.embedding_size` **must equal the last dimension of the file at
`embedding_path`**. Nothing infers it.

Outputs go to two places: `save_path` gets the training log (`results.txt`,
`loss_data.csv`); everything else goes to Hydra's run directory,
`output/<name>/` under the launch directory.

## File layouts

### Embedding NetCDF (stages 1-3)

| item | requirement |
|---|---|
| variable | `embedding` (name set by `--embedding_var_name` / `model.nn.embedding_var_name`) |
| dims | `(station_ids, time, embed_dim)`, or `(station_ids, embed_dim)` for one static embedding per station |
| coords | `station_ids` (matched to the task file by value, as strings) and `time` |

Time resolution can be anything. Each task day takes the latest embedding
dated on or before it, so a monthly file holds each month's embedding constant
and no separate code path is needed per resolution.

### Task NetCDF (stage 3)

| item | requirement |
|---|---|
| dims | time series `(station_ids, time)`, static attributes `(station_ids,)` |
| coords | `station_ids`, `time` (daily) |
| variables | every name in `model.nn.forcings`, `model.nn.attributes` and `train.target` |
| missing data | `NaN`. Target values below -10 are also treated as missing. |

### Spatial-holdout files (spatial template only)

| file | requirement |
|---|---|
| `gage_split_file` (CSV) | a `gage` column, plus `PUB_ID` (for `extent: PUB`) or `huc` (for `extent: PUR`) |
| `station_ids` (text) | one id per line, **in the same order as the task NetCDF's `station_ids` dimension** |

## Things that fail silently

These have all produced wrong results without raising an error.

1. **Stations with no embedding get zeros.** A task station with no matching
   `station_ids` entry in the embedding file gets an all-zero embedding, as do
   task days before the embedding file's first date. The loader logs the
   matched counts (`15/20 stations matched`). Read it.
2. **Spatial holdout needs integer station ids.** The basin split calls `int()`
   on every id, so `227225A` or `camels_01013500` raise. Temporal splits do not
   have this restriction.
3. **The spatial split is positional.** The `station_ids` text file is matched
   to the task file by line order, not by value. A reordered list assigns
   stations to the wrong folds.
4. **HBV physics forcings are positional.** `model.phy.raw_forcings` is read as
   `[precipitation, Tmax, Tmin, shortwave radiation, (optional measured PET)]`.
   If those cannot be read, the loader warns and continues with empty physics
   forcings.
5. **Checkpoint variable mismatches in stage 0.** See the note under stage 0.
