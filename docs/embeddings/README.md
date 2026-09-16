# Foundation-model embeddings for streamflow modeling

This branch contains the pipeline for using pretrained foundation-model
embeddings as inputs to streamflow models. It works with your own catchments:
nothing here is tied to a particular dataset.

```
 stage 0 (optional)        stage 1                  stage 2 (optional)       stage 3
 generate_embeddings  ->  train_condensed_  ->  apply_condenser  ->  src/dmg/__main__.py
 raw FM embeddings        embedding               same condenser on        fine-tune a
 (station, time, D)       D -> width condenser    other stations           streamflow model
```

| stage | script | needs streamflow? | needs GPU? |
|---|---|---|---|
| 0 | `scripts/generate_embeddings.py` | no | recommended |
| 1 | `scripts/train_condensed_embedding.py` | **optional** (see below) | recommended |
| 2 | `scripts/apply_condenser.py` | no | no |
| 3 | `python src/dmg/__main__.py --config-name <config>` | yes (it is the training target) | **yes** |

Ready-to-edit SLURM scripts for each stage are in `scripts/templates/`, and
config templates are in `conf/templates/` and
`conf/observations/_dataset_template.yaml`. Every value you need to fill in
is written as `<PLACEHOLDER>`.

---

## Setup

```bash
uv venv && source .venv/bin/activate
uv pip install -e .
uv pip install netCDF4
# only for conf/templates/embedding_finetune_hbv.yaml:
uv pip install -e '.[hydrodl2]'
```

Check your install on a GPU node. This needs no data files:

```bash
python scripts/smoke_test_embedding_ablations.py
```

---

## Stage 0: generate embeddings (optional)

**Skip this stage if you already have an embedding file** in the layout
described under [File layouts](#file-layouts).

You need a pretrained encoder checkpoint and a NetCDF containing the exact
input variables that checkpoint was trained on. Copy
`conf/templates/_encoder_arch_template.yaml`, set the architecture, checkpoint
path and variable lists, then run:

```bash
python scripts/generate_embeddings.py \
    --out_dir ${DMG_EMBEDDING_ROOT} \
    --config conf/templates/my_encoder.yaml \
    --out_dir /path/to/embeddings \
    --dataset_name MyDataset \
    --resolutions daily monthly
```

This writes `<out_dir>/<dataset_name>/<dataset_name>_embeddings_<resolution>.nc`.
`daily` gives full resolution and is large. `monthly`, `seasonal` and `annual`
are mean-pooled and much smaller.

> **Read the variable lists off the checkpoint, not from a training config.**
> Weights are copied by name and shape only. If a variable list is wrong, the
> load still succeeds for the variables that match and silently leaves the
> rest at random initialization. The template shows how to list the
> checkpoint's real variables. Afterwards, check the `Loaded N parameters`
> log line.

---

## Stage 1: train a condenser

A condenser is a linear map from the raw embedding width (e.g. 256) down to a
smaller width (e.g. 64). It is trained on the whole station set and time range.
The condenser only ever sees the embedding, so once trained it can be applied
to any stations (see stage 2).

It has two modes. The arguments you pass select the mode:

### Supervised: you have streamflow

```bash
python scripts/train_condensed_embedding.py \
    --embedding_nc MyDataset_embeddings_daily.nc \
    --task_nc my_task_data.nc --target_var streamflow \
    --condensed_width 64 --recon_weight 1.0 \
    --out condensers/MyDataset_condenser_daily64.pt --export_embedding
```

Loss = `streamflow_loss + recon_weight * reconstruction_loss`. A small
auxiliary LSTM tries to predict streamflow from the condensed embedding alone.
This pushes the condenser to keep information that matters for streamflow.
The auxiliary LSTM is discarded after training. `--recon_weight` above 1
favours keeping the embedding general; below 1 favours streamflow relevance.
The embedding is aligned onto the task file's daily calendar and station set.

### Recon-only: no streamflow

```bash
python scripts/train_condensed_embedding.py \
    --embedding_nc MyDataset_embeddings_daily.nc \
    --condensed_width 64 \
    --out condensers/MyDataset_condenser_daily64_recon.pt --export_embedding
```

Loss = `reconstruction_loss` only. This is a plain autoencoder compression on
the embedding file's own stations and time axis. Here `--start_time` and
`--end_time` subset the embedding file itself.

`--task_nc` and `--target_var` go together: pass both or neither. Passing
only one is an error. The script prints the mode it chose at startup, and the
checkpoint records it under `mode`.

**Memory:** the full embedding array for the selected stations and time range
is loaded into memory. The script prints its size before reading. At daily
resolution this can be tens of GB.

`--export_embedding` also writes `<out>_embedding.nc`, the condensed embedding
for the training stations, ready for stage 3.

---

## Stage 2: apply a condenser (optional)

This applies a trained condenser (from either mode) to another embedding file
with the same embedding width, for example a larger set of stations or
ungauged ones. It needs no task file and no streamflow.

```bash
python scripts/apply_condenser.py \
    --condenser condensers/MyDataset_condenser_daily64.pt \
    --embedding_nc OtherStations_embeddings_daily.nc \
    --out OtherStations_condensed_daily64.nc
```

The output keeps the input file's own stations and time axis. You don't need
to align it to anything, because stage 3 does the alignment when it loads.

---

## Stage 3: fine-tune a streamflow model

1. Copy `conf/observations/_dataset_template.yaml` to
   `conf/observations/<my_dataset>.yaml` and fill it in.
2. Copy a template from `conf/templates/`, point its `defaults:` entry at
   `<my_dataset>`, and set `embedding_path`, the dates, `target`, `forcings`,
   `attributes` and `embedding_size`.
3. Run from the repo root:

   ```bash
   python src/dmg/__main__.py --config-name templates/my_experiment
   ```

| template | what it does | extra requirements |
|---|---|---|
| `embedding_finetune_temporal.yaml` | **Start here.** Predicts streamflow directly. Train and test are split by date. | none |
| `embedding_finetune_spatial_pub.yaml` | Same model, k-fold holdout by basin (prediction in ungauged basins) | `gage_split_file`, `station_ids`, integer station ids |
| `embedding_finetune_hbv.yaml` | The network predicts HBV parameters and HBV produces streamflow | `hydrodl2` package |

`model.nn.embedding_size` **must equal the last dimension of the file at
`embedding_path`**. Nothing infers it. Use 256 for a raw 256-wide embedding, or
the `--condensed_width` you trained with.

Outputs go to two places:

- `save_path` gets the training log (`results.txt`, `loss_data.csv`).
- Everything else goes to Hydra's run directory, `output/<name>/` under the
  directory you launched from: model checkpoints (`model/`), metrics,
  normalization statistics and the resolved config.

---

## File layouts

### Embedding NetCDF (stages 1-3)

| item | requirement |
|---|---|
| variable | `embedding` (name set by `--embedding_var_name` / `model.nn.embedding_var_name`) |
| dims | `(station_ids, time, embed_dim)`, or `(station_ids, embed_dim)` for one static embedding per station |
| coords | `station_ids` (matched to the task file by value, as strings) and `time` |

The time resolution can be anything. At load time, each task day uses the
latest embedding dated on or before that day. So a monthly file holds each
month's embedding constant across that month, and no separate code path is
needed per resolution.

### Task NetCDF (stage 1 supervised, stage 3)

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

---

## Things that fail silently

These have all caused wrong results without raising an error:

1. **Stations with no embedding get zeros.** When a station in the task file
   has no matching `station_ids` entry in the embedding file, it gets an
   all-zero embedding. The same happens for task days before the embedding
   file's first date. The loader logs a warning with the matched counts
   (`15/20 stations matched`). Read it.
2. **Spatial holdout needs integer station ids.** The basin split converts
   every id with `int()`, so ids like `227225A` or `camels_01013500` raise an
   error. Temporal splits have no such restriction.
3. **The spatial split is positional.** The `station_ids` text file is matched
   to the task file by line order, not by value. A reordered list assigns
   stations to the wrong folds.
4. **HBV physics forcings are positional.** `model.phy.raw_forcings` is read as
   `[precipitation, Tmax, Tmin, shortwave radiation, (optional measured PET)]`.
   If those variables can't be read, the loader logs a warning and continues
   with empty physics forcings.
5. **Checkpoint variable mismatches in stage 0.** See the note under stage 0.
