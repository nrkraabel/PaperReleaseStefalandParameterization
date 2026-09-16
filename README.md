# Foundation-model embeddings in a differentiable hydrologic model

This repository is a modified copy of [𝛿MG](https://github.com/mhpi/generic_deltamodel),
the PyTorch differentiable modeling framework, adapted so that a differentiable
rainfall-runoff model can take pretrained foundation-model embeddings as input.
It is the code accompanying `<PAPER TITLE>`, `<AUTHORS>`, `<VENUE/YEAR>`.

Upstream 𝛿MG trains a neural network on raw forcings and static attributes,
and that network predicts the parameters of a process-based model such as HBV.
The work here inserts a frozen, pretrained encoder in front of that network.
The encoder is run once over the forcing record to produce a per-catchment,
per-timestep embedding; the embedding is stored to disk, optionally compressed
by a learned condenser, and then fed to the parameterization network through a
small trainable adapter. HBV itself is unchanged.

Everything upstream 𝛿MG could do, it still does. What is new here is the
embedding path: a loader, a parameterization network, an adapter layer, a
frozen-encoder wrapper, and the offline scripts that produce and compress the
embeddings. [CHANGES_VS_DMG.md](./CHANGES_VS_DMG.md) lists exactly which files
are new or modified and why.

## Contents

| Path | What is in it |
|---|---|
| `src/dmg/` | The framework, with the embedding additions (see CHANGES_VS_DMG.md) |
| `scripts/` | The four-stage embedding pipeline, plus data prep and SLURM jobs |
| `conf/camels531/` | CAMELS-531 experiments, every component in the paper |
| `conf/Caravan3026/` | Caravan-3026 experiments, every component in the paper |
| `conf/templates/` | Blank configs for running the pipeline on your own catchments |
| `analysis/` | Collects finished multi-seed runs and produces the figures and tables |
| `docs/embeddings/` | The pipeline guide, including the failure modes worth knowing |
| `docs/` | Upstream 𝛿MG documentation, unmodified |

## Install

```bash
uv venv && source .venv/bin/activate
uv pip install -e .
uv pip install netCDF4
uv pip install -e '.[hydrodl2]'   # HBV and the other process-based models
```

Then check the install on a GPU node. This needs no data:

```bash
python scripts/smoke_test_embedding_ablations.py
```

Run experiments as `python src/dmg/__main__.py --config-name <name>`. A few
modules under `src/dmg/models/neural_networks/` import each other as
`from models....` rather than `from dmg.models....`, which resolves because
Python puts the script's own directory on `sys.path`. Launching that way is
therefore the supported entry point; `python -m dmg` will not work for the
embedding configs.

## Paths

No absolute paths are baked into the configs. Six environment variables stand
in for them. Copy `.env.example`, fill it in, and source it before running
anything:

| Variable | What it points at |
|---|---|
| `DMG_DATA_ROOT` | Task NetCDFs, gage lists, PUB/PUR split CSVs |
| `DMG_EMBEDDING_ROOT` | Generated embeddings and trained condensers |
| `DMG_OUTPUT_ROOT` | Where runs write checkpoints, predictions and metrics |
| `DMG_CHECKPOINT_ROOT` | Pretrained foundation-model checkpoints |
| `DMG_REPO` | This repository, for the SLURM jobs |
| `DMG_ENV` | The virtualenv, for the SLURM jobs |

## The pipeline

Four stages. [docs/embeddings/README.md](./docs/embeddings/README.md) is the
full guide; this is the shape of it.

```
stage 0                stage 1              stage 2            stage 3
generate_embeddings -> train_condensed_ -> apply_condenser -> src/dmg/__main__.py
frozen encoder over     embedding           same condenser     train the
the forcing record      D -> width          on other gauges    streamflow model
```

Stages 0 to 2 are offline and produce NetCDF files. Only stage 3 trains the
differentiable model. If you already have an embedding file in the layout
documented in the guide, start at stage 3.

## Experiments

Two datasets, run as 5-fold prediction-in-ungauged-basins (PUB) holdouts, three
seeds each (111111, 222222, 333333).

CAMELS-531, `conf/camels531/`, 1980 to 1999:

| Config | Component |
|---|---|
| `EmbeddingMFFormerPUB` / `PUR` | The method: MFFormer embeddings, adapter, HBV |
| `Embedding40MFinalizedPUB`, `EmbeddingICLM256PUB` | Alternative pretrained encoders |
| `EmbeddingStefaLandGrid64PUB` | Gridded land embedding instead of the forcing encoder |
| `EmbeddingNonePUB` | Embedding path with the adapter removed |
| `LSTMHBVPUB` / `PUR` | The 𝛿HBV LSTM baseline |
| `LSTMPUB` / `PUR` | LSTM without HBV |
| `LSTMAlphaEarthHBVPUB` | AlphaEarth static embeddings instead of a forcing encoder |
| `NoPretrainingHBVPUB` | Same transformer, trained from scratch, no pretraining |
| `RawFmInputsPUB` | The encoder's raw input variables, no encoder |

Caravan-3026, `conf/Caravan3026/`, 1998 to 2008:

| Config | Component |
|---|---|
| `EmbeddingDailyPUB_PureSpatial` | The method at daily embedding resolution |
| `EmbeddingMonthlyPUB_*`, `EmbeddingAnnualPUB_*` | Coarser embedding resolutions |
| `EmbeddingDailyNoAdapterPUB_PureSpatial` | Ablation: adapter removed |
| `EmbeddingDailyLinearProbePUB_PureSpatial` | Ablation: linear probe on the embedding alone |
| `EmbeddingDailyAsInputPUB_PureSpatial` | Ablation: embedding concatenated to the LSTM inputs |
| `EmbeddingDaily*NoHBV*` | Same, with HBV removed |
| `EmbeddingDailyMSWEPPUB_*` | MSWEP precipitation instead of ERA5-Land |
| `LSTMHBVPUB_*`, `LSTMNoHBVPUB_*` | Baselines |
| `LSTMHBVAlphaEarthPUB_*` | AlphaEarth static embeddings |
| `MFFormerNoPretrainPUB_*` | Same transformer, no pretraining |
| `RawFmInputsPUB_PureSpatial` | Encoder inputs, no encoder |

`_PureSpatial` trains and tests on the same calendar window and splits only by
basin. `_SpatialTemporal` splits by basin and by date. `Temporal` splits by
date only.

### Running them

`scripts/jobs/` holds one worked SLURM job per component, as a three-seed array.
They are written for SLURM but the last few lines are a plain `python` call, so
they read fine as a record of what was run. Fill in `<PARTITION>` and
`<ACCOUNT>` before submitting.

```bash
sbatch scripts/jobs/caravan3026/10_embedding_hbv_pub_purespatial.sh
```

Configs without a matching job run the same way. Copy
`scripts/jobs/run_experiment.sh`, which takes a config name and does the
three-seed sweep for any of them.

Jobs numbered `00` and `01` are the stage 0 and stage 1 pipeline steps, and
must finish before the `1x` fine-tuning jobs can start.

## Analysis

`analysis/` turns finished runs into the paper's figures and tables. Collect
first, then plot:

```bash
python analysis/collect_caravan3026_seeds.py
python analysis/caravan3026_cdf_3seed_panels.py
```

The collection scripts copy raw prediction and target arrays rather than the
stored `metrics.json`, and the plotting scripts recompute per-basin NSE and KGE
from those arrays. That is deliberate: a re-tested run leaves a stale
`metrics.json` next to fresh predictions, and reading the JSON silently
reproduces the superseded numbers. See the header of
`analysis/collect_caravan3026_seeds.py`.

## Data

The datasets are not redistributed here. You will need:

- CAMELS (US), and Caravan, from their own archives.
- A pretrained encoder checkpoint. The embedding stage reads the variable
  lists off the checkpoint, not off a training config. Getting this wrong
  fails quietly, so read the note in the pipeline guide.
- AlphaEarth and MSWEP, for the comparison runs that use them.
  `scripts/data_prep/` has the scripts that merge these into the task files.

## Citation

Please cite the paper, and 𝛿MG:

> `<PAPER CITATION>`

> Shen, C., Appling, A.P., Gentine, P. et al. Differentiable modelling to unify
> machine learning and physical models for geosciences. *Nat Rev Earth Environ*
> **4**, 552-567 (2023). https://doi.org/10.1038/s43017-023-00450-9

## License

Non-commercial license from The Pennsylvania State University, inherited
unchanged from 𝛿MG. See [LICENSE](./LICENSE). 𝛿MG is maintained by
[MHPI](http://water.engr.psu.edu/shen/).
