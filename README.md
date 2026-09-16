# Foundation-model embeddings in a differentiable hydrologic model

Code for A Landscape Representation Learning Model Improves Parameterization and Internal States of a Differentiable Hydrologic Model
A modified copy of [𝛿MG](https://github.com/mhpi/generic_deltamodel), adapted
so a differentiable rainfall-runoff model can take pretrained foundation-model
embeddings as input.

Upstream 𝛿MG trains a neural network on raw forcings and static attributes,
and that network predicts the parameters of a process-based model such as HBV.
Here a frozen, pretrained encoder sits in front of it. The encoder runs once
over the forcing record to produce a per-catchment, per-timestep embedding;
that embedding is stored, compressed by a learned condenser, and fed to the
parameterization network through a small trainable adapter. HBV is unchanged.

## Install

```bash
uv venv && source .venv/bin/activate
uv pip install -e .
uv pip install netCDF4
uv pip install -e '.[hydrodl2]'
```

Run experiments with `python src/dmg/__main__.py --config-name <name>`. Some
modules import each other as `from models....` rather than
`from dmg.models....`, which resolves because Python puts the script's own
directory on `sys.path`, so `python -m dmg` will not work.

## Paths

No absolute paths are baked in. Set these before running anything:

`DMG_DATA_ROOT` for task NetCDFs, gage lists and basin splits.
`DMG_EMBEDDING_ROOT` for generated embeddings and trained condensers.
`DMG_OUTPUT_ROOT` for checkpoints, predictions and metrics.
`DMG_CHECKPOINT_ROOT` for pretrained encoder checkpoints.
`DMG_REPO` and `DMG_ENV` for the SLURM jobs.

Configs read them through OmegaConf, which raises if one is unset.

## The pipeline

Four stages. Generate embeddings by running the frozen encoder over the
forcing record. Train a condenser, a linear map from the encoder's width down
to 64, on reconstruction loss. Optionally apply that condenser to another
station set. Then fine-tune the streamflow model.

Only the last stage sees streamflow, so the condenser can be applied to
ungauged basins. The embedding's own time resolution can be anything: each
task day takes the latest embedding dated on or before it, so daily, monthly
and annual files need no separate code path.

## Experiments

CAMELS-531 over 1980 to 1999, and Caravan-3026 over 1998 to 2008. Both run as
5-fold prediction-in-ungauged-basins holdouts, three seeds each (111111,
222222, 333333). Caravan also has splits by date, and by basin and date
together.

Components: the method itself; alternative pretrained encoders; the 𝛿HBV LSTM
baseline; LSTM without HBV; the same transformer trained from scratch;
AlphaEarth static embeddings; MSWEP precipitation in place of ERA5-Land; and
the encoder's raw input variables with no encoder, which separates the value
of pretraining from the value of the variable set. Ablations remove the
adapter, replace it with a linear probe on the embedding alone, concatenate
the embedding to the LSTM inputs, coarsen the embedding to monthly and annual,
and remove HBV.

Every component has a config, and the headline ones have a worked three-seed
SLURM job. There is also a generic job that takes any config name.

## Data

Nothing is redistributed here. You will need CAMELS (US) and Caravan from
their own archives, AlphaEarth and MSWEP for the runs that use them, and a
pretrained encoder checkpoint.

Read the encoder's variable lists off the checkpoint, not off a training
config. Weights are copied by name and shape only, so a wrong list still loads
the variables that match and leaves the rest randomly initialized, without
raising.

## Citation

Please cite the paper, and 𝛿MG:


> Shen, C., Appling, A.P., Gentine, P. et al. Differentiable modelling to unify
> machine learning and physical models for geosciences. *Nat Rev Earth Environ*
> **4**, 552-567 (2023). https://doi.org/10.1038/s43017-023-00450-9

## License

Non-commercial license from The Pennsylvania State University, inherited
unchanged from 𝛿MG. 𝛿MG is maintained by
[MHPI](http://water.engr.psu.edu/shen/).
