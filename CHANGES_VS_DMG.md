# What was changed in 𝛿MG

Baseline: [mhpi/generic_deltamodel](https://github.com/mhpi/generic_deltamodel).
This file lists what is new or modified here and why. Everything not mentioned
is upstream code, unchanged.

The shape of the change is small. Upstream, a neural network reads normalized
forcings and attributes (`xc_nn_norm`) and emits HBV parameters. Here a second
input stream carries a precomputed embedding (`xc_pretrained_norm`) alongside
it, and the parameterization network combines the two. Most of the edits below
exist to get that second stream from a NetCDF file on disk, through the loader
and sampler, into the network, without disturbing the single-stream path.

## New: the embedding path

| File | What it does |
|---|---|
| `core/data/loaders/embedding_finetune_loader.py` | Loads a precomputed embedding file alongside the task file and aligns the two. The embedding's own time resolution can be anything; each task day takes the latest embedding dated on or before it, so daily, monthly and annual files need no separate code path. |
| `models/neural_networks/embedding_finetuneing.py` | The parameterization network. Takes the embedding plus task forcings and attributes, runs the embedding through an adapter, and emits HBV parameters. Sized entirely from `embedding_size`, so it makes no assumption about which encoder produced the embedding. Also implements the ablations via `ablation_mode`: `linear_probe` and `embedding_as_input`. |
| `models/neural_networks/raw_fm_inputs_finetuneing.py` | Control condition. Same network shape, but fed the encoder's raw input variables instead of its embedding, which separates the value of pretraining from the value of the variable set. |
| `models/neural_networks/adapters/build_adapter.py` | Builds and applies the adapter by name, and is where `adapter_type: none` is handled. |
| `models/neural_networks/adapters/*.py` | The adapters themselves: `dual_residual` (used throughout the paper), plus `gated`, `feedforward`, `conv`, `attention`, `bottleneck` and `moe`. |
| `models/neural_networks/transformer/StefaLandDecLSTM.py` | The frozen encoder: a TransformerBackbone encoder with a single-layer LSTM decoder, matching the released checkpoint. |
| `trainers/finetune_trainer.py` | Training loop for the above. Handles the frozen encoder and the two input streams. |

## New: the offline pipeline

These are not part of the package. They produce the files stage 3 consumes.

| File | Stage |
|---|---|
| `scripts/generate_embeddings.py` | 0. Runs a frozen encoder over a forcing record and writes embeddings at daily, monthly, seasonal or annual resolution. |
| `scripts/train_condensed_embedding.py` | 1. Trains a linear condenser from the raw embedding width down to a smaller one, against reconstruction loss. It never sees streamflow. |
| `scripts/apply_condenser.py` | 2. Applies a trained condenser to another station set, for example ungauged basins. |

## Modified upstream files

| File | Change |
|---|---|
| `core/data/loaders/nn_dual_loader.py` | Substantially reworked to carry the second input stream, and to derive HBV's `[prcp, tmean, pet]` from `model.phy.raw_forcings` by Hargreaves-Samani, or to use measured PET directly when a fifth raw forcing is present. |
| `core/data/loaders/load_nc.py`, `loader_utils.py` | NetCDF reading and normalization statistics for the station-and-time-indexed task files these experiments use. |
| `core/data/samplers/hydro_sampler.py` | Subsets `xc_pretrained_norm` along with the rest of the batch, and skips it when the feature dimension is zero so plain LSTM configs are unaffected. |
| `models/delta_models/dpl_model.py` | Passes the whole batch dict to networks that set `ACCEPTS_BATCH_DICT`, instead of only `xc_nn_norm`. Also forwards the model-level `warmup` into the physics config as `warm_up`, so HBV sets its prediction cutoff correctly. |
| `models/model_handler.py` | Added `resolve_target_key`, an alias map from dataset target names (`Runoff`, `QObs`) to the physics model's output key (`streamflow`). Caravan and CAMELS disagree on this. |
| `core/utils/factory.py` | Registers the new network names, and picks the output width from the physics model's learnable parameter count, or from `out_size` when there is no physics model. |
| `core/utils/config.py` | Pydantic models accept extra keys, so experiment configs can carry fields the base schema does not define. |
| `core/utils/spatial_testing.py` | Redirects `output_dir`, `sim_dir`, `model_dir` and `plot_dir` per holdout. Without this every fold of a PUB run overwrites the same directory and only the aggregate survives. Also resolves the target key through the alias map above. |
| `models/neural_networks/direct_finetuneing.py` | Reworked; builds the frozen pretrained encoder that the non-embedding fine-tuning path uses. Only `stefaland_dec_lstm` is accepted, and an unknown `pretrained_type` now raises instead of silently falling back to another architecture. |
| `models/neural_networks/transformer/features_embedding.py`, `advanced_encoders.py` | Encoder internals the above depend on. |

## Removed from the upstream copy

Marketing images, the GitHub issue templates and CI workflows, the MoE and
tuning config trees, and the upstream tutorial notebooks and example configs.
Also removed: the kriging adapters, the stacked LSTM/MLP heads, the Triton
LSTM and the multi-timescale model handler, none of which any released config
selects. `models/multimodels/` and the Informer and Reformer
implementations are still present because the package imports them.

All absolute paths were replaced with environment variables. See the paths
table in [README.md](./README.md).
