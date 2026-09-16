#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=150GB
#SBATCH --gpus=1
#SBATCH --time=20:00:00
#SBATCH --partition=<PARTITION>
#SBATCH --account=<ACCOUNT>
#SBATCH --job-name=Caravan3026_Embedding_Daily_MSWEP_PureSpatial
#SBATCH --array=0-2
#SBATCH --output=%x_%A_%a.out

# MSWEP-precipitation arm of the Caravan3026 PUB comparison:
# daily condensed embedding + dHBV, MSWEP precip.
# Identical to its ERA5-Land twin (scripts/jobs/caravan3026/10_embedding_hbv_pub_purespatial.sh) except for
# the precipitation product -- see the config header for the full list of
# what does and does not change.
#
# Resources, vs the ERA5 twin:
#   * mem cut from 400GB to 150GB. Observed MaxRSS across the six ERA5
#     3-seed runs was 88-91 GB (embedding) and 12.6-13.1 GB (LSTM), so this
#     is still >1.6x / >6x headroom. The old 400GB/150GB asks were the
#     binding constraint on how many array tasks could sit on one node
#     (mgc-mri GPU nodes have 1540/770 GB for 10/3-4 GPUs), so trimming them
#     is what lets all 18 tasks of this batch run concurrently instead of
#     queueing in waves.
#   * cpus-per-task 2 -> 4. The nodes are 40 CPU / 10 GPU, so 4 is the
#     largest per-GPU share that still lets a full node of tasks schedule;
#     asking for 8 would make CPU, not GPU, the limit.
#   * time trimmed to 20:00:00 (ERA5 twins ran ~9h20m), which also makes these
#     eligible for backfill.
#
# Depends on:
#   * scripts/data_prep/mswep_forcing/03_merge_into_caravan.py (writes the
#     _MSWEP.nc this config reads)
#   * scripts/data_prep/mswep_forcing/04_fill_interior_nans.py -- MUST have run.
#     The merged file had one interior NaN day (label 2001-12-31) for 284 US
#     camels basins, and loader_utils.calc_stats is not nan-aware, so that
#     single NaN would have silently zeroed the entire precipitation channel.

source "${DMG_ENV}/bin/activate"
export PYTHONPATH="$PWD:$PYTHONPATH"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

SEEDS=(111111 222222 333333)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

OUT_DIR=${DMG_OUTPUT_ROOT}/Caravan3026/Embedding_MSWEP/daily/pure_spatial_1998-2008/seed${SEED}/

python ${DMG_REPO}/src/dmg/__main__.py \
    --config-name Caravan3026/EmbeddingDailyMSWEPPUB_PureSpatial \
    seed=${SEED} \
    save_path=$OUT_DIR \
    +out_path=$OUT_DIR \
    +run_dir=$OUT_DIR \
    +eval_output_key=[streamflow]
