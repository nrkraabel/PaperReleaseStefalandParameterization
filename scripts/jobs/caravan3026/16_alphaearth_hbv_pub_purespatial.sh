#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=200GB
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --partition=<PARTITION>
#SBATCH --account=<ACCOUNT>
#SBATCH --job-name=Caravan3026_LSTMHBV_AlphaEarth_PureSpatial
#SBATCH --array=0-2
#SBATCH --output=%x_%A_%a.out

# Differentiable HBV+LSTM baseline with 64 AlphaEarth satellite-embedding
# static attributes (ae_00..ae_63) appended -- see
# conf/Caravan3026/LSTMHBVAlphaEarthPUB_PureSpatial.yaml. Isolates whether
# AlphaEarth features help on their own, with no foundation-model embedding
# anywhere in the loop, which is what makes this directly comparable to
# scripts/jobs/caravan3026/13_lstm_hbv_pub_purespatial.sh (identical in every other respect).
# 200GB rather than the baseline's 150GB: run_spatial_testing() rebuilds the
# model/loader/trainer per holdout and accumulates every holdout's
# predictions/targets, so per-holdout static-attribute cost matters -- and
# this run carries 86 statics vs. the baseline's 22. That same jump from 47
# to 111 statics is what pushed run_embedding_global3434_pub_40mfinalized_
# alphaearth.sh from 120GB to 500GB.
# Depends on: the AlphaEarth merge step having completed
# (writes the ae_00..ae_63-augmented task file this config's data_path points
# to), which in turn depends on
# the station-id schema fix.

source "${DMG_ENV}/bin/activate"
export PYTHONPATH="$PWD:$PYTHONPATH"

SEEDS=(111111 222222 333333)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

# Model dir is AlphaEarth/ (not LSTMHBV/) so collect_caravan3026_seeds.py and
# caravan3026_cdf_3seed_panels.py pick these up as the 'AlphaEarth dHBV'
# series -- that series is currently a single dashed run and upgrades itself
# to a solid 3-seed average as soon as these seeds are collected.
OUT_DIR=${DMG_OUTPUT_ROOT}/Caravan3026/AlphaEarth/pure_spatial_1998-2008/seed${SEED}/

python ${DMG_REPO}/src/dmg/__main__.py \
    --config-name Caravan3026/LSTMHBVAlphaEarthPUB_PureSpatial \
    seed=${SEED} \
    save_path=$OUT_DIR \
    +out_path=$OUT_DIR \
    +run_dir=$OUT_DIR \
    +eval_output_key=[streamflow]
