#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=250GB
#SBATCH --gpus=1
#SBATCH --time=96:00:00
#SBATCH --partition=<PARTITION>
#SBATCH --account=<ACCOUNT>
#SBATCH --job-name=Caravan3026_MFFormerNoPretrain_PureSpatial
#SBATCH --array=0-2
#SBATCH --output=%x_%A_%a.out

# MFFormerDecLSTM architecture trained end-to-end from scratch (no loaded
# checkpoint, freeze_pretrained: False) -- heavier per-step than the plain
# CudnnLstmModel baselines since the full encoder trains too, so a bigger
# budget than scripts/jobs/caravan3026/13_lstm_hbv_pub_purespatial.sh, though still less than the
# daily-embedding jobs since there's no giant precomputed-embedding tensor.
#
# Depends on: scripts/data_prep/convert_caravan3026_task_station_schema.sh
# (data_path) -- pretrained_path (3026Carvan_refpoints.nc) already exists.

source "${DMG_ENV}/bin/activate"
export PYTHONPATH="$PWD:$PYTHONPATH"

SEEDS=(111111 222222 333333)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

OUT_DIR=${DMG_OUTPUT_ROOT}/Caravan3026/MFFormerNoPretrain/pure_spatial_1998-2008/seed${SEED}/

python ${DMG_REPO}/src/dmg/__main__.py \
    --config-name Caravan3026/MFFormerNoPretrainPUB_PureSpatial \
    seed=${SEED} \
    save_path=$OUT_DIR \
    +out_path=$OUT_DIR \
    +run_dir=$OUT_DIR \
    +eval_output_key=[streamflow]
