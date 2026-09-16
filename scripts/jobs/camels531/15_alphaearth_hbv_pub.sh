#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=120GB
#SBATCH --gpus=1
#SBATCH --time=96:00:00
#SBATCH --partition=<PARTITION>
#SBATCH --account=<ACCOUNT>
#SBATCH --job-name=Extended1980_1999_LSTMAlphaEarthHBV_PUB
#SBATCH --array=0-2
#SBATCH --output=%x_%A_%a.out

# Depends on Camels_Pretrain_AlphaEarth.nc existing
# (scripts/merge_alphaearth_camels_pretrain.py) -- no embedding-gen job
# dependency, just the one-off data merge.

source "${DMG_ENV}/bin/activate"
export PYTHONPATH="$PWD:$PYTHONPATH"

SEEDS=(111111 222222 333333)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

OUT_DIR=${DMG_OUTPUT_ROOT}/Extended1980_1999_Camels531/LSTMAlphaEarthHBV/PUB/seed${SEED}/

python ${DMG_REPO}/src/dmg/__main__.py \
    --config-name camels531/LSTMAlphaEarthHBVPUB \
    seed=${SEED} \
    save_path=$OUT_DIR \
    +out_path=$OUT_DIR \
    +run_dir=$OUT_DIR \
    +eval_output_key=[streamflow,recharge,percolation,SM,parFC]
