#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=150GB
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --partition=<PARTITION>
#SBATCH --account=<ACCOUNT>
#SBATCH --job-name=Caravan3026_LSTMHBV_PureSpatial
#SBATCH --array=0-2
#SBATCH --output=%x_%A_%a.out

# Differentiable HBV+LSTM baseline (no foundation model). See
# scripts/jobs/caravan3026/10_embedding_hbv_pub_purespatial.sh for the per-holdout memory-growth
# rationale behind this budget.
# Depends on: the station-id schema fix.

source "${DMG_ENV}/bin/activate"
export PYTHONPATH="$PWD:$PYTHONPATH"

SEEDS=(111111 222222 333333)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

OUT_DIR=${DMG_OUTPUT_ROOT}/Caravan3026/LSTMHBV/pure_spatial_1998-2008/seed${SEED}/

python ${DMG_REPO}/src/dmg/__main__.py \
    --config-name Caravan3026/LSTMHBVPUB_PureSpatial \
    seed=${SEED} \
    save_path=$OUT_DIR \
    +out_path=$OUT_DIR \
    +run_dir=$OUT_DIR \
    +eval_output_key=[streamflow]
