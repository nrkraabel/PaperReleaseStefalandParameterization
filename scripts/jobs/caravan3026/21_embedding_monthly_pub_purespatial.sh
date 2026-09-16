#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=150GB
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --partition=<PARTITION>
#SBATCH --account=<ACCOUNT>
#SBATCH --job-name=Caravan3026_Embedding_Monthly_PureSpatial
#SBATCH --array=0-2
#SBATCH --output=%x_%A_%a.out

# Lighter budget than the daily variant: monthly condensed embedding file is
# 755MB vs. daily's 15.3GB, so the per-holdout memory growth
# scripts/jobs/caravan3026/10_embedding_hbv_pub_purespatial.sh's comment describes is much
# smaller here. Still generous relative to a non-spatial run since
# run_spatial_testing() accumulates all 5 folds' predictions in memory.
#
# Depends on: scripts/data_prep/convert_caravan3026_task_station_schema.sh.

source "${DMG_ENV}/bin/activate"
export PYTHONPATH="$PWD:$PYTHONPATH"

SEEDS=(111111 222222 333333)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

OUT_DIR=${DMG_OUTPUT_ROOT}/Caravan3026/Embedding/monthly/pure_spatial_1998-2008/seed${SEED}/

python ${DMG_REPO}/src/dmg/__main__.py \
    --config-name Caravan3026/EmbeddingMonthlyPUB_PureSpatial \
    seed=${SEED} \
    save_path=$OUT_DIR \
    +out_path=$OUT_DIR \
    +run_dir=$OUT_DIR \
    +eval_output_key=[streamflow]
