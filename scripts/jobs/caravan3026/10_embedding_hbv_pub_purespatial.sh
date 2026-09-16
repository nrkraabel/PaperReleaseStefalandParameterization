#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=400GB
#SBATCH --gpus=1
#SBATCH --time=48:00:00
#SBATCH --partition=<PARTITION>
#SBATCH --account=<ACCOUNT>
#SBATCH --job-name=Caravan3026_Embedding_Daily_PureSpatial
#SBATCH --array=0-2
#SBATCH --output=%x_%A_%a.out

# 400GB budget follows the precedent set by
# an earlier run of the same shape on 3434 global gauges: a 120GB budget
# OOM-killed a similarly-scaled (3434-station) 5-fold PUB run because
# run_spatial_testing() rebuilds the model/loader/trainer per holdout AND
# accumulates every holdout's predictions/targets in memory -- peak memory
# grows with holdout count. This daily condensed embedding file is 15.3GB
# (bigger than that run's 7.2GB), so keeping the same generous budget.
# Depends on: the station-id schema fix having
# completed (writes the station_ids-fixed task file this config's data_path
# points to).

source "${DMG_ENV}/bin/activate"
export PYTHONPATH="$PWD:$PYTHONPATH"

SEEDS=(111111 222222 333333)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

OUT_DIR=${DMG_OUTPUT_ROOT}/Caravan3026/Embedding/daily/pure_spatial_1998-2008/seed${SEED}/

python ${DMG_REPO}/src/dmg/__main__.py \
    --config-name Caravan3026/EmbeddingDailyPUB_PureSpatial \
    seed=${SEED} \
    save_path=$OUT_DIR \
    +out_path=$OUT_DIR \
    +run_dir=$OUT_DIR \
    +eval_output_key=[streamflow]
