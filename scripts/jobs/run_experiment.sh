#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=120GB
#SBATCH --gpus=1
#SBATCH --time=48:00:00
#SBATCH --partition=<PARTITION>
#SBATCH --account=<ACCOUNT>
#SBATCH --job-name=DmgEmbeddingRun
#SBATCH --array=0-2
#SBATCH --output=%x_%A_%a.out

# Runs any config in conf/camels531/ or conf/Caravan3026/ over the three seeds
# used throughout the paper. The numbered scripts next to this one are the
# worked examples; this is the same thing with the config name as an argument,
# for the configs that do not have one.
#
#   sbatch --job-name=MyRun run_experiment.sh Caravan3026/EmbeddingDailyPUB_PureSpatial
#
# Memory is the one thing worth checking before submitting. run_spatial_testing()
# rebuilds the model, loader and trainer for every holdout and keeps each
# holdout's predictions and targets in memory, so peak usage grows with the fold
# count. The daily-resolution Caravan runs need 400GB; monthly and annual, and
# all of CAMELS-531, run in 120GB.

set -euo pipefail

CONFIG=${1:?usage: run_experiment.sh <config-name> [output-subdir]}
SUBDIR=${2:-$(echo "$CONFIG" | tr '/' '_')}

: "${DMG_ENV:?set DMG_ENV, see .env.example}"
: "${DMG_REPO:?set DMG_REPO, see .env.example}"
: "${DMG_OUTPUT_ROOT:?set DMG_OUTPUT_ROOT, see .env.example}"

source "${DMG_ENV}/bin/activate"
export PYTHONPATH="$PWD:$PYTHONPATH"

SEEDS=(111111 222222 333333)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

OUT_DIR=${DMG_OUTPUT_ROOT}/${SUBDIR}/seed${SEED}/

python ${DMG_REPO}/src/dmg/__main__.py \
    --config-name ${CONFIG} \
    seed=${SEED} \
    save_path=$OUT_DIR \
    +out_path=$OUT_DIR \
    +run_dir=$OUT_DIR \
    +eval_output_key=[streamflow]
