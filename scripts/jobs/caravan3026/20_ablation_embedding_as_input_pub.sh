#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=400GB
#SBATCH --gpus=1
#SBATCH --time=48:00:00
#SBATCH --partition=<PARTITION>
#SBATCH --account=<ACCOUNT>
#SBATCH --job-name=Caravan3026_Embedding_Daily_PureSpatial_AsInput
#SBATCH --array=0-2
#SBATCH --output=%x_%A_%a.out

# Ablation: daily embedding fed to the LSTM as ordinary input channels in
# place of the task's static attributes (ablation_mode: embedding_as_input).
# LSTM nx = 128 embedding + 7 forcings; the 22 Caravan static attributes are
# loaded but never reach the network.
#
# Same 3-seed / 5-fold-PUB protocol and 400GB budget as the baseline
# scripts/jobs/caravan3026/10_embedding_hbv_pub_purespatial.sh.

source "${DMG_ENV}/bin/activate"
export PYTHONPATH="$PWD:$PYTHONPATH"

SEEDS=(111111 222222 333333)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

OUT_DIR=${DMG_OUTPUT_ROOT}/Caravan3026/Embedding/daily_asinput/pure_spatial_1998-2008/seed${SEED}/

python ${DMG_REPO}/src/dmg/__main__.py \
    --config-name Caravan3026/EmbeddingDailyAsInputPUB_PureSpatial \
    seed=${SEED} \
    save_path=$OUT_DIR \
    +out_path=$OUT_DIR \
    +run_dir=$OUT_DIR \
    +eval_output_key=[streamflow]
