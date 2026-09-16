#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=400GB
#SBATCH --gpus=1
#SBATCH --time=48:00:00
#SBATCH --partition=<PARTITION>
#SBATCH --account=<ACCOUNT>
#SBATCH --job-name=Caravan3026_Embedding_Daily_PureSpatial_NoAdapter
#SBATCH --array=0-2
#SBATCH --output=%x_%A_%a.out

# Ablation: daily embedding + HBV, adapter layer removed (adapter_type: none).
#
# Same 3-seed / 5-fold-PUB protocol and the same 400GB budget as the baseline
# scripts/jobs/caravan3026/10_embedding_hbv_pub_purespatial.sh -- run_spatial_testing() rebuilds the
# model/loader/trainer per holdout AND accumulates every holdout's
# predictions/targets in memory, so peak memory grows with holdout count and
# is driven by the data, not by the (smaller) model.

source "${DMG_ENV}/bin/activate"
export PYTHONPATH="$PWD:$PYTHONPATH"

SEEDS=(111111 222222 333333)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

OUT_DIR=${DMG_OUTPUT_ROOT}/Caravan3026/Embedding/daily_noadapter/pure_spatial_1998-2008/seed${SEED}/

python ${DMG_REPO}/src/dmg/__main__.py \
    --config-name Caravan3026/EmbeddingDailyNoAdapterPUB_PureSpatial \
    seed=${SEED} \
    save_path=$OUT_DIR \
    +out_path=$OUT_DIR \
    +run_dir=$OUT_DIR \
    +eval_output_key=[streamflow]
