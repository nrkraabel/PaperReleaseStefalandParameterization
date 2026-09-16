#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=120GB
#SBATCH --gpus=1
#SBATCH --time=20:00:00
#SBATCH --partition=<PARTITION>
#SBATCH --account=<ACCOUNT>
#SBATCH --job-name=Camels531_CondensedDaily_PUB_NoAdapter
#SBATCH --array=0-2
#SBATCH --output=%x_%A_%a.out

# Ablation on the proposed method for the Extended1980_1999 Camels531
# experiment: the condensed (64-d) daily StefaLand256 embedding, 10-fold PUB.
# Deliberately launched as the SAME command as the baseline
# test_condensed_embedding_camels531_daily_pub.sh with one extra override, so
# config, window (train 1980/10-1999/09, test 1995/10-1999/09), folds, seeds
# and eval keys cannot drift from the run this is compared against.
# Ablation override: model.nn.adapter_type=none

source "${DMG_ENV}/bin/activate"
export PYTHONPATH="$PWD:$PYTHONPATH"

SEEDS=(111111 222222 333333)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

OUT_DIR=${DMG_OUTPUT_ROOT}/Extended1980_1999_Camels531/Camels_531_condensed_embeddings_daily_noadapter/PUB/seed${SEED}/
COND_DIR=${DMG_EMBEDDING_ROOT}/Condensers

python ${DMG_REPO}/src/dmg/__main__.py \
    --config-name EmbeddingPUB531 \
    seed=${SEED} \
    embedding_path=${COND_DIR}/Camels531_StefaLand256_condenser_daily_embedding.nc \
    model.nn.embedding_size=64 \
    model.nn.adapter_type=none \
    save_path=$OUT_DIR \
    +out_path=$OUT_DIR \
    +run_dir=$OUT_DIR \
    +eval_output_key=[streamflow,recharge,percolation,SM,parFC]
