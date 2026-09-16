#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=400GB
#SBATCH --gpus=1
#SBATCH --time=48:00:00
#SBATCH --partition=<PARTITION>
#SBATCH --account=<ACCOUNT>
#SBATCH --job-name=Caravan3026_Embedding_Daily_PureSpatial_NoHBV
#SBATCH --array=0-2
#SBATCH --output=%x_%A_%a.out

# Pure-ML counterpart of scripts/jobs/caravan3026/10_embedding_hbv_pub_purespatial.sh: same
# embedding+adapter architecture (EmbeddingFinetuneing) but with model.phy
# (Hbv_1_1p_Triton) removed entirely, so it's directly comparable to the
# CudnnLstmModel baseline in run_lstm_nohbv_pub_purespatial.sh /
# LSTMNoHBVPUB_PureSpatial.yaml. See scripts/jobs/caravan3026/10_embedding_hbv_pub_purespatial.sh
# for the 400GB/48h memory rationale (same embedding file, same PUB fold
# accumulation cost).
#
# Depends on: scripts/data_prep/convert_caravan3026_task_station_schema.sh having
# completed (writes the station_ids-fixed task file this config's data_path
# points to).

source "${DMG_ENV}/bin/activate"
export PYTHONPATH="$PWD:$PYTHONPATH"

SEEDS=(111111 222222 333333)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

OUT_DIR=${DMG_OUTPUT_ROOT}/Caravan3026/Embedding/daily/pure_spatial_1998-2008_nohbv/seed${SEED}/

python ${DMG_REPO}/src/dmg/__main__.py \
    --config-name Caravan3026/EmbeddingDailyNoHBVPUB_PureSpatial \
    seed=${SEED} \
    save_path=$OUT_DIR \
    +out_path=$OUT_DIR \
    +run_dir=$OUT_DIR \
    +eval_output_key=[streamflow]
