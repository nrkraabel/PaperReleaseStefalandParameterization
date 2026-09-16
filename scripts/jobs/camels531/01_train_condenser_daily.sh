#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48GB
#SBATCH --gpus=1
#SBATCH --time=06:00:00
#SBATCH --partition=<PARTITION>
#SBATCH --account=<ACCOUNT>
#SBATCH --job-name=TrainCondenser_Camels531_daily_MFFormer256
#SBATCH --output=%x_%j.out

# Standalone dual-loss condenser (scripts/train_condensed_embedding.py):
# jointly trains a 64-dim condenser against reconstruction +
# streamflow loss, using the FULL Camels531 basin/time set (QObs from
# Camels_Pretrain.nc directly) -- no PUB/PUR spatial holdout, no dMG
# NN-model/trainer/config machinery, unlike the finetuning-family configs.
# Depends on gen_embeddings_camels531_mfformer256.sh having completed first
# (writes the daily-resolution embedding file this reads).

source "${DMG_ENV}/bin/activate"
export PYTHONPATH="$PWD:$PYTHONPATH"

EMB_DIR=${DMG_EMBEDDING_ROOT}/Camels531Extended_MFFormer256
OUT_DIR=${DMG_EMBEDDING_ROOT}/Condensers
EMD_Size=64
python ${DMG_REPO}/scripts/train_condensed_embedding.py \
    --embedding_nc ${EMB_DIR}/Camels531Extended_MFFormer256_embeddings_daily.nc \
    --task_nc ${DMG_DATA_ROOT}/Camels_Pretrain.nc \
    --target_var QObs \
    --condensed_width ${EMD_Size} \
    --recon_weight 0.7 \
    --out ${OUT_DIR}/Camels531_MFFormer256_condenser_daily${EMD_Size}.pt \
    --export_embedding \
    --device cuda
