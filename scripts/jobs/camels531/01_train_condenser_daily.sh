#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48GB
#SBATCH --gpus=1
#SBATCH --time=06:00:00
#SBATCH --partition=<PARTITION>
#SBATCH --account=<ACCOUNT>
#SBATCH --job-name=TrainCondenser_Camels531_daily
#SBATCH --output=%x_%j.out

# Needs 00_gen_embeddings_stefaland256.sh to have finished.

source "${DMG_ENV}/bin/activate"
export PYTHONPATH="$PWD:$PYTHONPATH"

EMB_DIR=${DMG_EMBEDDING_ROOT}/Camels531Extended_StefaLand256
OUT_DIR=${DMG_EMBEDDING_ROOT}/Condensers
WIDTH=64

python ${DMG_REPO}/scripts/train_condensed_embedding.py \
    --embedding_nc ${EMB_DIR}/Camels531Extended_StefaLand256_embeddings_daily.nc \
    --condensed_width ${WIDTH} \
    --out ${OUT_DIR}/Camels531_StefaLand256_condenser_daily${WIDTH}.pt \
    --export_embedding \
    --device cuda
