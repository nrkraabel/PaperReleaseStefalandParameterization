#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=400GB
#SBATCH --gpus=1
#SBATCH --time=08:00:00
#SBATCH --partition=<PARTITION>
#SBATCH --account=<ACCOUNT>
#SBATCH --job-name=TrainCondenser_Caravan3026_daily
#SBATCH --output=%x_%j.out

# Needs 00_gen_embeddings_stefaland256.sh to have finished. 400GB because the
# whole daily embedding array for 3026 stations is held in memory.

source "${DMG_ENV}/bin/activate"
export PYTHONPATH="$PWD:$PYTHONPATH"

EMB_DIR=${DMG_EMBEDDING_ROOT}/Caravan3026_StefaLand256
OUT_DIR=${DMG_EMBEDDING_ROOT}/Condensers
WIDTH=64

python ${DMG_REPO}/scripts/train_condensed_embedding.py \
    --embedding_nc ${EMB_DIR}/Caravan3026_StefaLand256_embeddings_daily.nc \
    --condensed_width ${WIDTH} \
    --out ${OUT_DIR}/Caravan3026_StefaLand256_condenser_daily${WIDTH}.pt \
    --export_embedding \
    --device cuda
