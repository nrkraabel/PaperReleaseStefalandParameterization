#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128GB
#SBATCH --gpus=1
#SBATCH --time=08:00:00
#SBATCH --partition=<PARTITION>
#SBATCH --account=<ACCOUNT>
#SBATCH --job-name=TrainCondenser
#SBATCH --output=%x_%j.out

# Stage 1: train a linear condenser (embed_dim -> WIDTH).
#
# The whole embedding array is held in memory; the script prints its size
# before reading. Size --mem to at least that plus headroom.

source <VENV>/bin/activate
cd <REPO_ROOT>

EMB_NC=<EMBEDDING_ROOT>/<MyDataset>/<MyDataset>_embeddings_daily.nc
OUT_DIR=<EMBEDDING_ROOT>/condensers
WIDTH=64

python scripts/train_condensed_embedding.py \
    --embedding_nc ${EMB_NC} \
    --condensed_width ${WIDTH} \
    --out ${OUT_DIR}/<MyDataset>_condenser_daily${WIDTH}.pt \
    --export_embedding \
    --device cuda
