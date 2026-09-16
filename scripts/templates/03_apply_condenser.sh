#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --partition=<PARTITION>
#SBATCH --account=<ACCOUNT>
#SBATCH --job-name=ApplyCondenser
#SBATCH --output=%x_%j.out

# Stage 2: apply an already-trained condenser (from either mode) to an
# embedding file -- e.g. a larger or ungauged station set with the same
# embed_dim. No streamflow or task file needed. Reads stations in batches,
# so memory is bounded by the output array, not the input.

source <VENV>/bin/activate
cd <REPO_ROOT>

python scripts/apply_condenser.py \
    --condenser <EMBEDDING_ROOT>/condensers/<MyDataset>_condenser_daily64.pt \
    --embedding_nc <EMBEDDING_ROOT>/<OtherDataset>/<OtherDataset>_embeddings_daily.nc \
    --out <EMBEDDING_ROOT>/<OtherDataset>/<OtherDataset>_condensed_daily64.nc \
    --device cuda
