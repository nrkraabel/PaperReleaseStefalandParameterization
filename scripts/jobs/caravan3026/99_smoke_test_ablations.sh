#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16GB
#SBATCH --gpus=1
#SBATCH --time=00:15:00
#SBATCH --partition=<PARTITION>
#SBATCH --account=<ACCOUNT>
#SBATCH --job-name=SmokeTest_EmbeddingAblations
#SBATCH --output=%x_%j.out

# Shape/wiring check for the three EmbeddingFinetuneing ablation branches.
# Needs a GPU only because CudnnLstm.__init__ calls self.cuda(); it touches
# no data files and runs in seconds. Run this BEFORE submitting the 48h
# ablation arrays.

source "${DMG_ENV}/bin/activate"
export PYTHONPATH="${DMG_REPO}:$PYTHONPATH"

python ${DMG_REPO}/scripts/smoke_test_embedding_ablations.py
