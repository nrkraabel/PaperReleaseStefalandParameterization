#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --gpus=1
#SBATCH --time=08:00:00
#SBATCH --partition=<PARTITION>
#SBATCH --account=<ACCOUNT>
#SBATCH --job-name=GenerateEmbeddings
#SBATCH --output=%x_%j.out

# Stage 0 (optional): run a frozen foundation-model encoder over your inputs.
# Skip this stage if you already have an embedding NetCDF.
#
# Memory scales with stations x days x embed_dim; 'daily' output is by far
# the largest -- request only the resolutions you need.

source <VENV>/bin/activate
cd <REPO_ROOT>

python scripts/generate_embeddings.py \
    --out_dir ${DMG_EMBEDDING_ROOT} \
    --config conf/templates/_encoder_arch_template.yaml \
    --out_dir <EMBEDDING_ROOT> \
    --dataset_name <MyDataset> \
    --resolutions daily monthly \
    --device cuda
