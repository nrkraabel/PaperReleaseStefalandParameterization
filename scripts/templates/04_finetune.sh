#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128GB
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --partition=<PARTITION>
#SBATCH --account=<ACCOUNT>
#SBATCH --job-name=EmbeddingFinetune
#SBATCH --output=%x_%j.out

# Stage 3: fine-tune on the (condensed) embeddings. A GPU is required -- the
# LSTM decoder calls .cuda() unconditionally.
# CONFIG is a path under conf/ without the .yaml extension. Copy a template
# to your own name first rather than editing the template in place.

source <VENV>/bin/activate
cd <REPO_ROOT>

CONFIG=templates/embedding_finetune_temporal

python src/dmg/__main__.py --config-name ${CONFIG}
