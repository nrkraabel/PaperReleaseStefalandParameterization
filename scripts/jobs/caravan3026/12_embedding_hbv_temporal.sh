#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=400GB
#SBATCH --gpus=1
#SBATCH --time=72:00:00
#SBATCH --partition=<PARTITION>
#SBATCH --account=<ACCOUNT>
#SBATCH --job-name=Caravan3026_EmbeddingDaily_Temporal
#SBATCH --output=%x_%j.out

# Full-range temporal test: ALL 3026 basins (no PUB spatial holdout), train
# and test both over 1997-2018, 100 epochs. 400GB/72h -- daily condensed
# embedding file is 15.3GB and this doubles the PUB variant's epoch count,
# so keeping the same generous memory budget plus more time.
#
# Depends on: scripts/data_prep/convert_caravan3026_task_station_schema.sh.

source "${DMG_ENV}/bin/activate"
export PYTHONPATH="$PWD:$PYTHONPATH"

OUT_DIR=${DMG_OUTPUT_ROOT}/Caravan3026/Embedding/daily/temporal_1997-2012/

python ${DMG_REPO}/src/dmg/__main__.py \
    --config-name Caravan3026/EmbeddingDailyTemporal \
    save_path=$OUT_DIR \
    +out_path=$OUT_DIR \
    +run_dir=$OUT_DIR \
    +eval_output_key=[streamflow]
