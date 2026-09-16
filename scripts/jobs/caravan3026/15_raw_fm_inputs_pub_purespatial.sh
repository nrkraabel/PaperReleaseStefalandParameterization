#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=250GB
#SBATCH --gpus=1
#SBATCH --time=96:00:00
#SBATCH --partition=<PARTITION>
#SBATCH --account=<ACCOUNT>
#SBATCH --job-name=Caravan3026_RawFmInputs_PureSpatial
#SBATCH --array=0-2
#SBATCH --output=%x_%A_%a.out

# Control for the daily-embedding ablation set: no foundation model. The raw
# StefaLand input variables (5 ts + 48 static from 3026Carvan_refpoints.nc) go
# straight into the residual adapter as additional inputs; no encoder, no
# embedding file.
# Resources follow scripts/jobs/caravan3026/14_nopretrain_hbv_pub_purespatial.sh rather than the
# daily-embedding jobs: this config reads the same two NetCDF sources through
# the same NnDualLoader, and never opens the 41GB daily embedding file that
# drove the 400GB budget on those. 96h for the same reason -- that job pairing
# is the closest precedent for this data path.

source "${DMG_ENV}/bin/activate"
export PYTHONPATH="$PWD:$PYTHONPATH"

SEEDS=(111111 222222 333333)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

OUT_DIR=${DMG_OUTPUT_ROOT}/Caravan3026/RawFmInputs/pure_spatial_1998-2008/seed${SEED}/

python ${DMG_REPO}/src/dmg/__main__.py \
    --config-name Caravan3026/RawFmInputsPUB_PureSpatial \
    seed=${SEED} \
    save_path=$OUT_DIR \
    +out_path=$OUT_DIR \
    +run_dir=$OUT_DIR \
    +eval_output_key=[streamflow]
