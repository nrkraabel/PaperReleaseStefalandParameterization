#!/bin/bash
#SBATCH --job-name=mswep_merge
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2

# Copies the 4.7 GB source file and appends total_precipitation_sum_MSWEP plus
# time_offset_days to the copy. Submit only after the whole 02 array is done --
# a missing year silently becomes a NaN block rather than an error.
#
# 32G covers the three (3026 x 26662) float32 arrays held at once: the assembled
# MSWEP grid, the aligned output, and the ERA5-Land series read back for the
# validation summary (~320 MB each), plus the per-basin correlation loop. The
# wall clock is dominated by the file copy and the final compressed write.
#
# Pass --inplace instead to append to the master file rather than copying it.

module purge
module load anaconda3
source ${DMG_DATA_ROOT}/envs/WTD/bin/activate
export PYTHONUNBUFFERED=1

cd ${DMG_DATA_ROOT}/caravan_zenodo/mswep_forcing
mkdir -p logs

python 03_merge_into_caravan.py
