#!/bin/bash
#SBATCH --job-name=mswep_extract
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --array=1979-2023
#SBATCH --time=02:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1

# One task per calendar year of true MSWEP dates. The union of what the stations
# need is 1979-01-01 (start of the MSWEP record) through 2023-12-31 (the last day
# camelsgb's true axis reaches), so the array spans 1979-2023.
#
# I/O bound, not compute bound: each day is a ~5 MB compressed global field of
# which only rows 314-1443 are read, then a 381k-nonzero sparse mat-vec that
# costs microseconds. Measured ~0.13 s/day, so a 365-day task lands near a
# minute; the 2 h wall clock is slack for a contended filesystem.
#
# 8G is far more than needed -- the working set is one 1130x3600 float32 band
# (~16 MB) plus the weight matrix. Years are checkpointed and skipped if their
# chunk already exists, so resubmitting the array is safe and cheap.

module purge
module load anaconda3
source ${DMG_DATA_ROOT}/envs/WTD/bin/activate
export PYTHONUNBUFFERED=1

cd ${DMG_DATA_ROOT}/caravan_zenodo/mswep_forcing
mkdir -p logs chunks

python 02_extract_mswep.py --year "${SLURM_ARRAY_TASK_ID}"
