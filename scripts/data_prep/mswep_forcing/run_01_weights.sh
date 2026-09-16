#!/bin/bash
#SBATCH --job-name=mswep_weights
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1

# Steps 00 and 01 both already ran successfully; this script exists so the whole
# chain is reproducible from scratch. Step 00 takes ~10 min (3026 small file
# opens, latency bound); step 01 takes ~1 min (774k prepared-geometry cell tests,
# of which only the basin-boundary cells need a real intersection).

module purge
module load anaconda3
source ${DMG_DATA_ROOT}/envs/WTD/bin/activate
export PYTHONUNBUFFERED=1

cd ${DMG_DATA_ROOT}/caravan_zenodo/mswep_forcing
mkdir -p logs chunks

python 00_station_time_axis.py
python 01_build_weights.py
