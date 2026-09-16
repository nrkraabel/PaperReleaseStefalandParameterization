#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32GB
#SBATCH --time=02:00:00
#SBATCH --partition=<PARTITION>
#SBATCH --account=<ACCOUNT>
#SBATCH --job-name=ConvertCaravan3026StationIds
#SBATCH --output=%x_%j.out

# CPU-only, no GPU needed -- streams station-chunks via plain netCDF4 (see
# scripts/convert_caravan3026_task_station_schema.py), so 32GB is generous
# headroom for a 5.2GB source file. The direct login-node run OOM'd (exit
# 137) against this cluster's login-node cgroup memory cap, not an actual
# memory shortage -- hence routing this through Slurm instead.

source "${DMG_ENV}/bin/activate"
export PYTHONPATH="$PWD:$PYTHONPATH"

python ${DMG_REPO}/scripts/convert_caravan3026_task_station_schema.py
