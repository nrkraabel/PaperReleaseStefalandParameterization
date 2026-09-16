#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32GB
#SBATCH --time=02:00:00
#SBATCH --partition=<PARTITION>
#SBATCH --account=<ACCOUNT>
#SBATCH --job-name=MergeAlphaEarthCaravan3026
#SBATCH --output=%x_%j.out

# CPU-only, no GPU needed. Copies the 4.7GB Caravan3026 task file and appends
# 64 static AlphaEarth vars to the copy via netCDF4 append mode (see
# scripts/merge_alphaearth_caravan3026.py), so peak memory is a few hundred MB
# -- 32GB mirrors convert_caravan3026_task_station_schema.sh's budget. Routed
# through Slurm for the same reason that job was: this cluster's login-node
# cgroup cap OOM-kills multi-GB file work regardless of actual demand.
#
# Prerequisite for scripts/jobs/caravan3026/run_lstm_hbv_alphaearth_pub_*.sh.

source "${DMG_ENV}/bin/activate"
export PYTHONPATH="$PWD:$PYTHONPATH"

python ${DMG_REPO}/scripts/merge_alphaearth_caravan3026.py
