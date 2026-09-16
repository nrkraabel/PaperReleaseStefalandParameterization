#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48GB
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --partition=<PARTITION>
#SBATCH --account=<ACCOUNT>
#SBATCH --job-name=GenEmbeddings_Camels531Extended_StefaLand256
#SBATCH --output=%x_%j.out

source "${DMG_ENV}/bin/activate"
export PYTHONPATH="$PWD:$PYTHONPATH"

# --overwrite_norm_stats: out_root already has normalization_statistics_
# pretrained.json from the earlier broken run (22 static vars). load_norm_stats
# reuses that file wholesale whenever it exists (no check that it covers the
# current variable list), so without this flag the 26 static vars newly added
# by the corrected 48-var list would silently go unnormalized ("No
# normalization stats for X, skipping") instead of erroring -- forcing a
# fresh, complete stats file for the current (correct) variable list instead.
python ${DMG_REPO}/scripts/generate_embeddings.py \
    --config ${DMG_REPO}/conf/camels531/_StefaLandArchTemplate.yaml \
    --pretrain_data ${DMG_DATA_ROOT}/CamelsPretrain/Camels_refpoints_1979_2014.nc \
    --dataset_name Camels531Extended_StefaLand256 \
    --resolutions daily monthly seasonal annual \
    --window_days 500 \
    --context_days 300 \
    --overwrite_norm_stats \
    --device cuda
