#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=400GB
#SBATCH --gpus=1
#SBATCH --time=08:00:00
#SBATCH --partition=<PARTITION>
#SBATCH --account=<ACCOUNT>
#SBATCH --job-name=TrainCondenser_Caravan3026_daily_MFFormer256
#SBATCH --output=%x_%j.out

# Standalone dual-loss condenser (scripts/train_condensed_embedding.py):
# jointly trains a 64-dim condenser against reconstruction + streamflow
# loss, using the full 3026-station Caravan set. --task_nc is
# Caravan_camels_only_singlefile_dropHighNaN70.nc (despite the "camels_only"
# name, it's the 3026-station task file -- station dim/count and
# camels_-prefixed station_id values match this job's embedding file
# exactly, confirmed directly against 3026Carvan_refpoints.nc's own
# station_ids coord). Target var is 'streamflow' here, not 'Runoff' --
# 3026Carvan_refpoints.nc's own Runoff variable is unusable (100% NaN), which
# is why a separate task_nc is needed at all (same daily/embedding vs.
# task_nc split as Global3434, different reason).
# Depends on scripts/jobs/caravan3026/00_gen_embeddings_mfformer256.sh (job 54605069) having
# completed first -- hence --dependency=afterok:54605069.

source "${DMG_ENV}/bin/activate"
export PYTHONPATH="$PWD:$PYTHONPATH"

EMB_DIR=${DMG_EMBEDDING_ROOT}/Caravan3026_MFFormer256
TASK_NC=${DMG_DATA_ROOT}/caravan_zenodo/Caravan_camels_only_singlefile_dropHighNaN70.nc
OUT_DIR=${DMG_EMBEDDING_ROOT}/Condensers
EMD_Size=128
python ${DMG_REPO}/scripts/train_condensed_embedding.py \
    --embedding_nc ${EMB_DIR}/Caravan3026_MFFormer256_embeddings_daily.nc \
    --task_nc ${TASK_NC} \
    --target_var streamflow \
    --condensed_width ${EMD_Size} \
    --recon_weight 0.4 \
    --out ${OUT_DIR}/Caravan3026_MFFormer256_condenser_daily${EMD_Size}testingWeights.pt \
    --export_embedding \
    --device cuda
