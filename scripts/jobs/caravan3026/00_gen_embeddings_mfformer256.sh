#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --partition=<PARTITION>
#SBATCH --account=<ACCOUNT>
#SBATCH --job-name=GenEmbeddings_Caravan3026_MFFormer256
#SBATCH --output=%x_%j.out
# NOTE: original run used --dependency on the upstream stage's job id.
# Foundation model: MfformerGlobal20.pt (d_model=256, pretrained_type=mfformer
# -> MFFormerDecLSTM architecture -- see direct_finetuneing.py's
# build_pretrained_encoder). Uses conf/camels531/_MFFormerArchTemplate.yaml
# for the architecture/checkpoint/pretrained-variable fields, same as
# gen_embeddings_global3434_mfformer256.sh / gen_embeddings_smgwr4477_mfformer256.sh.
#
# --pretrain_data is the 3026-gauge Caravan reference-point file (gages_list_
# caravan3026.txt's station set), built by job 54605032 (CombineNc1992_2018)
# which was still running when this script was queued -- hence
# --dependency=afterok:54605032 so this won't start until that file exists.
# NOT Carvan_refpoints.nc (the older 16299-station file in the same
# directory) -- that one predates the 3026-station filtering and is the
# wrong/stale file for this run. Var presence against the checkpoint's 5 ts
# vars + 48 static vars hasn't been re-verified against this new file (it
# didn't exist yet) but should match, since it's built by the same pipeline
# as Global3434_refpoints.nc / GlobalSM_GWR_refpoints.nc, which do match.
#
# --window_days/--context_days: MFFormerDecLSTM's plain per-timestep
# PositionalEncoding has a fixed 1000-row table (no modulo wrap, unlike
# StefaLandPatchTFT) -- window_days+context_days+1 must stay <= 1000.

source "${DMG_ENV}/bin/activate"
export PYTHONPATH="$PWD:$PYTHONPATH"

python ${DMG_REPO}/scripts/generate_embeddings.py \
    --out_dir ${DMG_EMBEDDING_ROOT} \
    --config ${DMG_REPO}/conf/camels531/_MFFormerArchTemplate.yaml \
    --pretrain_data ${DMG_DATA_ROOT}/CarvanPretrain/3026Carvan_refpoints.nc \
    --dataset_name Caravan3026_MFFormer256 \
    --resolutions daily monthly seasonal annual \
    --window_days 500 \
    --context_days 300 \
    --device cuda
