#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48GB
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --partition=<PARTITION>
#SBATCH --account=<ACCOUNT>
#SBATCH --job-name=GenEmbeddings_Camels531Extended_MFFormer256
#SBATCH --output=%x_%j.out

source "${DMG_ENV}/bin/activate"
export PYTHONPATH="$PWD:$PYTHONPATH"

# Foundation model: MfformerGlobal20.pt (d_model=256). generate_embeddings.py's
# build_pretrained_encoder() now dispatches on pretrained_type: 'mfformer'
# builds MFFormerDecLSTM (src/dmg/models/neural_networks/transformer/
# MFFormerDecLSTM.py), the TransformerBackbone-encoder + single-layer-LSTM-
# decoder architecture this checkpoint actually contains -- confirmed via an
# exact 485/485 name+shape parameter match, ported from
# 30.MFFormer/MFFormer/models/MFFormer_dec_LSTM.py. Previously this built a
# StefaLandPatchTFT regardless of pretrained_type (a completely different
# tokenizer/depatcher architecture the checkpoint's state_dict doesn't match),
# loading only ~56% of parameters by chance name/shape overlap.
#
# --config points at conf/camels531/_MFFormerArchTemplate.yaml,
# not conf/HBVGlobalMFFormerRechargeSMSpatialTest.yaml: that config's
# pretrained_time_series_vars/pretrained_static_vars also don't match what
# this checkpoint was actually pretrained on (it lists 6 ts vars incl. Runoff
# and 22 static vars; the checkpoint's own per-variable embedding submodule
# names show the true set is 5 ts vars, no Runoff, and 48 static vars -- see
# _MFFormerArchTemplate.yaml's header comment). That mismatch is orthogonal
# to the architecture bug and likely affects any live training run using that
# config too, but it's this session's WIP file for a different (Global SM/
# recharge) task, so it's left alone here rather than edited in place.
#
# --window_days/--context_days: MFFormerDecLSTM uses a plain per-timestep
# PositionalEncoding with a fixed 1000-row learned position table (no wrap
# -- unlike StefaLandPatchTFT's TFTPositionalEncoding, which explicitly does
# `indices % max_len` to fold long sequences back into its own 1000-row
# table). generate_embeddings.py's defaults (window_days=3650,
# context_days=365) feed sequences ~4015 steps long, which is a hard out-of-
# bounds index into that table (CUDA "gather kernel index out of bounds"),
# not just slow/wasteful -- crashes here rather than silently degrading like
# the modulo-wrap path would. window_days+context_days+1 must stay <= 1000;
# 500+300+1=801 leaves headroom.
#
# --overwrite_norm_stats: out_root already has normalization_statistics_
# pretrained.json from the earlier broken run (22 static vars). load_norm_stats
# reuses that file wholesale whenever it exists (no check that it covers the
# current variable list), so without this flag the 26 static vars newly added
# by the corrected 48-var list would silently go unnormalized ("No
# normalization stats for X, skipping") instead of erroring -- forcing a
# fresh, complete stats file for the current (correct) variable list instead.
python ${DMG_REPO}/scripts/generate_embeddings.py \
    --out_dir ${DMG_EMBEDDING_ROOT} \
    --config ${DMG_REPO}/conf/camels531/_MFFormerArchTemplate.yaml \
    --pretrain_data ${DMG_DATA_ROOT}/CamelsPretrain/Camels_refpoints_1979_2014.nc \
    --dataset_name Camels531Extended_MFFormer256 \
    --resolutions daily monthly seasonal annual \
    --window_days 500 \
    --context_days 300 \
    --overwrite_norm_stats \
    --device cuda
