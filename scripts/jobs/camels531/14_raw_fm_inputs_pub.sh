#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=120GB
#SBATCH --gpus=1
#SBATCH --time=96:00:00
#SBATCH --partition=<PARTITION>
#SBATCH --account=<ACCOUNT>
#SBATCH --job-name=Extended1980_1999_RawFmInputs_PUB
#SBATCH --array=0-2
#SBATCH --output=%x_%A_%a.out

# No-foundation-model control: the raw StefaLand input variables (5 ts + 48
# static, no Runoff) go straight into the residual adapter as additional
# inputs. No encoder, no embedding file. 10-fold PUB, 3 seeds.
# Resources match scripts/jobs/camels531/13_nopretrain_hbv_pub.sh, the closest precedent for this
# data path (same NnDualLoader reading the same two NetCDF sources).

source "${DMG_ENV}/bin/activate"
export PYTHONPATH="$PWD:$PYTHONPATH"

SEEDS=(111111 222222 333333)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

OUT_DIR=${DMG_OUTPUT_ROOT}/Extended1980_1999_Camels531/RawFmInputs/PUB/seed${SEED}/

python ${DMG_REPO}/src/dmg/__main__.py \
    --config-name camels531/RawFmInputsPUB \
    seed=${SEED} \
    save_path=$OUT_DIR \
    +out_path=$OUT_DIR \
    +run_dir=$OUT_DIR \
    +eval_output_key=[streamflow,recharge,percolation,SM,parFC]
