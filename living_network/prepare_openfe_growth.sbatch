#!/usr/bin/env bash
#SBATCH --time=04:00:00
#SBATCH --job-name=prepare_openfe_growth
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=prepare_openfe_growth_%j.out
#SBATCH --error=prepare_openfe_growth_%j.err

set -eo pipefail
source ~/.bashrc
conda activate openfe_growth
set -u
cd /data1/choderaj/lemonsk/FEC_Sandbox
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
python prepare_openfe_growth.py \
  --bundle_dir /data1/choderaj/lemonsk/FEC_Sandbox/fah_growth_structures_v1/bundle \
  --output_dir /data1/choderaj/lemonsk/FEC_Sandbox/fah_growth_openfe_smoke_v1 \
  --kind smoke
