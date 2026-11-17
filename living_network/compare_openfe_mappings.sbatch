#!/usr/bin/env bash
#SBATCH --time=02:00:00
#SBATCH --job-name=compare_openfe_mappings
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=compare_openfe_mappings_%j.out
#SBATCH --error=compare_openfe_mappings_%j.err

set -eo pipefail
source ~/.bashrc
conda activate openfe_growth
set -u
cd /data1/choderaj/lemonsk/FEC_Sandbox
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
python compare_openfe_mappings.py \
  --bundle_dir /data1/choderaj/lemonsk/FEC_Sandbox/fah_growth_structures_v1/bundle \
  --output_dir /data1/choderaj/lemonsk/FEC_Sandbox/fah_growth_mapping_comparison_v1
