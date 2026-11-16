#!/usr/bin/env bash
#SBATCH --time=02:00:00
#SBATCH --job-name=audit_pilot_work
#SBATCH --cpus-per-task=1
#SBATCH --output=audit_pilot_work_%j.out
#SBATCH --error=audit_pilot_work_%j.err

set -eo pipefail
source ~/.bashrc
module load cuda/12.0
conda activate gnn_rbfe
set -u
cd /data1/choderaj/lemonsk/FEC_Sandbox
export PYTHONUNBUFFERED=1

python audit_fah_pilot_work.py \
  --analysis /data1/choderaj/lemonsk/FEC_Sandbox/moonshot/dashboards/sprint-4/2020-09-06-ugi-tBu-x3110-3v3m-2020-04-Jacobs/analysis.json \
  --specification fah_low_overlap_audit_cases.json \
  --output_dir /data1/choderaj/lemonsk/FEC_Sandbox/fah_pilot_work_audit_v1
