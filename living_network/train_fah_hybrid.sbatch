#!/usr/bin/env bash
#SBATCH --time=144:00:00
#SBATCH --job-name=train_hybrid
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --output=train_hybrid_%j.out
#SBATCH --error=train_hybrid_%j.err

set -eo pipefail
source ~/.bashrc
module load cuda/12.0
conda activate gnn_rbfe
set -u

cd /data1/choderaj/lemonsk/FEC_Sandbox
export FAH_ROOT=/data1/choderaj/lemonsk/FEC_Sandbox
export FAH_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export FAH_DEVICE=cuda
export FAH_PHASE=all
export PYTHONUNBUFFERED=1

python -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable in this allocation/environment"; print("GPU:", torch.cuda.get_device_name(0), flush=True)'
bash run_fah_gnn_suite.sh hybrid
