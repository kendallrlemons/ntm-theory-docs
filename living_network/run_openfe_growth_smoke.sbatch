#!/usr/bin/env bash
#SBATCH --time=06:00:00
#SBATCH --job-name=openfe_growth_smoke
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --array=1-4
#SBATCH --output=openfe_growth_smoke_%A_%a.out
#SBATCH --error=openfe_growth_smoke_%A_%a.err

set -eo pipefail
source ~/.bashrc
module load cuda/12.0
conda activate openfe_growth
set -u
SETUP=/data1/choderaj/lemonsk/FEC_Sandbox/fah_growth_openfe_smoke_v1
cd "$SETUP"
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1

python -c 'import openmm; names = [openmm.Platform.getPlatform(i).getName() for i in range(openmm.Platform.getNumPlatforms())]; assert "CUDA" in names, names; print(openmm.__version__, names, flush=True)'
task=$(sed -n "${SLURM_ARRAY_TASK_ID}p" smoke_tasks.txt)
test -n "$task"
name=$(basename "$task" .json)
mkdir -p "smoke_results/$name"
openfe quickrun "$task" -o "smoke_results/$name/result.json" -d "smoke_results/$name/work"
