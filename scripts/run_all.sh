#!/bin/bash
# ==============================================================================
# run_all.sh — Run the full NTM model training and evaluation pipeline
# ==============================================================================
#
# Usage:
#   bash run_all.sh /path/to/compound_smiles_stderr_differences.csv
#
# Or with SLURM:
#   sbatch run_all.sh /path/to/compound_smiles_stderr_differences.csv
#
# ==============================================================================
# SLURM directives (uncomment for cluster use)
# ==============================================================================
# #SBATCH --job-name=ntm-pipeline
# #SBATCH --gres=gpu:1
# #SBATCH --cpus-per-task=8
# #SBATCH --mem=64G
# #SBATCH --time=24:00:00
# #SBATCH --output=ntm_pipeline_%j.log
# ==============================================================================

set -euo pipefail

# ---- Configuration ----
INPUT_CSV="${1:?Usage: bash run_all.sh /path/to/input.csv}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="${PROJECT_DIR}/data"
RESULTS_DIR="${PROJECT_DIR}/results"
SAMPLE_SIZE="${SAMPLE_SIZE:-750000}"
SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-256}"
HIDDEN_DIM="${HIDDEN_DIM:-128}"
NUM_LAYERS="${NUM_LAYERS:-4}"

echo "============================================================"
echo "NTM Pipeline"
echo "============================================================"
echo "Input CSV:    ${INPUT_CSV}"
echo "Data dir:     ${DATA_DIR}"
echo "Results dir:  ${RESULTS_DIR}"
echo "Sample size:  ${SAMPLE_SIZE}"
echo "Epochs:       ${EPOCHS}"
echo "Batch size:   ${BATCH_SIZE}"
echo "Hidden dim:   ${HIDDEN_DIM}"
echo "Seed:         ${SEED}"
echo "============================================================"

mkdir -p "${DATA_DIR}" "${RESULTS_DIR}"

# ---- Helper ----
timestamp() { date "+%Y-%m-%d %H:%M:%S"; }

run_step() {
    local step_name="$1"
    shift
    echo ""
    echo "============================================================"
    echo "[$(timestamp)] STARTING: ${step_name}"
    echo "============================================================"
    time "$@"
    echo "[$(timestamp)] DONE: ${step_name}"
}

# ==============================================================================
# Step 0: Preprocess data
# ==============================================================================
if [ -f "${DATA_DIR}/train.csv" ] && [ -f "${DATA_DIR}/val.csv" ] && [ -f "${DATA_DIR}/test.csv" ]; then
    echo ""
    echo "[$(timestamp)] Data splits already exist in ${DATA_DIR}, skipping preprocessing."
    echo "  Delete ${DATA_DIR}/*.csv to re-run preprocessing."
else
    run_step "Data Preprocessing" \
        python "${SCRIPT_DIR}/00_preprocess_data.py" \
            --input "${INPUT_CSV}" \
            --output_dir "${DATA_DIR}" \
            --sample_size "${SAMPLE_SIZE}" \
            --seed "${SEED}"
fi

# ==============================================================================
# Step 1: LOMAP Baseline
# ==============================================================================
run_step "LOMAP Baseline" \
    python "${SCRIPT_DIR}/01_lomap_baseline.py" \
        --data_dir "${DATA_DIR}" \
        --output_dir "${RESULTS_DIR}/lomap" \
        --seed "${SEED}"

# ==============================================================================
# Step 2: MPNN
# ==============================================================================
run_step "MPNN" \
    python "${SCRIPT_DIR}/02_mpnn_model.py" \
        --data_dir "${DATA_DIR}" \
        --output_dir "${RESULTS_DIR}/mpnn" \
        --epochs "${EPOCHS}" \
        --batch_size "${BATCH_SIZE}" \
        --hidden_dim "${HIDDEN_DIM}" \
        --num_layers "${NUM_LAYERS}" \
        --seed "${SEED}"

# ==============================================================================
# Step 3: GAT
# ==============================================================================
run_step "GAT / AttentiveFP" \
    python "${SCRIPT_DIR}/03_gat_model.py" \
        --data_dir "${DATA_DIR}" \
        --output_dir "${RESULTS_DIR}/gat" \
        --epochs "${EPOCHS}" \
        --batch_size "${BATCH_SIZE}" \
        --hidden_dim "${HIDDEN_DIM}" \
        --num_layers "${NUM_LAYERS}" \
        --num_heads 4 \
        --seed "${SEED}"

# ==============================================================================
# Step 4: NTM (the key model)
# ==============================================================================
run_step "NTM (Learned Metric Tensor)" \
    python "${SCRIPT_DIR}/04_ntm_model.py" \
        --data_dir "${DATA_DIR}" \
        --output_dir "${RESULTS_DIR}/ntm" \
        --metric_type full \
        --epochs "${EPOCHS}" \
        --batch_size "${BATCH_SIZE}" \
        --hidden_dim "${HIDDEN_DIM}" \
        --num_layers "${NUM_LAYERS}" \
        --seed "${SEED}"

# ==============================================================================
# Step 5: Transformer
# ==============================================================================
run_step "Molecular Transformer" \
    python "${SCRIPT_DIR}/05_transformer_model.py" \
        --data_dir "${DATA_DIR}" \
        --output_dir "${RESULTS_DIR}/transformer" \
        --epochs "${EPOCHS}" \
        --batch_size 128 \
        --d_model 256 \
        --nhead 8 \
        --num_layers "${NUM_LAYERS}" \
        --seed "${SEED}"

# ==============================================================================
# Step 6: Difficulty Decomposition (requires trained NTM)
# ==============================================================================
run_step "Difficulty Decomposition" \
    python "${SCRIPT_DIR}/06_difficulty_decomposition.py" \
        --data_dir "${DATA_DIR}" \
        --ntm_model_dir "${RESULTS_DIR}/ntm" \
        --output_dir "${RESULTS_DIR}/decomposition" \
        --batch_size "${BATCH_SIZE}" \
        --seed "${SEED}"

# ==============================================================================
# Step 7: Compare all models
# ==============================================================================
run_step "Model Comparison" \
    python "${SCRIPT_DIR}/07_evaluate_and_compare.py" \
        --results_dir "${RESULTS_DIR}" \
        --output_dir "${RESULTS_DIR}/comparison" \
        --data_dir "${DATA_DIR}"

# ==============================================================================
# Summary
# ==============================================================================
echo ""
echo "============================================================"
echo "[$(timestamp)] PIPELINE COMPLETE"
echo "============================================================"
echo ""
echo "Results:"
echo "  LOMAP:          ${RESULTS_DIR}/lomap/results.json"
echo "  MPNN:           ${RESULTS_DIR}/mpnn/results.json"
echo "  GAT:            ${RESULTS_DIR}/gat/results.json"
echo "  NTM:            ${RESULTS_DIR}/ntm/results.json"
echo "  Transformer:    ${RESULTS_DIR}/transformer/results.json"
echo "  Decomposition:  ${RESULTS_DIR}/decomposition/"
echo "  Comparison:     ${RESULTS_DIR}/comparison/"
echo ""
echo "Key outputs:"
echo "  Model comparison table:  ${RESULTS_DIR}/comparison/model_comparison.csv"
echo "  Metric tensor analysis:  ${RESULTS_DIR}/ntm/metric_tensor.npz"
echo "  Difficulty decomp:       ${RESULTS_DIR}/decomposition/decompositions.npz"
echo "  Scatter plots:           ${RESULTS_DIR}/comparison/scatter_plots.png"
echo ""
