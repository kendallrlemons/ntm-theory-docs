# NTM Model Training & Evaluation Scripts

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Preprocess data
python 00_preprocess_data.py \
    --input /path/to/compound_smiles_stderr_differences.csv \
    --output_dir ../data \
    --sample_size 750000

# 3. Run models (can be run in parallel on cluster)
python 01_lomap_baseline.py     --data_dir ../data --output_dir ../results/lomap
python 02_mpnn_model.py         --data_dir ../data --output_dir ../results/mpnn
python 03_gat_model.py          --data_dir ../data --output_dir ../results/gat
python 04_ntm_model.py          --data_dir ../data --output_dir ../results/ntm --metric_type full
python 05_transformer_model.py  --data_dir ../data --output_dir ../results/transformer

# 4. Difficulty decomposition (requires trained NTM model)
python 06_difficulty_decomposition.py \
    --data_dir ../data \
    --ntm_model_dir ../results/ntm \
    --output_dir ../results/decomposition

# 5. Compare all models
python 07_evaluate_and_compare.py \
    --results_dir ../results \
    --output_dir ../results/comparison
```

## Script Overview

| Script | Purpose | GPU? | Time Est. |
|--------|---------|------|-----------|
| `00_preprocess_data.py` | Dedup, subsample, train/val/test split | No | ~10 min |
| `01_lomap_baseline.py` | LOMAP features → Ridge/RF/GBM regression | No | ~30 min |
| `02_mpnn_model.py` | Message Passing Neural Network | Yes | ~2-4 hrs |
| `03_gat_model.py` | Graph Attention Network | Yes | ~3-5 hrs |
| `04_ntm_model.py` | NTM with learned metric tensor | Yes | ~3-5 hrs |
| `05_transformer_model.py` | SMILES-based Transformer | Yes | ~4-6 hrs |
| `06_difficulty_decomposition.py` | Eigenstructure analysis of NTM metric | Yes | ~30 min |
| `07_evaluate_and_compare.py` | Compare all models, generate plots | No | ~5 min |

## File Dependencies

```
shared_utils.py          ← Shared featurization, dataset, collation, MPNNLayer
    ├── 02_mpnn_model.py
    ├── 03_gat_model.py
    ├── 04_ntm_model.py
    └── 06_difficulty_decomposition.py (also imports NTMModel from 04)
```

- `01_lomap_baseline.py` and `05_transformer_model.py` are self-contained
- `07_evaluate_and_compare.py` only reads saved results (JSON + NPZ files)

## Directory Structure (after running)

```
data/
├── train.csv
├── val.csv
└── test.csv

results/
├── lomap/
│   ├── results.json
│   ├── gbm_test_preds.npz
│   └── *.pkl (trained models)
├── mpnn/
│   ├── results.json
│   ├── best_model.pt
│   └── test_preds.npz
├── gat/
│   └── (same structure)
├── ntm/
│   ├── results.json
│   ├── best_model.pt
│   ├── test_preds.npz
│   ├── test_embeddings.npz    ← Latent embeddings for analysis
│   └── metric_tensor.npz      ← Learned M with eigendecomposition
├── transformer/
│   └── (same structure)
├── decomposition/
│   ├── decompositions.npz
│   ├── eigenvalue_spectrum.png
│   ├── transformation_clusters.png
│   ├── difficulty_summary.png
│   ├── cluster_stats.json
│   └── atom_attributions.json
└── comparison/
    ├── model_comparison.csv
    ├── model_comparison.tex
    ├── scatter_plots.png
    ├── error_distributions.png
    ├── stratified_performance.png
    ├── learning_curves.png
    └── ntm_distance_analysis.png
```

## Key Hyperparameters

All scripts accept `--help` for full argument list. Key defaults:

| Parameter | MPNN | GAT | NTM | Transformer |
|-----------|------|-----|-----|-------------|
| `hidden_dim` | 128 | 128 | 128 | 256 |
| `num_layers` | 4 | 4 | 4 | 4 |
| `batch_size` | 256 | 256 | 256 | 128 |
| `lr` | 1e-3 | 1e-3 | 1e-3 | 5e-4 |
| `epochs` | 100 | 100 | 100 | 100 |
| `patience` | 15 | 15 | 15 | 15 |

## SLURM Example

```bash
#!/bin/bash
#SBATCH --job-name=ntm-train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=8:00:00

module load cuda/12.0 python/3.10

source /path/to/venv/bin/activate
cd /path/to/ntm-theory-docs/scripts

python 04_ntm_model.py \
    --data_dir ../data \
    --output_dir ../results/ntm \
    --metric_type full \
    --hidden_dim 128 \
    --num_layers 4 \
    --epochs 100 \
    --batch_size 256
```
