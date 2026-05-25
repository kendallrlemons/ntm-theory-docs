# Living Network — Hyperspherical RBFE Difficulty Model

All-in-one script implementing Phases 1–5 of the experimental plan.

## Usage

```bash
python hypersphere_rbfe.py \
    --input /path/to/compound_smiles_stderr_differences.csv \
    --output_dir ./results \
    --sample_size 500000 \
    --epochs 60 \
    --batch_size 256 \
    --seed 42
```

All 5 phases run by default. Use `--phases` to run a subset.

### Run specific phases only

```bash
# Phase 1 only (data characterization — go/no-go gate)
python hypersphere_rbfe.py --input <csv> --output_dir ./results --phases 1

# Skip Phase 1 & 2, train encoder + validate
python hypersphere_rbfe.py --input <csv> --output_dir ./results --phases 3,4,5
```

Later phases automatically reload artifacts written by earlier phases (via
`./results/phase1/cleaned_split.parquet` and `./results/phase3_*/best.pt`).

### Flat Euclidean ablation

```bash
python hypersphere_rbfe.py ... --flat_ablation
```

Trains a second model without L2 normalization and without dispersion loss,
for comparison in Phase 4.

## Phase Output Structure

```
results/
├── config.json
├── phase1/
│   ├── cleaned_split.pkl              # train/val/test splits (pickle, no pyarrow needed)
│   ├── se_distribution.png
│   ├── pair_descriptors_sample.csv
│   ├── descriptor_se_correlations.csv
│   ├── descriptors_vs_se.png
│   ├── spearman_correlations.html     # interactive Plotly bar chart
│   ├── descriptor_scatter_interactive.html  # scatter grid w/ SMILES tooltips
│   └── go_no_go_gate.json             # pass/fail + counts
├── phase3_hypersphere/
│   ├── best.pt                        # best checkpoint (encoder + head + clusters)
│   └── history.json
├── phase3_flat_ablation/              # (only if --flat_ablation)
│   └── best.pt
├── phase4_hypersphere/
│   ├── correlations.json              # Pearson/Spearman for dist, OOD, pred
│   ├── scores.npz                     # raw embeddings + scores
│   ├── embedding_visualization.png    # 2D PCA colored by OOD and |SE|
│   ├── score_vs_se_scatter.png
│   ├── sphere_3d_interactive.html     # rotatable 3D hypersphere (Plotly)
│   └── sphere_stereographic.png       # static stereographic projection
├── phase4_flat_ablation/              # (only if --flat_ablation)
└── phase5/
    ├── baseline_comparison.csv        # Tanimoto / LOMAP-like / NTM OOD / NTM reg
    └── baseline_comparison.png
```

## Key Hyperparameters

| Flag | Default | What it controls |
|---|---|---|
| `--hidden_dim` | 128 | GNN hidden dim |
| `--num_layers` | 4 | Number of message-passing layers |
| `--embed_dim` | 128 | Hypersphere embedding dimension |
| `--num_clusters` | 8 | K for dispersion loss |
| `--w_contrastive` | 1.0 | Contrastive loss weight |
| `--w_dispersion` | 0.5 | Dispersion loss weight |
| `--w_regression` | 1.0 | Regression head loss weight |
| `--cluster_ema` | 0.9 | EMA decay for cluster centers |

## Loss Design (Option C — Hybrid)

1. **Continuous contrastive loss** — no threshold; target cosine similarity is
   `cos(π · |SE|/SE_q99)`. SE=0 maps to identical embeddings (cos=1); SE=q99
   maps to antipodal embeddings (cos=-1).

2. **Dispersion loss** — minimizes mean off-diagonal cosine similarity between
   `K` cluster centers, maintained via EMA over batch-wise K-means assignments.
   Initialized with scikit-learn K-means on first warmup batch.

3. **Regression head** — small MLP taking `[|h_A-h_B|, h_A*h_B, cos_sim]` →
   predicts `|SE|` directly. Trained with MSE.

Total: `L = w_c · L_contrast + w_d · L_dispersion + w_r · L_regression`.

## Go/No-Go Gate (Phase 1)

Phase 1 writes `phase1/go_no_go_gate.json`:

```json
{
  "n_descriptors_checked": 10,
  "n_consistent_with_intuition": 6,
  "gate_threshold": 2,
  "passed": true
}
```

Expected signs:
- `delta_*` descriptors → positive correlation with |SE|
- `tanimoto_morgan`, `mcs_coverage` → negative correlation

If fewer than 2 descriptors show the expected sign, the script prints a
warning and proceeds anyway (you may want to investigate the data).

## Cluster-Friendly Notes

- Auto-detects CUDA; falls back to CPU.
- All long-running steps print progress and ETA.
- Artifacts are checkpointed per phase; you can resume by re-running with a
  subset of `--phases`.
- `--num_workers` controls DataLoader parallelism.
