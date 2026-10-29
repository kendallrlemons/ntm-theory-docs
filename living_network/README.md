# Living Network — Hyperspherical RBFE Difficulty Model

All-in-one script implementing Phases 1–5 of the experimental plan.

## Verified RBFE rerun

Use fresh output directories for each run. Data-cache manifests bind cleaned
data to the input file and cleaning/split settings; graph bundles bind graphs
to ordered SMILES pairs and the MCS timeout; checkpoints bind weights to the
ordered pairs, signed labels, and split assignments. Legacy pickle caches,
list-only graph caches, and checkpoints without matching provenance are
rejected rather than migrated. `benchmark.json` records each pipeline run's
benchmark identity. These checks establish artifact alignment, not model
configuration compatibility.

Run these commands from `living_network`. This example uses the full input
dataset and fixed split settings:

```bash
CSV=/data1/choderaj/lemonsk/FEC_Sandbox/compound_smiles_stderr_differences.csv

python mcs_precompute.py \
    --input "$CSV" \
    --output_dir ./graphs_hybrid_topo_v3 \
    --sample_size 0 \
    --seed 42 \
    --mcs_timeout 5 \
    --num_workers 8

# Supervised hybrid-topology reference
python hybrid_topo_rbfe.py \
    --input "$CSV" \
    --output_dir ./results_hybrid_topo_v3 \
    --graph_cache_dir ./graphs_hybrid_topo_v3 \
    --split_method scaffold \
    --seed 42 \
    --sample_size 0 \
    --val_frac 0.1 \
    --test_frac 0.1 \
    --mcs_timeout 5 \
    --phases 1,3,4

# Independent hypersphere model
python hypersphere_rbfe.py \
    --input "$CSV" \
    --output_dir ./results_hypersphere_v3 \
    --split_method scaffold \
    --seed 42 \
    --sample_size 0 \
    --val_frac 0.1 \
    --test_frac 0.1 \
    --phases 1,3,4

# Combined hybrid + hypersphere model and comparison
python hybrid_hypersphere_rbfe.py \
    --input "$CSV" \
    --output_dir ./results_hybrid_hypersphere_v3 \
    --graph_cache_dir ./graphs_hybrid_topo_v3 \
    --split_method scaffold \
    --seed 42 \
    --sample_size 0 \
    --val_frac 0.1 \
    --test_frac 0.1 \
    --mcs_timeout 5 \
    --hybrid_ckpt ./results_hybrid_topo_v3/phase3_hybrid/best.pt \
    --hypersphere_ckpt ./results_hypersphere_v3/phase3_hypersphere/best.pt \
    --phases 2,3,4,5
```

Set `--num_workers` to the CPU count allocated to the precompute job. Real
LOMAP scoring in the combined comparison requires `lomap2` and `gufe` from
conda-forge.

The standalone hybrid-topology output is the supervised reference checkpoint
for the combined comparison. Keep these output/cache directories separate
from prior runs; do not reuse old graph caches or checkpoints.

### Resume

Resume phases in an existing v3 output directory with the same input and
cleaning/split settings. Every script reloads Phase 1 through the validated
shared data loader, including when Phase 1 is not selected. A cache made by an
older version or for different settings requires a fresh output directory;
there is no automatic migration.

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
├── benchmark.json                  # ordered data/label/split fingerprint
├── phase1/
│   ├── cleaned_split_scaffold.pkl     # characterization copy (hypersphere)
│   ├── se_distribution.png
│   ├── pair_descriptors_sample.csv
│   ├── descriptor_se_correlations.csv
│   ├── descriptors_vs_se.png
│   ├── spearman_correlations.html     # interactive Plotly bar chart
│   ├── descriptor_scatter_interactive.html  # scatter grid w/ SMILES tooltips
│   └── go_no_go_gate.json             # pass/fail + counts
├── phase1_hybrid/
│   ├── cleaned_split_scaffold.pkl     # validated shared train/val/test frame
│   └── cleaned_split_scaffold.pkl.manifest.json
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

2. **Dispersion diagnostic** — reports mean off-diagonal cosine similarity
   between `K` cluster centers maintained via EMA over batch-wise K-means
   assignments. The centers are updated from detached embeddings, so this
   term has no gradient into the encoder.

3. **Regression head** — small MLP taking `[|h_A-h_B|, h_A*h_B, cos_sim]` →
   predicts `|SE|` directly. Trained with MSE.

Reported total: `L = w_c · L_contrast + w_d · L_dispersion + w_r · L_regression`.
The dispersion term is diagnostic and contributes no encoder gradient.

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

## Artifact formats

`mcs_precompute.py` writes `graphs_all.pt` as a versioned bundle containing
the graph list, ordered pair keys, graph-featurization version, and MCS
timeout. Chunk files use the same bundle format and are checked against
`cleaned.pkl` before merging. Inline per-split caches use the same format.
Pair order is aligned by exact ordered SMILES keys; swapped endpoints are
different identities. Failed graph entries (`None`) retain their row
positions.

Data caches are accompanied by `.manifest.json` files recording the input
file hash, request settings, and cached-pickle hash. Checkpoints include the
benchmark identity and cannot be loaded against missing or different
benchmark provenance.

These alignment repairs do not add a descriptor-based predictor or establish
that any model outperforms real LOMAP; descriptor-baseline evaluation is a
separate next stage.

## Cluster-Friendly Notes

- Auto-detects CUDA; falls back to CPU.
- All long-running steps print progress and ETA.
- Artifacts are checkpointed per phase; resume with a subset of `--phases`
  only when the data manifest and benchmark identity match.
- `--num_workers` controls DataLoader parallelism.
