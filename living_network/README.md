# Living Network — Hyperspherical RBFE Difficulty Model

All-in-one script implementing Phases 1–5 of the experimental plan.

> **Scaffold correction:** The earlier `scaffold` implementation used an
> invalid RDKit keyword and fell back to full-SMILES grouping. Prior v3
> reference and descriptor commands below are historical and are superseded
> for future scaffold-held-out runs. The label is now known to be a
> difference of per-compound quantities. The current next step is
> source-preserving Fah-XChem edge extraction; no training is part of this
> step.

## Current next action: prepare sequential network-growth experiments

See [the network-growth experiment](../docs/fah_network_growth_experiment.md).
`prepare_fah_growth.py` freezes source-specific initial networks and ligand
arrival batches from the original known-reference benchmark. Its coverage
report distinguishes observed reference connections from alternative edges
that need new simulations. Submit `prepare_fah_growth.sbatch` on one CPU.
This is experiment preparation, not a completed growth-policy evaluation.

## Earlier workflow: source-preserving Fah-XChem extraction

Run these commands on the cluster from `living_network`. First install the
optional streaming JSON dependency into the repository `.venv` (activate it
first):

```bash
python -m pip install -r requirements-fah.txt
```

To inspect available analyses without creating files or downloading:

```bash
python download_fah_analyses.py \
  --output_dir /data1/choderaj/lemonsk/FEC_Sandbox/moonshot \
  --list-only
```

Download the listed `analysis.json` files with resumable verification:

```bash
python download_fah_analyses.py \
  --output_dir /data1/choderaj/lemonsk/FEC_Sandbox/moonshot
```

The downloader follows paginated S3 listings for the configured sprint
prefixes, retains each object key hierarchy beneath the output directory, and
records size, ETag, and SHA-256 in `download_manifest.json`. Verified files
are skipped; resumable `.part` downloads use `curl`. It refuses to overwrite
unmanaged files or files that fail manifest verification. Do not download the
full corpus locally.

Then extract the edge tables from the downloaded analyses:

```bash
python fah_edge_dataset.py \
  --input_dir /data1/choderaj/lemonsk/FEC_Sandbox/moonshot \
  --output_dir /data1/choderaj/lemonsk/FEC_Sandbox/moonshot_edges_v2
```

Extraction requires a fresh output directory; reruns do not overwrite existing
tables.

The loader writes `compounds.csv`, `microstates.csv`, `edges.csv`,
`generations.csv`, and `manifest.json`. These preserve source identities and
source values without filtering or model/split assignment. Original raw work
values remain in each source JSON; refer to a transformation by its source
SHA-256 and transformation index.

Schema v2 preserves repeated microstate records using
`source_microstate_index`. Endpoint resolution uses all structures associated
with each compound/microstate key. Conflicting structures leave the microstate
SMILES blank and mark `resolution_status=ambiguous_structure`; identical
repeated structures resolve without dropping source records. The manifest
counts duplicate and ambiguous keys. Filter unresolved endpoints before
microstate-based featurization.

For the completed table audit, overlap findings, and generation-level
interpretation, see [Fah-XChem edge-table audit](../docs/fah_edge_audit.md).
The reproducible audit entry point is `audit_fah_tables.py`.

Next, follow [Fah-XChem benchmark protocol](../docs/fah_benchmark_protocol.md) for schema-v2 preparation of separate known-reference and strict network-holdout settings. The original v1 split had empty validation and pilot test sets and must not be evaluated.

For the additional hypersphere and hybrid-topology GNN comparisons on the
same frozen tasks, see [Fah GNN comparison](../docs/fah_gnn_comparison.md).

**Interpretation limits:** The phase-specific `stderr * sqrt(num_work_pairs)`
columns are effort-normalized proxies, not validated intrinsic difficulty
estimates. There is no combined proxy because the two legs have different
sampling counts. Work pairs need not be independent. BAR overlap, realized
counts, and reliability are post-simulation measurements and must not enter a
pre-simulation predictor. Receptor variants measure protocol sensitivity, not
independent replicate noise. Units are preserved from the source; verify
upstream units before conversion.

## Historical compound-level uncertainty model command

The label-structure diagnostic found `y(A, B) ~= c + u_A + v_B` (held-out
edge R² 0.991; triangle residual 10% of the permutation null). This
historical experiment recovers `u`, `v`, `c` from training edges only, trains
structure -> `u` and structure -> `v` models on training compounds, selects
capacity on validation pairs only, and predicts held-out test pairs as
`c + u_hat(A) + v_hat(B)`. Its command is retained for context, not as the
current next action:

```bash
python compound_uncertainty_model.py \
  --input /data1/choderaj/lemonsk/FEC_Sandbox/compound_smiles_stderr_differences.csv \
  --reference_dir /data1/choderaj/lemonsk/FEC_Sandbox/results_hybrid/results_hybrid_topo_v4 \
  --lomap_reference_dir /data1/choderaj/lemonsk/FEC_Sandbox/results_hybrid/results_descriptors_scaffold_v4 \
  --output_dir /data1/choderaj/lemonsk/FEC_Sandbox/results_hybrid/compound_model_v1 \
  --threads 8
```

It compared `|prediction|` with corrected real LOMAP on the same 7,646 test
pairs (no LOMAP recomputation; reference benchmark, pair set, labels, and
scoring version verified). CPU-only; no GPU, LOMAP/GUFE, graph cache, or
checkpoint. Its outputs include `summary.json`, `comparison.csv`,
`validation_search.json`, `train_compound_values.csv`, and
`test_predictions.csv`. The test-only two-slot fit is an evaluation ceiling
and is never used for training or selection.

## Completed label-structure diagnostic (history)

Command retained for the record:

```bash
python label_structure_diagnostic.py \
  --input /data1/choderaj/lemonsk/FEC_Sandbox/compound_smiles_stderr_differences.csv \
  --reference_dir /data1/choderaj/lemonsk/FEC_Sandbox/results_hybrid/results_hybrid_topo_v4 \
  --output_dir /data1/choderaj/lemonsk/FEC_Sandbox/results_hybrid/label_structure_v1
```

The earlier diagnostic used training-split labels only and reported
reversed-pair antisymmetry, edge-held-out R² for `s_A - s_B` versus
`c + u_A + v_B`, and triangle closure against a permutation null. Self-pairs
were counted separately and excluded from fits. Its output was
`label_structure.json`.

## Completed validation-only descriptor learning curve (history)

Validation Spearman was 0.0705 at 100,000 pairs and 0.0627 at 500,000; MSE
improved over the training mean by about 0.5% at both sizes. The command is
retained for the record:

```bash
python descriptor_learning_curve.py \
  --input /data1/choderaj/lemonsk/FEC_Sandbox/compound_smiles_stderr_differences.csv \
  --reference_dir /data1/choderaj/lemonsk/FEC_Sandbox/results_hybrid/results_hybrid_topo_v4 \
  --output_dir /data1/choderaj/lemonsk/FEC_Sandbox/results_hybrid/results_descriptor_learning_curve_v1 \
  --train_sizes 100000,500000 \
  --threads 8
```

This fits fixed 15-leaf/50-iteration descriptor models at the nested 100,000
and 500,000 training sizes, using all validation rows. The 100,000-row
prefix matches the original pandas sample; each fit restores source-row order.
Only training and validation pairs are passed to feature extraction, fitting,
and validation metrics. The full split is loaded to verify benchmark identity;
test labels are used for that provenance check only. No test features,
predictions, or metrics are produced, and test data does not select the
training budget. Each size reports its own training-mean baseline.

This is CPU-only: no GPU, LOMAP/GUFE, graph cache, GNN checkpoint, or GNN
inference is needed. Set `--threads` within the allocated CPU count. The
script writes `validation_learning_curve.csv`, `learning_curve.json`,
`validation_predictions_100000.csv`, `validation_predictions_500000.csv`,
`model_100000.pkl`, `model_500000.pkl`, training/validation feature caches
and manifests, `learning_curve_manifest.json`, and `benchmark.json`.
Completed feature/model caches can resume with identical arguments. Feature
generation is cached after completion but has no incremental
per-feature interruption checkpoint. Changing sizes or dependency/model
settings requires a fresh output directory. Share `validation_learning_curve.csv`
and `learning_curve.json`; no test predictions are needed.

The latest paired-bootstrap intervals all include zero; make no superiority
claim. Copy the updated `descriptor_learning_curve.py` and its shared v4
dependencies to the cluster before running.

## Completed rescore-only comparison (history)

The matched no3d audit confirmed a mapper-configuration effect in its
34 diagnostic cases; see the results note below. The saved-prediction rescore
was completed without fitting models, running GNN inference, generating
graphs, or changing the split. Its command and outputs are retained here.

```bash
python rescore_lomap.py \
  --predictions /data1/choderaj/lemonsk/FEC_Sandbox/results_hybrid/results_descriptors_fast_v1/test_predictions.csv \
  --output_dir /data1/choderaj/lemonsk/FEC_Sandbox/results_hybrid/results_descriptors_rescored_v2 \
  --workers 8 --mcs_timeout 5
```

The rescore output preserves original predictions in a separate directory
and contains `test_predictions.csv`, `comparison.csv`,
`comparison_previous_cohort.csv`, `coverage.json`, and
`rescore_manifest.json`. The score version change invalidated old
LOMAP feature/progress caches; the graph cache was unaffected. This remains
a historical ligand-held-out cohort, not a scaffold-held-out evaluation.

## Completed Phase 1 v4 and no3d audit (history)

The strict-scaffold Phase 1 run is complete: train 2,272,233, validation
16,377, test 7,646; 669,351 straddling pairs were dropped. Do not rerun Phase
1 now. Its command is retained here for the record:

```bash
python hybrid_topo_rbfe.py \
  --input /data1/choderaj/lemonsk/FEC_Sandbox/compound_smiles_stderr_differences.csv \
  --output_dir /data1/choderaj/lemonsk/FEC_Sandbox/results_hybrid/results_hybrid_topo_v4 \
  --split_method scaffold --seed 42 --sample_size 0 \
  --val_frac 0.1 --test_frac 0.1 --phases 1
```

The earlier no3d audit command and notes are retained as completed history,
not recommendations to repeat the audit:

```bash
python audit_lomap.py \
  --predictions /data1/choderaj/lemonsk/FEC_Sandbox/results_hybrid/results_descriptors_fast_v1/test_predictions.csv \
  --raw_csv /data1/choderaj/lemonsk/FEC_Sandbox/compound_smiles_stderr_differences.csv \
  --output_dir /data1/choderaj/lemonsk/FEC_Sandbox/results_hybrid/lomap_audit_no3d_v2 \
  --per_group 8 --seed 42 --mcs_timeout 5 --max3d 0
```

The audit sampled at most 8 cases from each of four disjoint diagnostic
strata, plus two controls (at most 34 cases); it is not a random representative
performance sample. It recorded each failure stage, MMFF status, and installed
source. `--max3d 0` disables geometric trimming as a diagnostic; it is not
tuning against performance. The optional `--raw_csv` flag reads only the CSV
header to inventory available columns and cannot recover pair, target,
protocol, or effort provenance. Real LOMAP auditing requires `lomap2` and
`gufe` on the cluster.

For any later training, review the phase counts first; scaffold straddlers
are dropped, so requested fractions do not guarantee nonempty or large
splits. The strict split version invalidates old Phase 1 manifests; they are
rejected, not regenerated in place. Any future model trained on this split
needs a new checkpoint and must not reuse a v3 checkpoint. The identity-keyed
`graphs_hybrid_topo_v3` bundle can be reused only if it covers all requested
ordered pairs and has the matching graph version and MCS timeout; the full
precompute is expected to cover them.

## Historical v3 run examples (superseded for future scaffold runs)

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

The descriptor comparison below is an additional experiment; it does not
establish that any model outperforms real LOMAP.

## Descriptor baseline against the completed GNN

### Faster first comparison: test-only LOMAP

The `test-only` mode trains the descriptor model without LOMAP features and
scores real LOMAP only on the 7,522 test pairs. The default `both` mode below
is unchanged and scores LOMAP on the capped training set plus all validation
and test pairs (114,999 calls for the specified benchmark). `test-only`
therefore makes about 15 times fewer LOMAP calls, but this does not guarantee
a 15-fold wall-clock speedup. LOMAP is CPU work; GPU allocation does not
accelerate it. Keep `--train_cap 100000` for this first run.

Run from `living_network` on the cluster in a fresh `results_descriptors_fast_v1`
directory:

```bash
python descriptor_rbfe.py \
  --input /data1/choderaj/lemonsk/FEC_Sandbox/compound_smiles_stderr_differences.csv \
  --reference_dir /data1/choderaj/lemonsk/FEC_Sandbox/results_hybrid/results_hybrid_topo_v3 \
  --output_dir /data1/choderaj/lemonsk/FEC_Sandbox/results_hybrid/results_descriptors_fast_v1 \
  --hybrid_ckpt /data1/choderaj/lemonsk/FEC_Sandbox/results_hybrid/results_hybrid_topo_v3/phase3_hybrid/best.pt \
  --graph_cache_dir /data1/choderaj/lemonsk/FEC_Sandbox/results_hybrid/graphs_hybrid_topo_v3 \
  --train_cap 100000 --workers 8 --threads 8 --mcs_timeout 5 \
  --lomap_mode test-only
```

The script pins the benchmark split/seed and validates the reference
manifest/identity. `lomap_progress.pkl` atomically checkpoints completed
scores every 100 results; both successful scores and failed mappings are
marked done and reused when rerun with the same mode/settings. A changed
request is rejected, so use a new output directory when changing mode or
settings. Do not lower `--mcs_timeout` as a speed workaround: that changes the
LOMAP scoring setting. The currently running pre-change process cannot gain
this resume behavior by editing the file on disk, and its uncheckpointed
in-memory partial work cannot be recovered by the new progress file; restart
with the updated script and fresh output directory.

`--workers` and `--threads` should not exceed the CPU allocation for the
cluster job. No graph precomputation or GNN retraining is needed; the optional
GNN comparator only performs inference with the supplied existing checkpoint
and graph cache. The `test-only` run uses the real LOMAP test scores as a
comparator, but does not train on LOMAP features.

### Full LOMAP-feature comparison (`both`, unchanged default)

Run this experiment from `living_network` on the cluster, after the
`results_hybrid_topo_v3` reference run is complete. The reference directory
must contain the validated `phase1_hybrid/cleaned_split_scaffold.pkl` and its
manifest, plus `benchmark.json`; the JSON alone is not sufficient. The
script reloads the frame through the shared Phase 1 loader and checks its
identity against the reference benchmark. It pins seed 42, sample size 0,
scaffold splitting, and 0.1 validation/test fractions. The
reference benchmark is expected to have identity
`3262fee113b448460271646d5967266fb76c9dfcf4e0bbc79061d79164d02473` and
counts of train 2,399,294 / val 7,477 / test 7,522. These are a check against
the supplied completed benchmark, not values hardcoded in the experiment.

```bash
python descriptor_rbfe.py \
  --input /data1/choderaj/lemonsk/FEC_Sandbox/compound_smiles_stderr_differences.csv \
  --reference_dir /data1/choderaj/lemonsk/FEC_Sandbox/results_hybrid/results_hybrid_topo_v3 \
  --output_dir /data1/choderaj/lemonsk/FEC_Sandbox/results_hybrid/results_descriptors_v1 \
  --hybrid_ckpt /data1/choderaj/lemonsk/FEC_Sandbox/results_hybrid/results_hybrid_topo_v3/phase3_hybrid/best.pt \
  --graph_cache_dir /data1/choderaj/lemonsk/FEC_Sandbox/results_hybrid/graphs_hybrid_topo_v3 \
  --train_cap 100000 --workers 8 --threads 8 --mcs_timeout 5
```

Allocate the requested CPU worker and thread counts to the cluster job, and
adjust them to match its actual allocation. Real LOMAP scoring is an extra
feature and a comparator; it requires the `lomap2` and `gufe` packages from
conda-forge and may be expensive. Scores come from the existing
`hybrid_topo_rbfe._lomap_like_score` implementation, are processed in
parallel, and are cached with a manifest. Pair descriptors are 25 symmetric features:
endpoint descriptor sums, absolute descriptor deltas, Morgan shared/changed
bit counts, and Morgan Tanimoto difficulty. Variant 2 adds real LOMAP as an
extra feature. Both variants use the same eligible training and validation
rows; selection among 15/31 leaves and 50/100/200 iterations uses validation
Spearman only. Test labels are used only after that selection for final
reporting; the validation search does not receive test arrays.

The default train cap of 100,000 is an intentional compute budget for LOMAP
preprocessing and differs from the full-data GNN training. It caps only the
training rows; validation and test membership are retained in full. Nonfinite
LOMAP failures are excluded from both descriptor variants for a matched
comparison. Final metrics use the intersection of finite predictions from all
included comparators. `coverage.json` records selected/eligible counts and
the common test comparison coverage; conclusions are conditional on that
coverage.
Missing or invalid descriptor features, LOMAP failures, and (when enabled)
failed GNN graph rows can make the eligible test cohort differ from prior
tables. `test_predictions.csv` retains per-pair predictions and the common
eligibility flag for auditing.

The run writes `split_diagnostics.json`, `coverage.json`,
`validation_search.json`, `comparison.csv`, `test_predictions.csv`,
`features.pkl` and its manifest, `models.pkl`, and `benchmark.json`. No GNN
retraining or graph precomputation is done. To skip GNN inference, omit both
`--hybrid_ckpt` and `--graph_cache_dir`; they must be supplied together.

Optional staged execution uses the same arguments and output directory:

```bash
python descriptor_rbfe.py ... --phases features
python descriptor_rbfe.py ... --phases train,evaluate
```

The first command performs and caches descriptor/LOMAP feature generation.
The second reuses the checked feature cache and does not repeat LOMAP work.
Changing `--train_cap` changes the cache request; use a new output directory.
`--train_cap 0` uses all training pairs and can be much more expensive.
Reported top-25% hard-pair metrics are retrospective cohort rankings, not a
deployment cutoff. Paired scores do not by themselves establish statistical
superiority.

## Cluster-Friendly Notes

- Auto-detects CUDA; falls back to CPU.
- All long-running steps print progress and ETA.
- Artifacts are checkpointed per phase; resume with a subset of `--phases`
  only when the data manifest and benchmark identity match.
- `--num_workers` controls DataLoader parallelism.
