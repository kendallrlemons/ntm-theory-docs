# Hypersphere and hybrid-topology GNN comparison

This experiment adds two GNNs to the frozen schema-v2 Fah benchmark.
It does not regenerate splits, change targets, rerun LOMAP, or reuse
checkpoints from the earlier compound-stderr-difference experiments.
The tree results have already been examined, so these are explicitly
additional retrospective comparisons, not a prospectively blinded study.

## Models and training protocol

**Hypersphere:** reuse `HypersphereEncoder`, atom/bond features, and
single-molecule graph construction from `hypersphere_rbfe.py`. Each ligand
gets a unit-length embedding. The regression head receives absolute
embedding differences, elementwise products, and cosine similarity.
Training includes the original continuous cosine-target loss:
`target_cos = cos(pi * clamp(target / training_q99, 0, 1))`.
Its weight is fixed at 1.0.

The old cluster-center dispersion term is omitted because those centers
are detached EMA buffers and that term has no gradient to model parameters.
No cluster-based OOD score is evaluated. This is an adapted supervised
hypersphere method with its geometry loss, not a claim to reproduce the
old task or all original diagnostics.

**Hybrid topology:** reuse `build_hybrid_graph` and `HybridGNN.encode`
from `hybrid_topo_rbfe.py`. The graph merges MCS core atoms and labels
endpoint-unique atoms. It uses the original MCS timeout of five seconds
and the existing disconnected-graph fallback if no core is found. Endpoint
SMILES are sorted before constructing this symmetric-target representation.
The pooled embedding feeds the original-width regression head.

Both use 128 hidden/embedding dimensions, three message-passing layers,
dropout 0.1, batch size 64, Adam with learning rate 0.001 and weight decay
1e-5, and gradient clipping at norm 5. Regression predicts a standardized
log target; its mean and standard deviation come from training rows only.
This preserves the log-target choice of the tree experiment while
conditioning the neural loss numerically. Hypersphere uses standardized
log-target MSE plus the cosine loss; hybrid topology uses only that MSE.

Training runs for at most 60 epochs and stops after 10 consecutive epochs
without validation-Spearman improvement. The best validation checkpoint
is selected, with no test-based selection. This is one fixed configuration
and one seed (42) per run, not an architecture-specific hyperparameter
search or an estimate of neural training-seed variability.

Pilot-informed heads receive one extra input: the standardized log of
the generation-0 proxy, again using training-only normalization. Only the
pilot tasks support that mode. Later-generation counts are used solely
to define targets, not as prediction inputs.

## Scope

For each architecture, run these six experiments:

| Setting | Task | Input mode |
|---|---|---|
| Known reference | Pooled edge stderr | Structure |
| Known reference | Complex generation-1 proxy | Structure |
| Known reference | Complex generation-1 proxy | Structure + pilot |
| Known reference | Solvent generation-1 proxy | Structure |
| Known reference | Solvent generation-1 proxy | Structure + pilot |
| Strict network holdout | Pooled edge stderr | Structure |

Strict pilot tasks remain unsupported because their frozen test cohorts
are empty. They must not be recreated with a different split.

## Files and environment

New files to copy into the cluster script directory:

- `prepare_fah_gnn.py`
- `fah_gnn_models.py`
- `run_fah_gnn.py`
- `run_fah_gnn_suite.sh`

They also require the current `hypersphere_rbfe.py`, `hybrid_topo_rbfe.py`,
`run_fah_benchmark.py`, `descriptor_rbfe.py`, and `rbfe_artifacts.py`
alongside them. Use the established environment with CUDA-enabled PyTorch,
RDKit, NumPy, pandas, SciPy, scikit-learn, matplotlib, Plotly, joblib, and
threadpoolctl. Graph preparation and training must use matching RDKit and
PyTorch versions; RDKit must match the frozen benchmark (2025.09.2 for the
supplied cluster artifacts).

The latest shared metric helper explicitly treats constant-predictor
correlations as undefined. Copy it for these runs, but do not rerun the
completed tree experiment.

## Step 1: CPU graph preparation

Run from the script directory on a CPU allocation:

```bash
for architecture in hypersphere hybrid; do
  python prepare_fah_gnn.py \
    --benchmark_dirs \
      /data1/choderaj/lemonsk/FEC_Sandbox/fah_benchmark_v2/known_reference \
      /data1/choderaj/lemonsk/FEC_Sandbox/fah_benchmark_v2/strict_network \
    --architecture "$architecture" \
    --output_dir "/data1/choderaj/lemonsk/FEC_Sandbox/fah_gnn_graphs_v1/$architecture" \
    --workers 8
done
```

No model training or LOMAP scoring happens here. Each cache includes
structure keys required by all runnable tasks in both settings, so pilot
variants reuse the same graphs. Hypersphere caches individual molecules;
hybrid topology caches unique canonical unordered pairs.

SQLite commits completed graphs every 64 entries. Rerunning the identical
command resumes interrupted preparation. Completed caches have a database
checksum and a manifest tying them to benchmark identities, graph-builder
code, package versions, and endpoint ordering. Each cache records failed
keys; the hybrid cache also reports graphs with no MCS core. No labels
are stored in the graph database.

A graph preparation interruption can lose at most the uncommitted batch.
Never run two writers on the same cache directory simultaneously.

## Step 2: GPU training and comparison

After graph preparation finishes, submit these as separate jobs, each
with one GPU. Each command runs its six experiments sequentially:

```bash
bash run_fah_gnn_suite.sh hypersphere
```

```bash
bash run_fah_gnn_suite.sh hybrid
```

The launcher defaults to `/data1/choderaj/lemonsk/FEC_Sandbox`. Override
`FAH_ROOT` if needed, `FAH_PYTHON` for the Python executable, and
`FAH_THREADS` for CPU threads (default four). An optional second argument
restricts it to `known_reference` or `strict_network`.

For example, one explicit run is:

```bash
python run_fah_gnn.py \
  --benchmark_dir /data1/choderaj/lemonsk/FEC_Sandbox/fah_benchmark_v2/known_reference \
  --reference_dir /data1/choderaj/lemonsk/FEC_Sandbox/fah_results_known_reference_v2 \
  --graph_cache /data1/choderaj/lemonsk/FEC_Sandbox/fah_gnn_graphs_v1/hypersphere \
  --architecture hypersphere --task edge --mode structure \
  --output_dir /data1/choderaj/lemonsk/FEC_Sandbox/fah_gnn_results_v1/known_reference/hypersphere/edge/structure \
  --threads 4
```

The default is `--phase all --device cuda`. A missing GPU is an error,
not a silent CPU fallback. To separate training from evaluation:

```bash
FAH_PHASE=train bash run_fah_gnn_suite.sh hypersphere
FAH_PHASE=evaluate FAH_DEVICE=cpu bash run_fah_gnn_suite.sh hypersphere
```

The same options work for hybrid topology. Use the same installed
environment, code, and artifacts for both stages. CPU evaluation includes
graph inference and bootstrap calculations and can take time.

## Outputs, resumption, and comparisons

Outputs are organized as:

```text
fah_gnn_results_v1/
  known_reference/<architecture>/<task>/<mode>/
  strict_network/<architecture>/edge/structure/
```

Each successful run writes:

- `request.json`: benchmark, cache, reference-file, code, and package identities.
- `checkpoint.pt`: latest optimizer/model state, RNG state, selected
  model weights, training-only statistics, and early-stopping state.
- `history.json`: losses and validation correlation per epoch.
- `predictions.csv`: every original test row, keyed by edge ID, with graph
  eligibility, GNN predictions, and saved tree/LOMAP/pilot comparators.
- `results.json`: coverage, selected epoch, correlations, MSE, per-source
  comparisons, and paired chemistry-group bootstrap intervals.

Training checkpoints are atomic at epoch boundaries. Resuming repeats an
incomplete epoch from its start and preserves epoch shuffle and Torch RNG
state. Exact CPU resume was tested; GPU kernels may still have
nondeterministic behavior, which PyTorch is configured to warn about.
Do not alter code or settings mid-run. Completed result files prevent
automatic retraining or reevaluation. Only load graph/checkpoint files
generated by this trusted workflow.

Graph failures are reported by split, and test rows remain in predictions
with missing GNN values. No positional truncation or unkeyed cache reuse
is permitted. Each paired comparison uses common finite predictions;
all graph-valid test metrics are also reported. Baseline predictions must
match the complete frozen test IDs, structures, targets, resampling groups,
and (where applicable) pilot inputs before comparison.

Each GNN is compared against the saved tree model and corrected LOMAP.
Pilot tasks additionally compare against the saved learned tree-plus-pilot
model and direct pilot estimate. The same group-bootstrap definition is
used as in the tree benchmark. The single strict reference-family
limitation remains; no intervals establish performance across unseen
reference populations.

Send `results.json` and `predictions.csv` from each completed run, preserving
their directory names. Checkpoints and graph databases are not needed to
review the comparisons. Do not select the winning architecture by test
performance and then describe that test as an untouched confirmation.

Synthetic checks cover both model forward/backward paths, hypersphere
unit norms, graph failure/row alignment, exact CPU epoch resume, reference
identity rejection, graph-cache resumption/checksums, and paired evaluation.
No real-data GNN training or full graph preprocessing was run locally.

After this fixed comparison, preserve all results (including failures or
negative comparisons) before beginning the separately specified
budget-allocation experiment.
