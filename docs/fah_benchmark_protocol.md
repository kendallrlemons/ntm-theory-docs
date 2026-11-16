# Fah-XChem structure and pilot benchmark protocol

This protocol separates two information settings. It does not establish
the validity of a fixed-budget allocation simulation yet.

## Upstream findings

Inspected upstream Fah-XChem commits:

- `48106cd3a3c44793f2f98288a4fcfbcf77aef64c` (current inspected tree).
- `7933e2606add5a69de1b307040b28701dd455503` (latest upstream commit returned
  before the sprint-8 analysis timestamp, 2021-06-04T18:13:56Z).

These are inspected source versions, not proven runtime versions for the
downloaded analyses. The JSON does not establish that provenance.

In `fah_xchem/analysis/__init__.py`, `analyze_phase` pools extracted work
pairs and calls `compute_relative_free_energy`; it then separately calls
the same function for each generation. Generations with insufficient
valid work values can be omitted from the generation list. In
`analysis/free_energy.py`, the function filters forward/reverse work
pairs, computes BAR, and reports the number retained after filtering.
Thus pooled and per-generation counts need not add up. Exported generation
work lists are already filtered and do not necessarily reconstruct the
original pooled input.

Filtering is version-dependent. The historical source defaults to
`max_n_devs=6`; the current inspected source defaults to 100. Both inspected
implementations override the estimated spread with 5.0 in the outlier
routine. Do not recompute historical labels using current defaults.
`extract_work.py` converts work to dimensionless units by dividing by kT.
Keep the exported source units throughout this benchmark.

The reliability flag is derived from generation free-energy agreement
with pooled estimates. It is therefore a post-simulation consistency
indicator. Selecting only reliable records would condition a pilot
comparison on the behavior being evaluated. This protocol does not
filter on reliability and never uses it as a predictor.

Source links:

- https://github.com/choderalab/fah-xchem/blob/7933e2606add5a69de1b307040b28701dd455503/fah_xchem/analysis/__init__.py
- https://github.com/choderalab/fah-xchem/blob/7933e2606add5a69de1b307040b28701dd455503/fah_xchem/analysis/free_energy.py
- https://github.com/choderalab/fah-xchem/blob/7933e2606add5a69de1b307040b28701dd455503/fah_xchem/analysis/report.py
- https://github.com/choderalab/fah-xchem/blob/48106cd3a3c44793f2f98288a4fcfbcf77aef64c/fah_xchem/analysis/extract_work.py

## Frozen tasks and splits

`prepare_fah_benchmark.py` verifies the exported CSV checksums, excludes
unresolved/invalid microstate structures, and excludes identical canonical
endpoint structures. It retains source-scoped edge IDs and repeated
measurements across protocols without pretending these are independent.

**Schema 1 is superseded.** Its independent endpoint assignments produced
25,870 training and 2,324 test edge records but no validation records.
Both pilot tasks had only training data. No model results were generated
from that split. Inspection showed that every resolved source analysis is
a star network with a fixed reference compound. Requiring both endpoints
to share an assignment tied each network to its reference's assignment.

Schema 2 provides two separate generalization settings, explicitly chosen
before fitting any models. In both, compound-linked Murcko groups are
constructed globally across sources and endpoints, before target validity
or pilot eligibility filtering. Empty acyclic scaffolds remain one shared
group. Each source's reference side is inferred only if exactly one
endpoint side has a single compound ID; otherwise preparation stops
rather than guessing its role. This handles sprint 4's reversed endpoint
orientation.

### Primary: known reference

Predict new variable-ligand scaffold families connected to known reference
structures. SHA-256 with seed 42 assigns variable chemistry groups to
train/validation/test using fixed fractions 0.6/0.2/0.2. All occurrences of
a variable group across sources receive the same assignment.

References may occur across splits. Any variable ligand in a chemistry
group belonging to any reference is excluded. Variable-group, canonical
variable-ligand, and variable-compound-ID overlap across splits is asserted
absent. Each task further excludes validation/test rows whose exact
reference structure is absent from that task's training set.

This is **not both-endpoints-unseen prediction**. The claim is conditional
on established reference compounds and observed network contexts.

### Secondary: strict network holdout

Predict transformations with both endpoint chemistry groups unseen in
training. Reference chemistry families are sorted by a fixed seed-42 hash.
If at least three reference families exist, the last two blocks of
`max(1, floor(0.2 * number_of_families))` families become validation and
test; the remainder become training. There is exactly one deterministic
assignment, with no seed search or label-based balancing. Fewer than three
reference families yields training only and unsupported held-out tasks.

Networks sharing a reference chemistry family stay together. A variable
ligand belonging to a reference family assigned elsewhere is excluded.
Other variable groups occurring in multiple assigned splits are excluded
from all those splits. The retained data must have no endpoint chemistry
group, canonical ligand, or compound-ID overlap between splits.

This strict setting may lose many records or fail coverage requirements.
It also combines chemistry and protocol shifts, and may have very few
held-out reference families. It is a separate stress test, not a direct
estimate of the same generalization problem as the primary setting.

### Coverage and targets

Both settings report all three splits, including explicit zero counts,
exclusion reasons, variable/reference group counts, and task status.
Evaluation skips a task if any split has fewer than 100 rows or five
variable chemistry groups. If no task qualifies, it stops before scoring.
These are minimum execution checks, not guarantees of statistical power.
Do not adjust seeds or assignments to obtain better performance.

Each setting includes the same three targets:

1. **Edge:** observed pooled binding free-energy stderr. This is uncertainty
   at the realized effort, not a validated intrinsic difficulty.
2. **Pilot complex:** generation-1 complex stderr times the square root of
   its filtered work-pair count.
3. **Pilot solvent:** the corresponding solvent quantity.

Pilot tasks require exactly two exported generations labeled 0 and 1.
Generation-0 features are used to predict generation 1. Missing generations
are not imputed. Shared clone history, omitted generations, and successful
export selection mean this remains a retrospective continuation experiment,
not an independent-replicate or prospective validation.

## Models, evaluation, and resumption

The structure features reuse the existing 25-feature pair descriptor
implementation. Models are fixed candidates: histogram gradient boosting
with 15 or 31 leaves and 100 iterations (internal early stopping disabled),
and a 300-tree random forest with minimum leaf size 5. Each fits log target
on training data only; validation Spearman chooses the candidate. Test
targets do not select features, capacity, or training budget. No refitting
on validation data occurs.

The full exports were previously inspected descriptively to develop this
protocol. This is a frozen retrospective evaluation after that audit, not
a prospectively blinded confirmatory study. Future confirmation should use
new data or a separately reserved campaign.

Pilot tasks compare structure-only predictions, structure plus log
generation-0 proxy, and the untrained direct generation-0 proxy baseline.
Later-generation counts are used to define the later target only, never
as model inputs. A training-mean baseline is also reported.

Corrected real LOMAP uses the existing scorer with `threed=False`,
`max3d=0.0`, `shift=False`, and timeout 5 seconds. It is scored only on unique
canonical unordered test pairs. Rank-based comparisons use the common
finite-score cohort; all-test model metrics and coverage are also reported.
LOMAP scores are not calibrated uncertainty estimates.

The report includes Pearson/Spearman, source-specific results, and 1,000
paired chemistry-group bootstrap replicates. In the known-reference
setting, resampling is over variable chemistry groups conditional on the
fixed references; repeated protocol records for a variable group move
together. In the strict setting, each replicate resamples both endpoint
groups; an edge connecting two different groups receives the product of
their counts, while a within-group edge receives one count.
Weighted ranks are recomputed within each replicate. Intervals require at
least 200 valid replicates. These are descriptive intervals across the
frozen family of comparisons, without multiplicity correction. Per-record
metrics give more weight to chemistry with more protocol records; report
source-specific results and group counts alongside pooled metrics.
Known-reference intervals do not quantify uncertainty over new references.
Strict intervals must be interpreted cautiously when few reference
families are held out; test reference and variable group counts are
included in results.

The evaluator writes package/code identity, checked feature caches, a
LOMAP progress cache (every 20 completed pairs), selected model artifacts,
validation selection scores, test predictions, coverage, and `results.json`.
An interrupted scoring run can resume with identical inputs and code.
Model fitting may repeat after interruption; a completed results file
prevents automatic reevaluation. Use the same RDKit version for preparation
and evaluation. Do not modify a frozen experiment in place.

## Cluster commands

Copy these new files to the cluster alongside the current
`descriptor_rbfe.py`, `hybrid_topo_rbfe.py`, `rbfe_artifacts.py`, and their
existing local dependencies. Use the established RDKit/scikit-learn/LOMAP/
GUFE/PyTorch environment. The scorer imports the existing pipeline, but
this experiment does not train a GNN or require GPUs.

Run preparation first, using the original extraction manifest filename:

```bash
python prepare_fah_benchmark.py \
  --edges /data1/choderaj/lemonsk/FEC_Sandbox/moonshot_edges_v2/edges.csv \
  --generations /data1/choderaj/lemonsk/FEC_Sandbox/moonshot_edges_v2/generations.csv \
  --manifest /data1/choderaj/lemonsk/FEC_Sandbox/moonshot_edges_v2/manifest.json \
  --setting both \
  --output_dir /data1/choderaj/lemonsk/FEC_Sandbox/fah_benchmark_v2
```

Inspect/share `fah_benchmark_v2/coverage.json` before expensive scoring.
The directory contains separate `known_reference/benchmark.json` and
`strict_network/benchmark.json` manifests and task CSVs. No labels were used
to optimize coverage. Original extraction outputs need not be regenerated.

For settings with adequate coverage, run evaluation separately:

```bash
python run_fah_benchmark.py \
  --benchmark_dir /data1/choderaj/lemonsk/FEC_Sandbox/fah_benchmark_v2/known_reference \
  --output_dir /data1/choderaj/lemonsk/FEC_Sandbox/fah_results_known_reference_v2 \
  --workers 8 --threads 8

python run_fah_benchmark.py \
  --benchmark_dir /data1/choderaj/lemonsk/FEC_Sandbox/fah_benchmark_v2/strict_network \
  --output_dir /data1/choderaj/lemonsk/FEC_Sandbox/fah_results_strict_network_v2 \
  --workers 8 --threads 8
```

The worker and thread counts must fit the allocated CPU resources.
Preparation and full real-data scoring/training are intended for the
cluster, not the laptop. Share each setting's `results.json` and available
`*_predictions.csv` files after evaluation. Unsupported tasks are explicitly
listed in results and receive no predictions.

## Allocation work still needed

The current proxy uses filtered counts, not independent effective sample
size or GPU cost. A generation represents an additional simulation block,
not necessarily a short inexpensive pilot. A valid allocation experiment
must charge pilot cost, define equal total computational budgets, and
validate uncertainty scaling with additional work while accounting for
clone dependence. This benchmark measures the predictive signal needed
for that next experiment; it does not claim an allocation benefit.
