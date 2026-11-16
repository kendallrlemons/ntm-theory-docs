# Pilot-informed transfer to a held-out campaign

## Question and scope

Can generation-0 information support prediction of generation-1
phase uncertainty proxies in the held-out sprint-4 campaign?
Does molecular structure improve on a learned pilot-only predictor?

This is a new retrospective experiment, protocol
`campaign_pilot_gen0_to_gen1_v1`. The original structure-only strict
holdout failure and earlier pilot eligibility results remain unchanged.
An exploratory direct-pilot check on this campaign already found complex
Pearson/Spearman 0.521/0.532 and solvent 0.795/0.849. Those values motivated
this experiment; it is not an untouched prospective test.

The original pilot eligibility rule required exactly two exported
generations. This campaign generally has six. Here, valid generations
0 and 1 are required; additional generations neither disqualify a record
nor enter the primary predictor/target. This changes eligibility, not
the frozen campaign or chemistry assignments.

## Frozen data and target rules

- Start from the original schema-v2 `strict_network/edge.csv`.
  Keep its retained edge IDs, SMILES, train/validation/test assignments,
  reference/variable groups, and bootstrap groups exactly.
- Verify the parent manifest and edge CSV, the source extraction
  manifest, and the generation CSV checksums.
- Join by source-scoped edge ID and verify source SHA identity.
  Duplicate phase/generation identifiers are errors.
- For each phase separately, require finite positive stderr and retained
  work count for generation 0 and generation 1.
- Input pilot proxy: `gen0.stderr * sqrt(gen0.num_work_pairs)`.
- Target: `gen1.stderr * sqrt(gen1.num_work_pairs)`.
- Retain reliability values only as metadata. They do not filter records
  or enter the feature matrix.
- Keep retained and raw generation work counts for future cost analysis;
  counts are not separate model features.
- Require at least 100 rows and five variable chemistry groups in each
  split. Unsupported tasks are reported rather than reassigned.

The preparation script also supports a known-reference parent, where
references absent from a task's training cohort are excluded according
to the established rule. The supplied submission scripts run the strict
setting, which is the new-campaign question.

## Models

The CPU runner fits the same three frozen candidates per feature set as
the earlier descriptor benchmark: histogram gradient boosting with 15
or 31 leaves and 100 iterations, and a 300-tree random forest with
minimum leaf size five. Training uses log(target), seed 42, and
validation Spearman only for model selection.

| Method | Inputs |
|---|---|
| Direct pilot | Unmodified generation-0 proxy |
| Learned pilot-only | One feature: log(generation-0 proxy) |
| Structure-only trees | Original 25 molecular-pair features |
| Structure-plus-pilot trees | Those 25 features and log(pilot proxy) |
| Structure-only hypersphere | Molecular graphs |
| Structure-plus-pilot hypersphere | Molecular graphs and log(pilot proxy) |

The hypersphere configuration and objectives remain fixed as in
`fah_gnn_comparison.md`: 128 dimensions, three layers, 60 epochs maximum,
patience ten, batch size 64, seed 42, standardized log-target regression
and the cosine auxiliary loss. Train all models from scratch on the new
tasks. No strict-test labels enter fitting or checkpoint selection.

Original cached descriptor features and corrected LOMAP scores are
reused with identity/checksum checks. LOMAP is a supplementary structural
comparator. The primary pilot comparisons use all eligible test records
and do not discard records merely because LOMAP is missing.

GNN graph reuse additionally verifies that every pilot row is a subset
of the exact frozen parent edge table with unchanged structures, split,
and grouping. The graph manifest must belong to that parent benchmark,
the builder recipe and package versions must match, the database checksum
must pass, and every required graph key must be present.

## Cluster files

Copy these new Python files:

- `living_network/prepare_fah_campaign_pilot.py`
- `living_network/run_fah_campaign_pilot.py`

Replace the cluster `run_fah_gnn.py` with its updated version. It now
accepts `--parent_benchmark_dir`, verifies parent membership before graph
reuse, and evaluates the learned pilot-only comparator. It records the
actual imported graph module filenames, including cluster-local names
such as `hybrid_topo.py`.

Keep the existing graph builders, `prepare_fah_gnn.py`,
`fah_gnn_models.py`, `descriptor_rbfe.py`, `rbfe_artifacts.py`, and
`run_fah_benchmark.py` alongside them. Preserve the versions that produced
the existing caches. Replacing graph builders merely to rename files
would invalidate the caches. Use the same `gnn_rbfe` environment.

These new runs have separate directories. Do not resume or overwrite the
completed historical GNN runs with the updated runner.

Copy the three submission scripts into the same cluster directory:

1. `prepare_fah_campaign_pilot.sbatch`: one CPU, derives pilot tables.
2. `fit_fah_campaign_pilot.sbatch`: four CPUs, fits descriptor and
   pilot-only models and writes their comparisons.
3. `train_fah_campaign_pilot_hypersphere.sbatch`: one GPU and four CPUs,
   runs structure-only and structure-plus-pilot hypersphere for both phases.

All scripts activate `gnn_rbfe` after sourcing `.bashrc`, and enable
unset-variable checks only after environment initialization.
Add any site-required GPU partition/account directives to the GPU script.

## Run order

First:

```bash
sbatch prepare_fah_campaign_pilot.sbatch
```

Inspect/share:

```text
/data1/choderaj/lemonsk/FEC_Sandbox/fah_campaign_pilot_v1/strict_network/coverage.json
```

Once coverage is confirmed, submit the CPU fitting job:

```bash
sbatch fit_fah_campaign_pilot.sbatch
```

Wait for that job to complete successfully, then:

```bash
sbatch train_fah_campaign_pilot_hypersphere.sbatch
```

The GPU runner needs the completed CPU predictions as its references.
No source download, graph preparation, or LOMAP rescoring is required.

Preparation requires a fresh output directory. CPU fitting preserves a
run request and skips completed runs; after interruption it refits the
small baseline models rather than resuming individual trees. GNN training
retains the established epoch-checkpoint resume behavior.

Outputs:

```text
fah_campaign_pilot_results_v1/strict_network/
  trees/
    results.json
    pilot_complex_predictions.csv
    pilot_solvent_predictions.csv
    request.json
    <selected model files>
  hypersphere/
    pilot_complex/structure/
    pilot_complex/structure_plus_pilot/
    pilot_solvent/structure/
    pilot_solvent/structure_plus_pilot/
```

Send the tree `results.json`, its two prediction CSVs, and each GNN run's
`results.json` and `predictions.csv`, preserving folder names.

## Interpretation and subsequent work

The key comparisons are learned pilot-only versus direct pilot, and
structure-plus-pilot versus learned pilot-only. This separates learned
calibration of early uncertainty from the incremental value of molecular
structure. A same-target structure-only control also avoids comparing
new phase-proxy correlations directly with the old pooled-edge-stderr
correlations.

Use paired bootstrap groups inherited from the parent. The strict test
still has one held-out reference family; group-bootstrap intervals do
not establish generalization across a population of unseen campaigns,
and some resamples omit that reference entirely.

The experiment has one training seed. It does not quantify variability
over retraining, validate effective sample size, or establish physical
simulation savings. Shared clone histories and separately filtered work
remain relevant. Pilot cost must be charged in the allocation study.

The first implementation intentionally uses generation 0 to predict
generation 1. Preparation records availability of later generations.
A subsequent sampling-effort study should define how additional early
work is combined and which nonoverlapping later samples form the target.
Do not average separately filtered uncertainty estimates or call retained
work counts GPU time without validating that interpretation.

Twelve focused synthetic tests passed across the campaign-pilot and GNN
test files, including six-generation eligibility, unchanged parent
assignments, generation-1 label changes not affecting pilot inputs,
generation-2+ changes not affecting the primary task, duplicate/source
rejection, descriptor/pilot-only CLI integration, and graph/checkpoint
integrity. Full-data preparation and fitting remain cluster work.
