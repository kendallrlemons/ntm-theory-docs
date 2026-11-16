# Fah GNN comparison results

All 12 planned runs were received under
`/Users/lemonsk/Downloads/fah_gnn_results_v1`.
The reproducible review is `living_network/review_fah_gnns.py`; its frozen
outputs are `/Users/lemonsk/Downloads/fah_gnn_review_v1/review.json` and
`summary.csv`.

## Verification

For every run, the review checked request identity, the checkpoint file
checksum, the original tree prediction/result checksums, complete test
edge IDs, endpoint structures, source identities, targets, and bootstrap
groups. It reproduced model correlations and MSE from the saved CSV and
confirmed that the selected epoch was the earliest maximum validation
Spearman in the saved history. All 12 checkpoints passed their hash check.
No model was retrained. No graph failures reduced these test cohorts.

The cluster request records consistent code hashes across all runs,
including a module named `hybrid_topo.py` rather than the local
`hybrid_topo_rbfe.py`. Archive those exact cluster source files and the
environment with the result artifacts; matching runtime metadata is not
proof of byte identity with the current local source.

The imported LOMAP values in these GNN exports have tiny floating-point
tie differences from the original tree reports. For example, strict
Spearman is 0.327233 here versus 0.327339 in the original tree report.
The differences are numerically negligible for these conclusions, but
the original files are preserved rather than silently rewritten.
Constant training-mean correlations in older cluster reports should be
treated as undefined. They do not affect model comparisons.

## Known-reference pooled edge uncertainty

Matched cohort: 7,895 records, from 7,899 graph-valid test records.
The test set has 1,215 variable chemistry groups and five reference
chemistry groups.

| Method | Pearson | Spearman | MSE |
|---|---:|---:|---:|
| Corrected LOMAP | 0.1862 | 0.0626 | Not calibrated |
| Boosted trees | 0.6651 | 0.6772 | 0.023756 |
| Hybrid-topology GNN | 0.6895 | 0.6898 | 0.022274 |
| Hypersphere GNN | 0.7103 | 0.7163 | 0.020562 |

Paired 95% chemistry-group bootstrap intervals for differences:

| Comparison | Pearson difference | Spearman difference |
|---|---|---|
| Hypersphere minus trees | [0.01792, 0.07874] | [0.01592, 0.06027] |
| Hypersphere minus hybrid | [0.00344, 0.05774] | [0.00405, 0.05584] |
| Hybrid minus trees | [-0.01491, 0.05754] | [-0.01607, 0.03663] |
| Hypersphere minus LOMAP | [0.43112, 0.61721] | [0.56689, 0.73562] |

The hypersphere model exceeds 0.7 on both point-estimate correlations.
The direct architecture comparison has positive unadjusted intervals
for both metrics. Hybrid is numerically above trees, but its intervals
include zero. These findings are conditional on the established
reference networks and this frozen configuration/seed.

## Known-reference pilot targets

The two phase targets are generation-1 stderr times square root of its
filtered work count. They are distinct from pooled edge stderr.

| Task | Method | Pearson | Spearman | MSE |
|---|---|---:|---:|---:|
| Complex | Direct pilot | 0.7935 | 0.7503 | 1.939445 |
| Complex | Trees + pilot | 0.8281 | 0.7852 | 1.121531 |
| Complex | Hybrid + pilot | 0.8262 | 0.7848 | 1.132696 |
| Complex | Hypersphere + pilot | 0.8381 | 0.7999 | 1.121721 |
| Solvent | Direct pilot | 0.8720 | 0.8706 | 0.181336 |
| Solvent | Trees + pilot | 0.8897 | 0.8919 | 0.120793 |
| Solvent | Hybrid + pilot | 0.8912 | 0.8963 | 0.119066 |
| Solvent | Hypersphere + pilot | 0.8988 | 0.9097 | 0.111755 |

Complex uses 4,265 common records; solvent uses 4,273.
Hypersphere-plus-pilot versus trees-plus-pilot has positive intervals
for both correlations in both phases:

- Complex: Pearson [0.00459, 0.01546]; Spearman [0.00648, 0.02297].
- Solvent: Pearson [0.00171, 0.01578]; Spearman [0.01140, 0.02686].

Against hybrid-plus-pilot, hypersphere has positive intervals for both
complex-phase correlations. For solvent, the Spearman difference interval
is [0.00880, 0.01868], while the Pearson interval [-0.00410, 0.01955]
includes zero. Its solvent Pearson advantage over direct pilot use also
has an interval including zero, despite the higher point estimate.

Complex hypersphere MSE is essentially tied with, and slightly worse
than, the tree model's MSE. Higher correlation does not automatically
mean better calibration or better allocation.

For completeness, the structure-only phase-proxy correlations are:

| Phase | Trees r / rho | Hybrid r / rho | Hypersphere r / rho |
|---|---|---|---|
| Complex | 0.6055 / 0.5677 | 0.5845 / 0.5826 | 0.6259 / 0.6142 |
| Solvent | 0.5739 / 0.6689 | 0.5691 / 0.6880 | 0.6114 / 0.7097 |

These structure-only phase predictors are the appropriate prior methods
to compare with pilot-informed predictors of the same proxy. The pooled
edge-stderr model must not be substituted as if it predicted that proxy.

## Strict reference-network transfer

There are 4,653 matched records from the single sprint-4 test network.

| Method | Pearson | Spearman |
|---|---:|---:|
| Corrected LOMAP | 0.2714 | 0.3272 |
| Trees | -0.0021 | -0.0494 |
| Hybrid GNN | 0.0080 | -0.0474 |
| Hypersphere GNN | -0.0073 | -0.0790 |

Neither GNN repairs the transfer failure. LOMAP remains better on this
network. The supplied endpoint bootstrap retains only 623/1,000
replicates because resamples without the sole reference family have no
evaluable edges. These intervals cannot characterize variation across
unseen reference families. Strict pilot tasks remain unsupported.

## Selection and limitations

Hypersphere has the highest saved validation Spearman among the three
model families for all five known-reference task/input combinations:

| Task/input | Trees | Hybrid | Hypersphere |
|---|---:|---:|---:|
| Edge / structure | 0.62545 | 0.67295 | 0.69662 |
| Complex / structure | 0.53282 | 0.56385 | 0.59474 |
| Complex / structure + pilot | 0.78129 | 0.78101 | 0.80430 |
| Solvent / structure | 0.62016 | 0.68119 | 0.71118 |
| Solvent / structure + pilot | 0.90802 | 0.91212 | 0.92139 |

This provides a validation-based basis for carrying hypersphere forward
as the primary learned policy, while retaining trees and hybrid as
comparators. Architecture comparisons were added after inspecting the
tree results; they are retrospective. The bootstrap intervals are
unadjusted across multiple comparisons, are conditional on known
references, and do not capture training-seed variability. Preserve
negative results and do not make a general unseen-network claim.

## Next: allocation experiment

Freeze these model weights and predictions. The first allocation study
should use the known-reference setting and compare uniform allocation,
a prespecified LOMAP weighting, the frozen structure-only phase-proxy
models, direct pilot estimates, and frozen learned pilot models.
Use identical evaluation edges for every policy.

Before assigning physical meaning to savings:

1. Define the budget unit and objective explicitly. Filtered work-pair
   counts are not GPU time or independent effective sample size.
2. Charge generation-0 effort to pilot policies. Do not grant pilot
   information for free while giving all methods the same remaining
   budget.
3. Keep complex and solvent sampling separate. Their costs and noise
   scales differ; a combined edge cannot be assigned an arbitrary shared n.
4. Distinguish a mathematical sensitivity study under `variance=s^2/n`
   from measured simulation savings. Generation-1 proxies can serve as
   retrospective reference values under that assumption, not proven
   effort-independent truth.
5. Define whether the objective is mean edge variance or network-level
   uncertainty. A star network's shared reference and clone dependence
   cannot be ignored when claiming network covariance reductions.

The correlation benchmark is complete. It motivates this next study but
does not itself validate the variance-scaling law or an allocation gain.
