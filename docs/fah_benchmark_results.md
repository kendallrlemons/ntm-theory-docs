# Fah-XChem benchmark results: schema 2

Results supplied as `results_known.json`, `results_strict.json`, and four
prediction CSVs in `/Users/lemonsk/Downloads`. All prediction rows have
unique source-scoped edge IDs and test split labels. Their targets agree
with the original exported edge/generation tables (tolerance 1e-12).
Pilot-direct values agree with generation-0 stderr times square root of
its work count. Model and LOMAP correlations reproduce with SciPy to
absolute tolerance 1e-12 using pandas `float_precision="round_trip"`.
Default CSV parsing can perturb LOMAP ties and change Spearman in its
sixth decimal place.

The verified additional diagnostics and prediction file hashes are
captured in `/Users/lemonsk/Downloads/fah_results_verified.json`. Bootstrap
intervals below are the supplied cluster results, not a second model
selection or rerun of the bootstrap. The six supplied result/prediction
files remain unchanged.

## Known-reference edge uncertainty

The primary claim is prediction on new variable-ligand scaffold groups
connected to established reference structures. It is not a claim that
both endpoints are unseen.

Validation selected 31-leaf histogram gradient boosting from the frozen
candidate set. There are 7,899 test records, 1,215 variable chemistry
groups, and five reference chemistry groups. LOMAP is finite on 7,895.

| Method | Common-cohort Pearson | Common-cohort Spearman |
|---|---:|---:|
| Structure model | 0.665122 | 0.677161 |
| Corrected LOMAP | 0.186232 | 0.062596 |

The paired 95% variable-group bootstrap interval for model minus LOMAP is
[0.3919, 0.5680] for Pearson and [0.5299, 0.6966] for Spearman (1,000 valid
replicates each). These intervals are conditional on the known reference
networks and do not quantify uncertainty over new references.

On all test rows, model MSE is 0.023746 versus 0.041729 for the training
mean. Conventional test R² is 0.4243. This is an improvement in numerical
error as well as ranking, but does not establish calibrated uncertainty
under new simulation budgets.

As a post-hoc diagnostic, subtracting each source's test mean from both
targets and structure predictions gives pooled within-source Pearson
0.6073. Thus the pooled association is not solely between-source mean
differences. This is a descriptive diagnostic, not a newly fitted
predictor or deployment procedure.

Results vary by source. Known-reference edge Spearman is approximately
0.30 on sprint 8, 0.40 on sprint 7, and 0.74–0.83 on several sprint-11/11A
variants. LOMAP is better in three of the 23 source-specific Spearman
comparisons, which have 11, 28, and 28 common records respectively. No
claim is made that the model dominates every source.

## Pilot-informed prediction

Targets here differ from the pooled edge task: they are generation-1
phase stderr multiplied by square root of filtered work-pair count.
Generation-0 information is available to pilot methods. Models are
evaluated on held-out variable chemistry with known references.

| Task / method | Common-cohort Pearson | Common-cohort Spearman |
|---|---:|---:|
| Complex: structure only | 0.605519 | 0.567722 |
| Complex: corrected LOMAP | 0.324331 | 0.272346 |
| Complex: direct pilot proxy | 0.793506 | 0.750323 |
| Complex: structure + pilot | 0.828093 | 0.785167 |
| Solvent: structure only | 0.573907 | 0.668912 |
| Solvent: corrected LOMAP | 0.322194 | 0.330973 |
| Solvent: direct pilot proxy | 0.871993 | 0.870601 |
| Solvent: structure + pilot | 0.889742 | 0.891896 |

Complex has 4,267 test rows (4,265 common), solvent 4,275 (4,273 common).
Validation selected hgb31 for complex structure-plus-pilot and hgb15
for solvent structure-plus-pilot.

The more informative incremental comparison is the learned pilot model
versus direct pilot use. On all test rows, paired 95% variable-group
bootstrap intervals for the difference are:

| Task | Pearson difference interval | Spearman difference interval |
|---|---|---|
| Complex | [0.02554, 0.04454] | [0.02561, 0.04453] |
| Solvent | [-0.02219, 0.05568] | [0.00960, 0.03283] |

Both complex improvements and the solvent rank improvement have
intervals above zero. The solvent Pearson improvement is inconclusive.
These are unadjusted intervals for the frozen set of comparisons.
Because the learned model can also recalibrate the pilot signal, its
advantage over the direct baseline does not isolate a causal contribution
from structural descriptors; a learned pilot-only comparator would be
needed for that claim.

All-test MSE is 1.1212 for the complex learned pilot model versus 1.9391
for the direct pilot, and 0.12074 versus 0.18130 for solvent. These are
descriptive point comparisons without MSE bootstrap intervals. The complex
and solvent target scales differ.

The >0.8 correlations are simulation-informed results, not structure-only
performance. Generations share clone histories; effort-normalized
uncertainties remain proxies. These results do not yet measure an
equal-budget allocation benefit or prospective independent replication.

## Strict network holdout

The strict edge test contains 4,684 records from one reference family
and one source (sprint 4); 4,653 have finite LOMAP. Validation selected
the random forest.

| Method | Common-cohort Pearson | Common-cohort Spearman |
|---|---:|---:|
| Structure model | -0.002075 | -0.049424 |
| Corrected LOMAP | 0.271384 | 0.327339 |

The learned model fails to rank uncertainty in this held-out network,
and LOMAP performs better. All-test model R² is -0.1122. Although its MSE
(0.028974) is below the transferred training-mean baseline (0.040621),
that does not rescue its failed ranking performance.

The supplied endpoint-group bootstrap retains only 623/1,000 replicates:
resamples omitting the sole reference group have no evaluable edges.
Its model-minus-LOMAP intervals are negative, but cannot establish
population-level uncertainty over unseen reference families. Treat this
as one observed transfer failure, not a general estimate across networks.

Strict pilot tasks have no eligible test rows and were not evaluated.
Keep this limitation and the failed strict edge result in the report.

## Reporting correction and next experiment

The original correlation helper could assign tiny Pearson values to
constant training-mean predictions due to floating-point centering.
Constant-predictor correlations and their bootstrap comparisons are
undefined and should be omitted. The helper now checks for constant
inputs explicitly; its regression test and the weighted-rank test pass.
This does not change any model/LOMAP/pilot comparison above. Do not rerun
training just to correct those baseline fields.

The supported conclusions are a strong known-reference structural
baseline and a stronger simulation-informed continuation predictor,
with a failed strict network transfer case. Freeze these results before
designing the next experiment. For allocation, define a budget and charge
pilot effort, then validate how uncertainty changes with added sampling;
correlation alone is not proof of resource savings.
