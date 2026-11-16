# Pilot-informed strict-campaign results

Artifacts: `/Users/lemonsk/Downloads/strict_network`.
Verification snapshot: `/Users/lemonsk/Downloads/strict_campaign_pilot_review.json`.
Both baseline tasks and all four hypersphere runs completed. No test
records were lost to graph construction.

The review reproduced correlations and MSE from saved predictions,
matched test IDs/structures/groups to the original strict edge export,
and checked generation-0 inputs and generation-1 targets against the
original `generations.csv`. It verified all four checkpoint file hashes,
the referenced baseline artifacts, request identities, and the best
validation epoch in each training history. No fitting was repeated.

## All eligible test records

These are generation-1 phase stderr times square root of the retained
work count, not pooled edge stderr. Complex has 4,684 test records;
solvent has 4,659. Values below use all eligible records, irrespective
of whether LOMAP returned a score.

| Phase | Method | Pearson | Spearman | MSE |
|---|---|---:|---:|---:|
| Complex | Structure-only trees | -0.0768 | -0.0501 | 3.44381 |
| Complex | Structure-only hypersphere | 0.0379 | 0.0344 | 2.61979 |
| Complex | Direct pilot | 0.5206 | 0.5316 | 2.79391 |
| Complex | Learned pilot-only | 0.5187 | 0.5303 | 1.39141 |
| Complex | Structure + pilot trees | 0.5173 | 0.5310 | 1.83180 |
| Complex | Structure + pilot hypersphere | 0.5026 | 0.5365 | 1.31173 |
| Solvent | Structure-only trees | 0.0721 | 0.0431 | 1.22970 |
| Solvent | Structure-only hypersphere | 0.1566 | 0.1455 | 1.69802 |
| Solvent | Direct pilot | 0.7951 | 0.8492 | 0.89036 |
| Solvent | Learned pilot-only | 0.8018 | 0.8489 | 0.46327 |
| Solvent | Structure + pilot trees | 0.7888 | 0.8477 | 0.78015 |
| Solvent | Structure + pilot hypersphere | 0.8107 | 0.8513 | 0.39152 |

Pilot information supplies useful signal in this held-out campaign.
Structure-only transfer remains weak on the same phase-proxy targets.
Most ranking performance is already supplied by the direct pilot.

The learned pilot-only model reduces MSE relative to direct pilot use by
50.2% for complex and 48.0% for solvent, with little change in Spearman.
This is consistent with improved numerical calibration rather than a
large improvement in ordering edges. Those percentages are descriptive
point estimates, not bootstrap intervals.

Hypersphere plus pilot has 5.7% lower complex MSE and 15.5% lower solvent
MSE than learned pilot-only, again point estimates without MSE intervals.
These compare different model classes and cannot by themselves isolate
the contribution of structural inputs.

## Incremental rank/correlation evidence

Supplied paired group-bootstrap 95% intervals, hypersphere plus pilot
minus learned pilot-only:

| Phase | Pearson difference interval | Spearman difference interval |
|---|---|---|
| Complex | [-0.02648, -0.00630] | [-0.00102, 0.01392] |
| Solvent | [0.00357, 0.01389] | [-0.00006, 0.00471] |

Complex Pearson is worse for the hypersphere model despite its lower
MSE. Solvent Pearson improves modestly. Neither phase establishes a
positive Spearman increment over learned pilot-only under these
intervals. Solvent hypersphere-versus-direct-pilot Spearman also has
an interval including zero.

For trees, adding structure to the learned pilot-only model does not
establish a correlation improvement in complex; in solvent, Pearson is
worse. Its MSE is also higher in both phases.

The strict endpoint bootstrap has only 621 valid complex replicates
and 641 valid solvent replicates out of 1,000. Resamples excluding the
sole held-out reference have no evaluable edges. These unadjusted
intervals cannot estimate variability across unseen reference families,
and do not include training-seed variability.

## Training and selection

Hypersphere best epochs were 5/9 for complex structure/structure-plus-pilot,
and 5/5 for solvent. Histories ran for 15/19 and 15/15 epochs, respectively:
the early selected epochs are consistent with the fixed patience rule.

Validation Spearman for complex structure-plus-pilot is 0.74697 for trees
and 0.74506 for hypersphere. Solvent values are 0.75556 and 0.76630.
Thus a family choice by validation Spearman would favor trees for
complex and hypersphere for solvent. Do not choose a new primary policy
by the lowest observed test MSE and call that selection held out.

## Meaning for the next experiment

These results support collecting pilot information when moving to a new
campaign. They do not establish general structure-only transfer, nor
that structural embeddings are the primary source of pilot-informed
ranking accuracy.

The next allocation comparison should retain direct pilot, learned
pilot-only, both structure-plus-pilot models, and the predefined
structure-only/uniform controls. Charge pilot cost at equal total
budgets and keep complex and solvent phases separate.

Improved MSE is not proof of improved allocation. For example, multiplying
every predicted spread by a common positive constant changes calibration
but leaves allocations proportional to predicted spread unchanged.
Actual allocation losses must be evaluated, with the assumed
variance-versus-effort law stated explicitly.

The complex phase remains the main predictive limitation. More early
sampling and campaign-specific adaptation are candidates for subsequent
experiments; neither is established as a remedy by these results.
