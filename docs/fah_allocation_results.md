# Frozen-prediction allocation sensitivity study

Script: `living_network/evaluate_fah_allocation.py`.
Outputs: `/Users/lemonsk/Downloads/fah_allocation_v1/`.
The study uses the verified strict-campaign pilot predictions, without
refitting models, changing predictions, or removing poorly predicted edges.
It is a retrospective mathematical sensitivity study, not measured
simulation-time savings.

## Objective and cost model

For each phase separately, use the generation-1 proxy as the reference
spread `s_i` and evaluate mean production-stage variance:

```text
R = (1/N) sum_i s_i^2 / n_i
```

Here `n_i` is continuous production effort. The model assumes the
inverse-effort variance law and a constant production cost per effort
unit. Neither effective sample size nor this law has been established
for the original BAR/clone data.

Let `C` be the sum of generation-0 raw work-pair counts across the cohort.
One raw pilot work pair costs one proxy unit. Policies using pilot
information pay `C`; structure-only and uniform policies do not.
Each policy receives the same total budget `B`. Its production effort is:

```text
sum_i n_i = (B - pilot_cost_charged) / production_unit_cost
```

Pilot samples are charged but are **not included in the production
estimate**. This models a separate pilot/production workflow. It does not
model a pooled final estimate that reuses pilot work. Allowing such reuse
would require a justified pooling and dependence model, not simply adding
counts from separately filtered generations.

For predicted spreads `w_i`, allocation solves the constrained surrogate:

```text
minimize sum_i w_i^2 / n_i
subject to sum_i n_i = available effort and n_i >= minimum

solution: n_i = max(minimum, lambda * w_i)
```

The floor is enforced by water filling. Allocating proportional to
predicted variance instead of spread would optimize a different rule.

## Fixed comparisons and sensitivity grid

The protocol was written before running this allocation study and is
captured in `request.json`, including script and input hashes.

- Total budgets: `B/C = 1.25, 1.5, 2, 3, 5, 10`.
- Production unit costs: 1, 2, 4.
- Minimum production efforts: 1 and 30.
- Primary scenario: production cost 1, minimum effort 1.
- Primary cohort: all 4,684 complex or 4,659 solvent test records.
- Supplementary LOMAP cohort: 4,653 complex or 4,628 solvent records.
  Every policy is reevaluated on that same smaller cohort.

Policies include uniform, structure-only trees and hypersphere, direct
pilot, learned pilot-only, both structure-plus-pilot models, uniform after
paying for a pilot, and oracle references with/without pilot charges.
Oracles use the reference labels solely as nondeployable bounds.

The supplementary LOMAP policy uses the fixed weight
`max(1 - similarity, 1e-6)`. This is an explicit untuned allocation heuristic
based on LOMAP difficulty, not a claim that LOMAP defines an optimal
allocation policy or predicts calibrated spread.

Infeasible budget/floor combinations are recorded as infeasible, not
silently assigned more budget. The output contains 1,512 scenario/policy
rows. No confidence intervals are claimed for this single-campaign,
fixed-reference-proxy calculation.

## Equal total budget versus uniform without a pilot

Numbers are risk ratios: below 1 is better than uniform. These primary
scenario ratios include the pilot charge.

| Phase / policy | B = 3C | B = 5C | B = 10C |
|---|---:|---:|---:|
| Complex: direct pilot | 1.769 | 1.723 | 2.086 |
| Complex: learned pilot-only | 1.488 | 1.240 | 1.102 |
| Complex: trees + pilot | 1.508 | 1.256 | 1.117 |
| Complex: hypersphere + pilot | 1.752 | 1.709 | 2.073 |
| Solvent: direct pilot | 1.349 | 1.124 | 0.999 |
| Solvent: learned pilot-only | 1.330 | 1.108 | 0.985 |
| Solvent: trees + pilot | 1.336 | 1.113 | 0.989 |
| Solvent: hypersphere + pilot | 1.324 | 1.103 | 0.981 |

Solvent hypersphere saves about 1.9% of the variance proxy at `B=10C`,
compared with spending the entire same budget uniformly without a pilot.
Its floor-free break-even budget is approximately `8.52C`. The learned
pilot-only threshold is `8.82C`, and direct pilot `9.92C`. These are
analytic screens; active floors can change the crossover.

Complex pilot policies do not beat uniform anywhere on the primary
budget grid. Structure-only trees/hypersphere also do not beat uniform
in either phase on that primary grid.

On the separate common-LOMAP cohort, the fixed LOMAP allocation ratio
at `B=10C` is 0.9932 for complex and 0.9823 for solvent. Those values must
not be compared directly with different-cohort primary ratios.

## If a pilot is already required for every policy

The uniform-after-pilot control pays exactly the same `C` as learned pilot
policies. In the primary solvent scenario, hypersphere has a risk ratio
of approximately 0.8826 relative to that control: an 11.7% improvement in
production-stage variance proxy. Learned pilot-only gives 11.3%.

This is useful for a workflow that already requires a pilot, but it
answers a different question from whether purchasing the pilot is worth
its cost. Both comparisons charge the pilot; only the available baseline
differs. Neither comparison includes pilot work in the final estimator.

## Why complex allocation fails despite better prediction MSE

Ten complex edges, about 0.21% of the cohort, have direct pilot proxies
between 2.85e-7 and 3.71e-7, hypersphere predictions around 1.3e-5–1.7e-5,
and generation-1 reference proxies between 3.05 and 7.18.

The policies consequently allocate those edges at the minimum production
effort. Variance penalizes underallocation through division by `n_i`;
overall correlation or MSE can obscure this small but important set.
All ten edges remain in the study.

The ratio to uniform can grow with budget because these edges remain at
the floor while uniform allocates them more effort. Absolute policy risk
need not grow for its ratio to uniform to worsen.

The predeclared minimum-effort sensitivity demonstrates this dependence.
At `B=10C`, production unit cost 1, changing the floor from 1 to 30 moves
complex hypersphere's risk ratio from 2.073 to 1.109. That is less harmful
but still worse than uniform. It is not a newly selected or tuned policy.

The learned pilot-only predictor is more conservative on these edges
(approximately 0.673), but still underestimates their reference spreads.
It is more robust than hypersphere in this allocation comparison even
though its overall prediction MSE was slightly higher.

## Limited headroom and pilot overhead

Without effort floors, the perfect no-cost oracle's ratio to uniform is:

```text
(mean(s))^2 / mean(s^2)
```

That ratio is 0.9125 for complex and 0.8389 for solvent. Thus even perfect
knowledge can reduce the variance proxy by only about 8.7% and 16.1%,
respectively, before considering pilot overhead.

If that oracle also pays the pilot charge, it cannot beat uniform below
approximately `11.43C` for complex or `6.21C` for solvent in the
floor-free model. This explains why paying for an entire early generation
can erase benefits even with strong prediction correlations.

For arbitrary positive weights, the recorded floor-free diagnostic is:

```text
A = sum(w) * sum(s^2/w) / (N * sum(s^2))
pilot-policy / uniform risk = A * B/(B-C)
break-even B/C = 1/(1-A), only if A < 1
```

These diagnostic values are not deployed allocations without floors.
For complex direct-pilot and hypersphere weights, near-zero estimates
make the unconstrained diagnostic extremely poor.

## Validation and reproducibility

Nine synthetic checks passed: proportional solutions, active floors,
global-scale invariance, infeasible budgets, exact minimum budgets,
cost conservation, uniform pilot overhead, analytic break-even, and the
oracle bound, including invalid-input rejection.

The actual run checks all prediction hashes against the prior verified
review, aligns edge IDs and structures, rejects nonfinite/nonpositive
required inputs, verifies every feasible allocation spends the stated
budget, and asserts no policy beats the no-cost oracle.

```bash
python living_network/evaluate_fah_allocation.py \
  --results_root /Users/lemonsk/Downloads/strict_network \
  --verified_review /Users/lemonsk/Downloads/strict_campaign_pilot_review.json \
  --output_dir /Users/lemonsk/Downloads/fah_allocation_v1
```

The existing output directory is frozen. Use a fresh output path to
reproduce the calculation. It requires only NumPy/pandas and the saved
CSV/review artifacts; no GPU, model loading, LOMAP rescoring, or new MD.

Outputs include `request.json`, `primary_results.csv`,
`allocation_results.csv`, `floor_free_diagnostics.json`, and per-phase
`*_policy_inputs.csv` snapshots sufficient to reproduce every allocation.

## Next scientific step

These are variance-proxy comparisons under an assumed law, not GPU-time
savings or validated network-level uncertainty reductions. Real pilot
overhead, clone dependence, effective sample size, and any reuse of pilot
work still need validation.

The next policy design should protect against severe underestimates and
be selected on training/validation campaigns. The current log-target
regressors were not trained to optimize allocation risk. Under the assumed
objective, a Bayes-optimal weight is
`sqrt(E[s^2 | available information])`, which need not equal an
exponentiated log-target prediction. Comparing such risk-oriented weights
and a conservative allocation floor is a new experiment, not grounds
to tune the present strict-test results until they improve.
