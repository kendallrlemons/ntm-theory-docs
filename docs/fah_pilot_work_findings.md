# Findings from the selected pilot-work audit

Inputs: `/Users/lemonsk/Downloads/fah_pilot_work_audit_v1`.
Corrected diagnostic outputs:
`/Users/lemonsk/Downloads/fah_pilot_work_review_v2`.

All four payload checksums match the cluster audit report. The selected
records contain 30 transformations, 360 phase-generation records, and
10,550 exported work pairs. Historical source values remain unchanged.

## Confirmed work-data anomaly

All ten selected generation-0 complex-phase cases have inverted work
supports under the forward/reverse convention that reproduces the control
estimates: every forward work is smaller than every sign-reversed reverse
work. The gap between supports is 547.06–968.86 in source work units.

Every paired forward-plus-reverse work value in those ten records is
negative. Mean forward-plus-reverse work is approximately -693 to -1046.
None of the other 350 audited phase-generation records has an inverted
support gap or negative mean forward-plus-reverse work. These are
descriptive sample findings, not a universal finite-sample validity test.

The anomaly is concentrated in generation-0 forward work. For example,
case 1022 has mean forward work -1086.40 in generation 0 and -105.71 in
generation 1, while mean reverse work is 126.34 and 125.82, respectively.

All ten complex cases have 30 exported pairs, a reported retained count
of 30, and the same 30 clone IDs present in generations 0 and 1. This does
not support explanations based simply on absent work or missing clones.
Reuse of clone IDs also does not establish independent replication.

The reported stderr of roughly 5e-8–7e-8 and reported overlap around
1e-14–1e-13 are not credible evidence of a well-resolved pilot estimate.
The exported work pattern and subsequent generation changes demonstrate
a serious pilot-data/estimation pathology. The upstream cause remains
unresolved: the audit cannot distinguish initialization, work accumulation,
export, or other historical processing problems without additional logs
and provenance. A corrected numerical root alone does not fix this.

## Correction to the audit diagnostic

The initial diagnostic used log-average logistic acceptance on both
sides. It handled near-zero acceptance but failed when both means rounded
to one. The inverted work gaps exposed that missing case. Consequently,
the original `diagnostic_delta_f` values for those ten records must not
be interpreted as trustworthy alternative estimates.

The corrected solver compares complementary logistic tails when both
acceptance averages exceed one half. This preserves the root and sign of
the equal-count BAR equation while resolving near-one saturation.
It also records log-complement averages. No uncertainty formula is
recomputed and no historical labels are replaced.

Seven synthetic tests now pass, including the newly added extreme
inverted-work regression. Independently, a direct logistic equation
evaluated with 400-digit Decimal arithmetic agrees with the corrected
solver within 1e-7 for case 1022 and control 3571:

| Record | Independent root | Corrected diagnostic |
|---|---:|---:|
| Case 1022 | -603.5170525673 | -603.5170525662 |
| Control 3571 | -35.9569745987 | -35.9569745988 |

Across the 350 other phase-generation records, corrected diagnostic
roots agree with the source point estimates to within 5.2e-10. For the
ten anomalous pilots, corrected roots differ from reported points by
about 11–88 source units. Their acceptance complements remain tiny
(log means approximately -277 to -488); these mathematical solutions
are not evidence that the corresponding sampling is reliable.

This supports a numerical-instability concern in the anomalous regime,
but does not identify the historical estimator version or prove which
internal code path ran. Exported work lists may already be filtered;
equal counts alone do not prove identical historical inputs.

## Implications

- Do not delete the ten edges or replace their pilot labels with the
  diagnostic roots. A point root is not a reliable uncertainty estimate.
- Preserve the prior benchmark/allocation results as demonstrations of
  what happens when the historical estimates are used without safeguards.
- The protected-allocation design must not interpret tiny pilot stderr
  as sufficient evidence to withhold sampling.
- A uniform floor can protect allocations without repairing these
  values, but its strength must be selected on validation data.
- Generation-0 overlap/work diagnostics are legitimate pilot information.
  The full-run reliability flag or generation-1 agreement cannot be used
  to decide a generation-0 allocation.
- Trace the generation-0 forward-work anomaly before claiming validated
  pilot pooling, corrected uncertainties, or actual simulation savings.

No new cluster audit is needed for these selected records: the corrected
diagnostics were computed locally from the checksummed selected JSON.
The complete source snapshot and historical logs would be needed for a
broader upstream investigation.
