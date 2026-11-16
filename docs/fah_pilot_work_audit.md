# Audit of near-zero pilot uncertainty and protected-allocation plan

## Confirmed from exported generation summaries

The first allocation study identified ten strict-campaign complex edges
with hypersphere-plus-pilot estimates below 0.01. Their identities are
frozen in `living_network/fah_low_overlap_audit_cases.json`. This is a
post-hoc diagnostic selection, not a proposed data exclusion or feature.

For these ten generation-0 complex estimates:

| Quantity | Observed range |
|---|---|
| Reported stderr | 5.2021e-8 to 6.7776e-8 |
| Reported BAR overlap | 2.6645e-14 to 8.9928e-14 |
| Exported/retained work-pair counts | 30 / 30 for every case |
| Generation-1 BAR overlap | 0.01429 to 0.14793 |
| Generation-1 minus generation-0 free-energy point | 328.70 to 565.75, in source units |

Among the other 4,674 generation-0 complex records in this source, the
smallest reported stderr is 0.22848 and the smallest overlap is
1.5141e-10. These are descriptive comparisons, not thresholds selected
for a future allocator.

The extremely small reported errors occur alongside essentially zero
reported overlap and large subsequent changes. They should not be
interpreted as evidence of reliable sampling. A numerical or estimator
failure is a leading hypothesis, but the summaries alone do not
establish the mechanism or distinguish a failed numerical solution from
genuinely poor overlap in the work distributions.

## Source required

The relevant sprint-4 snapshot is on the cluster:

```text
/data1/choderaj/lemonsk/FEC_Sandbox/moonshot/dashboards/sprint-4/2020-09-06-ugi-tBu-x3110-3v3m-2020-04-Jacobs/analysis.json
```

Expected size: 152,080,619 bytes.
Expected SHA-256:
`86e26cd0531e1617efca3c97da766e1bfe6ecc37d9ab97a589a8c3baafded8e1`.

The local Moonshot source snapshots inspected are sprint 8 and sprint 12;
the sprint-4 raw-work audit has not been run locally. Its input structures
were checked using the existing extractor and a local source record.
The audit fails explicitly if the actual source has an incompatible
schema.

## Targeted audit

`audit_fah_pilot_work.py` streams the source and exports the ten cases plus
20 deterministic controls. Controls are noncase strict-test IDs selected
by a fixed SHA-256 ordering without using outcomes. Both phases and all
exported generations are retained for the selected transformations.

The audit checks source identity before and after processing and records:

- Forward/reverse work arrays and clone IDs, preserving pair order.
- Generation labels from `gen`, not the position in the JSON list.
- Reported estimates, retained counts, and overlap.
- Exported pair counts, nonfinite entries, unique/duplicate clone IDs,
  and directional work spread.
- Clone-ID intersection between generations 0 and 1.
- A separate stable equal-count BAR-equation diagnostic on exported works.

Exported generation work lists may already have been filtered by
Fah-XChem. They are not necessarily unfiltered trajectory work values,
nor do they necessarily reconstruct the historical pooled input.
Count agreement alone does not prove identical filtering.

The diagnostic assumes dimensionless work with the standard convention
`forward=W_AB`, `reverse=W_BA`. With equal forward/reverse counts, it
solves:

```text
mean(sigmoid(delta - W_forward))
  = mean(sigmoid(-delta - W_reverse))
```

It uses log-sigmoid/log-sum-exp evaluation and bracketed bisection to
avoid underflow plateaus. No outlier filtering is applied. Nonfinite or
empty work lists are reported without silently dropping samples.

This produces a mathematical root for the exported arrays, not a new
uncertainty estimate or proof of statistical convergence. Check agreement
with historical estimates on controls before interpreting disagreement
on cases. If reported input counts differ from the exported lists, that
also limits interpretation. No frozen labels or model outputs are changed.

Six tests passed, covering an analytic root, extreme-dissipation
underflow resistance, direction/offset invariance, count/clone reporting,
nonfinite handling, source identity, and end-to-end export integrity.

## Cluster execution

Copy these files into the existing cluster script directory:

- `living_network/audit_fah_pilot_work.py`
- `living_network/fah_low_overlap_audit_cases.json`
- `living_network/audit_fah_pilot_work.sbatch`

The existing `fah_edge_dataset.py` must remain alongside them. The script
uses its hash helper and requires `ijson`, already specified in
`requirements-fah.txt`. It needs one CPU and no GPU.

```bash
sbatch audit_fah_pilot_work.sbatch
```

It reads the already downloaded source; no download, training, or full
corpus extraction is necessary. Output directory:

```text
/data1/choderaj/lemonsk/FEC_Sandbox/fah_pilot_work_audit_v1/
```

Send this small output folder, containing:

- `report.json`
- `generation_diagnostics.csv`
- `work_pairs.csv`
- `clone_reuse.csv`
- `selected_transformations.json`

The selected JSON preserves source transformation records for inspecting
schema or estimator details without transferring the entire snapshot.

## Protected-allocation protocol, specified but not executed

The independent design choices are frozen in
`living_network/fah_protected_allocation_protocol.json`:

- Compare learned pilot-only and hypersphere weights.
- Guarantee each edge a fraction of the uniform production allocation.
  Candidate fractions are 0.25, 0.5, 0.75, and 1.0.
- For available production effort `R`, use the existing water-filling
  allocation with `floor=max(1, alpha*R/N)`.
- Include uniform allocation without purchasing a pilot as a fallback.
- Select predictor/fraction or fallback separately for each phase and
  budget using only the original validation campaign.
- Resolve numerical ties toward uniform, then the simpler predictor,
  and then stronger uniform protection.
- Preserve the existing strict campaign as retrospective evidence.
  An additional untouched campaign is required for confirmation.

Under the assumed inverse-effort law, an `alpha` uniform floor bounds
risk by `1/alpha` times uniform risk at the **same production effort**.
This is a protection bound, not a guarantee of improvement or a bound
that erases pilot overhead.

No overlap cutoff, exclusion of the ten cases, or per-edge repair is
specified. The raw-work findings must first establish whether the
historical inputs/labels are usable. This is an explicit implementation
gate, not a claim that the protected policy has been trained or evaluated.

The first protection ablation would retain current frozen predictors.
Training a predictor of `sqrt(E[s^2 | information])` is a subsequent
allocation-oriented experiment. Reusing pilot work in the final estimate
also remains unresolved: do not assume generation counts can simply be
added or separately filtered estimates averaged.
