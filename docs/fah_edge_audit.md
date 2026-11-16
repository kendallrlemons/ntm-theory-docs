# Fah-XChem edge-table audit

This is a descriptive audit, not a trained-model evaluation. The supplied
`edges.csv` and `generations.csv` match the SHA-256 checksums in
`manifest_d.json` (manifest SHA-256
`cfb82a2d3890dc0a005a74410ae0a8d5d458a63e161856b967759a2657909412`).
The manifest records extraction schema 2. Its loader hash differs from the
current local loader; code-byte identity remains unresolved. The table
identities themselves are verified.

## Coverage and identity

- 25 analyses contain 48,483 transformation records.
- 45,662 records have resolved endpoint microstate structures, parseable
  RDKit SMILES, and finite nonnegative edge stderr.
- All 48,483 edge stderrs are finite and strictly positive. Every edge
  stderr agrees with the quadrature sum of the two phase stderrs at
  relative tolerance 1e-8 and absolute tolerance 1e-10.
- The resolved records contain 29,954 unordered pairs of canonical
  isomeric microstate SMILES (RDKit 2024.03.5). This is not a count of
  independent simulations. There are 64 records with identical canonical
  endpoint structures.
- 5,991 structure pairs occur in multiple source analyses.
- No redundant resolved record has exactly the same unordered structure
  pair and all eight phase summary fields (point, stderr, work count,
  overlap for both phases). This does not rule out shared underlying work
  samples, partial reuse, or duplicates with differing analysis estimates.
- 25,146 records have a true reliability flag; 18,653 have false, and
  4,684 have missing flags. All missing flags are in sprint 4. A missing
  flag must not be interpreted as a failed reliability check.

The 2,821 unresolved records are in sprint 12 (1,017 and 1,087 in its two
analyses), sprint 7 (705), and sprint-5-dimer (12).

Shared FAH project IDs do not establish duplicated simulation runs. Most
variants with shared project IDs use disjoint run ranges. Sprint 7 and
sprint 8 are an exception: 3,045 project/run keys overlap, but none match
resolved unordered structure pairs. Of those keys, 2,340 have conflicting
resolved structure pairs; the other 705 include unresolved sprint-7
endpoints. Keep source identity in all joins and investigate upstream
provenance before treating a project/run key as globally unique.

## Generation-level signal

For records with exactly two distinct generations in a phase, compare the
earlier generation's estimate against the later generation's estimate.
These comparisons use resolved edges and no model fitting. They do not
compare a generation with a pooled estimate that includes that generation.
The following results use `stderr * sqrt(num_work_pairs)` for each
generation separately:

| Analysis | Phase | Records | Pearson | Spearman |
|---|---|---:|---:|---:|
| Sprint 8 | Complex | 5,077 | 0.8312 | 0.8614 |
| Sprint 8 | Solvent | 5,077 | 0.9119 | 0.9439 |
| Sprint 7 | Complex | 2,340 | 0.7831 | 0.7793 |
| Sprint 10 restrained | Complex | 771 | 0.7024 | 0.7534 |
| Sprint 10 unrestrained | Complex | 763 | 0.7082 | 0.7451 |

The complete per-source table is emitted by `audit_fah_tables.py`; other
two-generation complex-phase correlations are lower (Spearman about
0.576–0.701 for the three eligible sprint-5 analyses). The listed
correlations are descriptive point estimates without grouped confidence
intervals, not final held-out benchmark results.

The sprint-8 raw-stderr correlations are also strong: complex Pearson
0.8097 / Spearman 0.8352, solvent Pearson 0.9111 / Spearman 0.9441.
Thus the generation-level relationship is not solely introduced by
multiplication by the square root of work count.

An additional direct inspection of the locally downloaded sprint-8 JSON
found generation labels 0 and 1 in both phases. All 5,077 transformations
share at least one clone ID between generations in each phase. Across
399,595 complex and 394,789 solvent shared-clone comparisons, no
forward/reverse work-value pair was exactly repeated. The work values
differ, but shared clone histories mean these generations cannot be
assumed to be independent replicate simulations.

## Sampling-count limits

The exported raw work counts agree with the sum of generation raw work
counts for every phase record. However, summing the generation-level
`num_work_pairs` does not always recover the pooled phase count:

- Complex phase: 9,360 of 48,483 records differ.
- Solvent phase: 1,010 of 48,483 records differ.
- Sprint 8 alone: 1,600 complex and 288 solvent records differ.

The cause requires upstream analysis-code inspection; filtering is a
hypothesis, not established by this audit. Do not silently replace one
count with another or interpret either as an effective independent
sample size. The effort-normalized quantity remains a proxy, not a
validated intrinsic difficulty or calibrated allocation coefficient.

## Next benchmark decisions

1. Separate structure-only prediction from prediction after pilot
   simulations. Pilot information is legitimate for adaptive allocation,
   but its simulation cost must be included.
2. Keep repeated chemistry across dashboards in the same split. For
   unseen-scaffold evaluation, group endpoint chemistry globally before
   splitting, rather than splitting dashboards independently.
3. Start with resolved structures and report self transformations
   separately. Report reliability-stratified results while preserving
   missing flags as a distinct category.
4. For pilot prediction, use earlier-generation features and a later
   generation target. Resolve temporal semantics, clone dependence, and
   count filtering before interpreting it as a prospective experiment.
5. Compare corrected LOMAP and structural models on the actual edge
   uncertainty target. Add pilot-based methods as a separate information
   tier, with grouped uncertainty intervals and explicit effort accounting.

Reproduce from `living_network` using an environment with pandas, NumPy,
SciPy, and RDKit:

```bash
python audit_fah_tables.py \
  --manifest /Users/lemonsk/Downloads/manifest_d.json \
  --edges /Users/lemonsk/Downloads/edges.csv \
  --generations /Users/lemonsk/Downloads/generations.csv \
  --output_dir /Users/lemonsk/Downloads/fah_audit_reproduction
```

The frozen audit outputs for this run are in
`/Users/lemonsk/Downloads/fah_audit_v2/`. The raw clone comparison above was
a separate direct source inspection, not part of the CSV audit script.
