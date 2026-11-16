# Growth readiness and a small validation measurement panel

Verified growth outputs:
`/Users/lemonsk/Downloads/fah_growth_v1`.
Panel outputs:
`/Users/lemonsk/Downloads/fah_growth_validation_panel_v1`.

## Verified readiness

All three payload hashes match the growth manifest. The initial networks
contain 22,686 edge records across 23 source snapshots. Of these sources,
22 support the specified five-batch schedule in both validation and test.
The remaining sprint-11A charged/restrained source has only two arrival
groups in each split and is reported as unsupported.

There are 660 ready batch contexts:
22 sources × 2 replay splits × 3 orders × 5 batches.
The schedules contain 46,290 rows representing 15,430 distinct held-out
edge records repeated across three arrival orders. These are not
46,290 independent measurements or 22 independent reference families.

Verification checked whole-group arrivals, training/arrival separation,
validation/test group separation, no repeated structures across batches,
equal cohorts across arrival seeds, prior-node counts, and observed/missing
candidate-pair counts.

Every arriving structure has just its reference connection observed.
None of the 660 contexts offers an observed alternative anchor. The
preparation succeeded, but connection-choice performance cannot be
evaluated retrospectively from these stars.

## Proposed development panel

The preparation script `living_network/prepare_fah_growth_panel.py`
produces a deliberately small, validation-only panel from:

`sprint-5-retrospective-x11498-dimer-neutral`

Source SHA:
`2fed73582cebfa4c2543722c371ffbf1badf0d37cc9575eed8ab08a15fa94135`.

This source has 57 initial training-edge records and 58 modeled nodes
including the reference. Its complete growth schedule contains 19 new
validation structures and 28 new test structures. Source choice was based
on tractability and the dimer-neutral context, not prediction performance.

The proposed subset has six validation structures, two at each of the
first three original arrival steps. It selects the first six
charge-compatible groups in the frozen seed-11 order and one hash-selected
structure per group. Unselected arrivals are not silently added to this
smaller panel's history.

Each new structure has three candidate connections:

1. The historical reference.
2. The most Morgan-similar same-charge previously present variable structure.
3. The second most similar such structure.

The result is 18 distinct structural pairs: six historical reference
connections and 12 alternative connections without existing observations.
One candidate uses an earlier arrival as its anchor, so the panel includes
the defining growth decision rather than only attachment to original nodes.

Morgan screening uses radius 2, 2048 bits and chirality, with the RDKit
version recorded. It is not LOMAP scoring, atom mapping, pose validation,
or evidence that these alternatives are easier. All eventual policies
must use the same feasible candidate pool.

The six structures are from validation, not test. The panel is for
development and feasibility, not untouched confirmation. The selected
campaign/structures still require chemical-series review; the source name
alone does not establish a strict congeneric series.

## Checks and reproducibility

All 18 pairs were checked for uniqueness and valid arrival identity.
Every anchor is an initial node or a selected structure from an earlier
batch. No future or same-batch anchor appears. No uncertainty, free-energy
outcome or prediction metric enters candidate selection.

```bash
python living_network/prepare_fah_growth_panel.py \
  --growth_dir /Users/lemonsk/Downloads/fah_growth_v1 \
  --output_dir /Users/lemonsk/Downloads/fah_growth_validation_panel_v1
```

The existing output is frozen; reruns require a fresh output directory.
`candidate_transformations.csv` contains exact structures, group and arrival
identity, candidate anchors and historical reference-edge IDs.
`panel_manifest.json` records the input/code/output hashes and limitations.

## What must be resolved before MD

This is a structural measurement proposal, not a runnable simulation
package. Next requirements are the simulation engine/workflow, prepared
Mpro receptor and ligand poses, microstate/protonation identity, atom-mapping
feasibility, and an agreed sampling/replicate budget.

There are 36 phase calculations per replicate if all 18 pairs are run
in complex and solvent. This is only a count of phase calculations, not
a complete cost estimate. Initialization, equilibration, independent
reference measurements, retries and any required initial-network
remeasurement add cost.

Historical reference results can only be reused if their simulation
protocol and physical state are compatible. A new protocol may require
rerunning the reference controls and required initial edges; one must not
combine incompatible free energies into a network.

The uniform baseline, LOMAP and learned routing comparisons are not yet
evaluated. Simulation costs, candidate feasibility and policy-selection
rules must be fixed before examining the new alternative-edge outcomes.
Pilot reliability safeguards and protection against underallocation remain
part of the planned adaptive comparison.
