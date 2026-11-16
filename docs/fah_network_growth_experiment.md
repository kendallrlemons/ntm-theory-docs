# Sequential growth of an established alchemical network

## Primary question

Given an established ligand network and its completed calculations, how
should incoming ligands be connected and sampled as information arrives?
The primary endpoint is the quality of ligand free-energy estimates at
equal total incremental simulation cost. Edge prediction correlation is
a supporting measurement, not the final objective.

This refocuses the next experiment on the known-reference use case.
Strict-reference transfer and the pilot-work anomaly remain relevant
stress tests; neither replaces the network-growth experiment.

## Readiness finding

Inspection of the existing `edges.csv` shows one fixed compound-structure
endpoint within each of the 25 source analyses. These are star networks.
There are repeated records and microstates, but they do not supply
observations for arbitrary alternative compound connections.

Consequently, existing data can support a synthetic fixed-reference
growth replay, subject to assumptions about sampling. They cannot
establish whether connecting a new ligand through an existing variable
ligand would outperform connecting it to the original reference.

Do not invent outcomes for missing edges or pool different receptor
variants/snapshots into a fictitious dense campaign. A model-generated
outcome used to evaluate that model's own connection choice is not
independent evidence.

## Implemented preparation

`living_network/prepare_fah_growth.py` consumes the original checksummed
schema-v2 known-reference benchmark on the cluster. It creates:

- `initial_edges.csv`: all training edges, scoped to their source snapshots.
- `arrival_schedule.csv`: five batches of whole held-out variable scaffold
  groups, with three fixed hash-based arrival orders (seeds 11, 23, 37).
- `choice_coverage.csv`: currently present nodes, incoming nodes, possible
  structural connections, observed connections, and missing alternatives.
- `growth_manifest.json`: frozen preparation protocol, hashes, and an
  explicit routing-evaluation readiness decision.

Validation and test are separate replay worlds, each starting from the
training network. Validation ligands are not silently inserted into the
test starting network. A group's ligands arrive together, and a candidate
partner must already be present before the batch. Within-batch connections
are outside this first protocol.

Only identity, chemistry, source and split fields enter preparation.
Future uncertainty, free energies, reliability, overlap and prediction
scores do not determine the schedule. The complete schedule is an
experiment-controller artifact; a future policy must only see its current
and previous batches.

The initial training network represents work already available before
growth; it is not counted again as new simulation expenditure. Frozen
models may use training-network information. An apparent growth trial
starting from only a few edges while its predictor had already seen the
remaining hidden campaign outcomes would not be an honest small-seed
experiment.

These are synthetic arrival orders, not reconstructed historical dates.
They are also scaffold-held-out stress tests; the entire Moonshot collection
must not be described as a single congeneric series. Campaign-specific
chemistry and receptor preparation still require review before a
prospective panel is chosen.

## Execution

Copy `prepare_fah_growth.py` and `prepare_fah_growth.sbatch` into the existing
cluster script directory and submit:

```bash
sbatch prepare_fah_growth.sbatch
```

This is a one-CPU preparation job using pandas; no GPU training, new
simulation, source download, LOMAP scoring or graph preparation is required.
It preserves the original benchmark and requires a fresh output directory:

```text
/data1/choderaj/lemonsk/FEC_Sandbox/fah_growth_v1/
```

Share `growth_manifest.json` and `choice_coverage.csv` first. They establish
the source-specific sizes and missing measurements before selecting a
tractable simulation panel. Preserve the initial-edge and arrival files.

Five synthetic checks passed: star-network availability, validation/test
separation, label and input-order invariance, whole-group arrivals and
repeated-record handling, split-leakage rejection, and insufficient
coverage reporting.

## Experiment after preparation

The next measurement stage needs a campaign-consistent panel with outcomes
for multiple possible connections per new ligand. Use available training
chemistry, source metadata, mapping feasibility and compute limits to
select the panel; do not pick it for favorable test prediction metrics.
Missing alternative edges require new simulations or an independently
validated source of equivalent calculations.

The planned comparisons are:

1. Fixed-reference expansion with uniform incremental sampling.
2. LOMAP-based connection selection with the same total incremental budget.
3. Learned connection selection using existing network uncertainty and
   predicted transformation uncertainty.
4. A simpler pilot-informed adaptive policy, with protection against
   unreliable early uncertainty estimates.

Mapping/charge/pose feasibility constraints must be shared across policies.
Connection to an uncertain existing ligand can transmit that uncertainty;
choosing the lowest predicted edge uncertainty alone is not the full
network objective. Any redundant edges or cycle checks consume the same
accounted budget. Complex and solvent costs must be tracked explicitly.

At each arrival round, record:

- Ligand free-energy error against a suitable independent reference.
- Uncertainty calibration and coverage, not just reported precision.
- Connected fraction of new ligands and failure/recovery behavior.
- Cumulative and per-arrival simulation cost, including pilots and retries.
- Whether accumulated evidence changes connection or sampling decisions.

Independent high-effort simulations can provide a sampling-focused
reference; experimental affinities additionally include force-field and
experimental error. Do not conflate the two. A later generation sharing
clone histories is not automatically an independent reference.

The exact simulation budget, candidate panel, scoring implementations and
validation-selection rules must be frozen before the new outcomes are
examined. This preparation does not claim those simulations or a complete
growth-policy evaluation have already run.

## Existing limitations remain

The earlier strict test has informed development and cannot be presented
as untouched confirmation of a new growth policy. The anomalous pilot
work records are retained; no corrected uncertainties or valid pooling
rule have been established. Any preliminary allocation replay using
`variance=s^2/n` remains explicitly assumption-based.
