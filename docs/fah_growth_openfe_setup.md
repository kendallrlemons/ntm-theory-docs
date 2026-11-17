# OpenFE/OpenMM setup for the growth validation panel

Status: **input structures validated; OpenFE transformations and MD not yet
tested.** Nothing here is a free-energy result.

## Public structures: what exists and what was used

Campaign: `sprint-5-retrospective-x11498-dimer-neutral`
(analysis SHA-256 `2fed7358…94135`, the frozen panel source).

| Public source | Finding | Use |
|---|---|---|
| Dashboard `transformations/RUN*/{old,new}_{ligand,protein}` | Post-simulation snapshots, only 34 of 136 runs, a different protein conformation per run; 9 of 16 non-reference panel nodes absent | Not used |
| `sprint-5-retrospective/setup/docked/<campaign>.sdf` | The campaign's prepared microstate poses: 137 records, explicit hydrogens, covers every simulated microstate | Ligand poses |
| `sprint-5-retrospective/setup/13442/RUNS/RUN*/hybrid_complex.pdb` | Relaxed perses setup complexes (protein + hybrid ligand + ions); some are translated by a periodic image | Receptor |
| `sprint-5-retrospective/setup/json/<campaign>.json` | Setup transformations; identical run/microstate triples to `analysis.json` | Identity check |

The unrelaxed docking receptor (`receptors/dimer/Mpro-x11498_0A_bound-protein.pdb`)
is referenced by the setup scripts but is not in the public bucket or the
`covid-moonshot` repository.

## Input bundle

`living_network/fetch_fah_growth_structures.py` downloads the pinned public
files (SHA-256 checked), validates them and writes a fresh bundle:

```bash
python living_network/fetch_fah_growth_structures.py \
  --panel_dir /Users/lemonsk/Downloads/fah_growth_validation_panel_v1 \
  --cache_dir /Users/lemonsk/Downloads/fah_growth_structures_v1/cache \
  --output_dir /Users/lemonsk/Downloads/fah_growth_structures_v1/bundle
```

Outputs: `ligands.sdf` (17 poses, unchanged coordinates and hydrogens),
`receptor.pdb`, `panel_edges.csv` (18 edges, anchor → new ligand),
`structure_validation.csv`, `bundle_manifest.json`.

Verified for all 17 ligands: one simulated microstate per panel structure;
isomeric identity against the panel and `analysis.json`; stereochemistry
perceived from 3D agrees; no unassigned stereocentres; all hydrogens explicit;
formal charge 0. The reference SMILES also occurs as `VLA-UCB-1dbca3b4-15_1`;
it is resolved to the campaign's initial microstate `MAT-POS-b3e365b9-1_1`.
The six historical reference edges match their run/microstate records.

Receptor: protein chains A and B (2 × 304 residues, 9,294 atoms, hydrogens
present) from the setup complex of **RUN323**. Rule: lowest run among the
panel's six historical reference edges whose setup complex is in the docking
frame. RUN281 is lower but is displaced by a periodic image (docked reference
56.7 Å from the protein), so it is rejected. His41 is neutral (HE2) and
Cys145 protonated in both chains, matching the campaign's
`His41(0) Cys145(0)` variant. Ligand and ions are removed.

### Pose/receptor clashes

The docked reference fits the receptor (minimum heavy-atom distance 2.26 Å,
no contacts below 2.2 Å). Eleven of 17 poses have a heavy-atom contact below
1.5 Å; the worst is `MAT-POS-9ff17035-3_1` at 0.24 Å. Upstream docking already
recorded nonzero `clash_score` for most of these, and the receptor is relaxed
around a different ligand. These values are diagnostics only; they were not
used to choose ligands or edges. They make minimization/equilibration
stability a required per-complex check.

## OpenFE setup

Environment (cluster, new; separate from `gnn_rbfe`):

```bash
mamba env create -f environment-openfe.yml   # openfe 1.13.0, python 3.12, cuda-version 12.0
```

`living_network/prepare_openfe_growth.py` verifies the bundle hashes, assigns
AM1-BCC charges (AmberTools) on the supplied poses, maps each edge with
`LomapAtomMapper(time=20, threed=True, max3d=1.0, element_change=True,
shift=False)`, and writes one complex and one solvent `Transformation` JSON
per mapped edge. Protocol: `RelativeHybridTopologyProtocol` defaults with
`protocol_repeats=1`, 300 K, CUDA, and 70 mM NaCl. `--kind smoke` shortens
sampling to 10 ps equilibration and 50 ps production; `--kind openfe_default`
keeps OpenFE default lengths. Edges without a mapping are reported as
`no_mapping`, not dropped silently.

## Cluster steps

Copy to `/data1/choderaj/lemonsk/FEC_Sandbox`:
`prepare_openfe_growth.py`, `prepare_openfe_growth.sbatch`,
`run_openfe_growth_smoke.sbatch`, `environment-openfe.yml`, and the bundle
directory as `fah_growth_structures_v1/bundle/`.

```bash
mamba env create -f environment-openfe.yml
sbatch prepare_openfe_growth.sbatch        # 1 CPU; writes fah_growth_openfe_smoke_v1/
sbatch run_openfe_growth_smoke.sbatch      # after the first job; 4 GPU array tasks
```

Both scripts set `PYTHONNOUSERSITE=1`: a NumPy 2.1.1 in `~/.local` otherwise
shadows the environment's NumPy 1.26.4 and breaks the OpenMM 8.1.2 import.

The smoke array runs complex and solvent legs for two edges listed in
`smoke_tasks.txt`: the first reference edge with unflagged poses, and the
first edge involving the worst-clashing pose. Return `mappings.csv`,
`openfe_manifest.json`, the `.out/.err` logs and `smoke_results/*/result.json`.

## Smoke test (2026-11-16)

Environment as resolved on the cluster: openfe 1.13.0, gufe 1.13.1,
OpenMM 8.1.2, NumPy 1.26.4, lomap2 3.3.0. All 18 edges were mapped and
written. The four smoke tasks (complex and solvent legs of
`MAT-POS-b3e365b9-1_1 → LON-WEI-4d77710c-9_1` and
`MAT-POS-9ff17035-3_1 → ALP-POS-f13221e1-2_1`) completed setup, simulation
and analysis without failed units, including the pose with a 0.24 Å receptor
contact. Complex legs took 430–482 s of simulation; solvent legs 41–52 s.

MBAR overlap scalars were 0.0000–0.0046. The structural analysis reported
`index 0 is out of bounds` in all four; the cause is unconfirmed. The
reported leg free energies come from 50 ps and are not results. SLURM logs
were not reviewed.

## Mapping comparison (2026-11-17)

`compare_openfe_mappings.py` output:
`/Users/lemonsk/Downloads/fah_growth_mapping_comparison_v1`
(CSV SHA-256 `45124cb5…28f0209`). Mapped heavy atoms per edge:

| Mapper | Mean | Range | Mean LOMAP score |
|---|---|---|---|
| LOMAP, `max3d=1.0` (used for smoke) | 14.2 | 13–18 | 0.058 |
| Kartograf defaults | 14.2 | 13–18 | 0.058 |
| LOMAP, no 3D cutoff | 18.2 | 13–27 | 0.236 |

- Kartograf and 3D-LOMAP map the same number of heavy atoms on every edge.
- Removing the 3D cutoff enlarges the mapping on 13 of 18 edges. The three
  `new_05` edges go from 13 to 24 heavy atoms (all heavy atoms; score
  0.71–0.74), and `ALP-POS-f13221e1-3_1 → new_02` from 15 to 22 (score 0.95).
  The docked poses therefore place chemically shared atoms more than about
  1 Å apart.
- Five of six reference edges map 13 heavy atoms under every mapper; the
  exception is `new_05`.
- Without the cutoff, an alternative anchor scores above the reference for
  `new_02` (0.95 vs 0.07) and `new_04` (0.43, 0.16 vs 0.02). For `new_01`,
  `new_03` and `new_06` every option scores below 0.1.
- Two 3D-LOMAP scores differ between the preparation and comparison runs
  (0.055 vs 0.037; 0.067 vs 0.045) with equal mapped-atom counts; unexplained.

No mapper has been selected. A no-3D mapping combined with the current poses
would join atoms whose coordinates disagree, so pose handling must be decided
with the mapper.

## Not yet established

- `cuda-version=12.0` resolved OpenMM 8.1.2, an older build.
- This protocol (OpenFE replica exchange, OpenFF default force field, AM1-BCC)
  differs from the historical perses nonequilibrium Folding@home protocol
  (openff-1.3.0). Historical free energies and uncertainties must not be
  combined with new results; required initial edges need remeasurement.
- Sampling budgets, repeats, the common candidate pool, and policy-selection
  rules must be frozen before any production outcome is examined.
- Whether pilot samples can contribute to final estimates is unresolved.
