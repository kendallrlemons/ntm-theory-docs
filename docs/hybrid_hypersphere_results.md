
# Hybrid + Hypersphere Model: Results, Methodology, and the Ligand-B Leakage Issue

*Status: all numerical results below are historical reports, not provenance-verified results. Earlier data-frame split fixes did not validate legacy graph-cache identity/order or checkpoint provenance. Treat all prior scores as provisional until the pipelines are rerun with fresh, versioned artifacts; this document preserves the historical tables without endorsing their rankings or conclusions.*

---

## 1. Background

Two encoder approaches were evaluated before this comparison:

| Script | Approach |
|---|---|
| `living_network/hybrid_topo_rbfe.py` | Merges ligand A + ligand B into **one** hybrid-topology graph via MCS (atoms tagged core / A-unique / B-unique), encodes with an MPNN → single embedding per pair → regression head predicts `\|SE\|`. Transformation-aware, flat Euclidean embedding, pure MSE loss. |
| `living_network/hypersphere_rbfe.py` | Encodes ligand A and ligand B **independently** (no knowledge of the transformation partner) with an MPNN, L2-normalizes each onto the unit hypersphere, trains with a 3-component loss (continuous contrastive + dispersion + regression). |

`living_network/hybrid_hypersphere_rbfe.py` combines both ideas: it reuses the hybrid-topology encoder's single transformation-aware embedding (via `HybridGNN.encode()`), L2-normalizes it onto the hypersphere, and trains two variants:

- **Phase 2 (ablation)** — hybrid encoder + L2-norm + regression-only loss. Isolates whether hyperspherical geometry *alone*, with no contrastive/dispersion pressure, helps over the flat-Euclidean baseline.
- **Phase 3 (combined)** — hybrid encoder + L2-norm + full 3-component loss.

Because `HybridGNN` produces only **one** embedding per pair (not two independent per-ligand embeddings), the original hypersphere contrastive loss — which compares `h_a` vs `h_b` *within* a pair — doesn't apply. It was reformulated as `batch_pairwise_contrastive_loss`: transformations `i, j` with similar observed `|SE|` labels (`|SE_i - SE_j|` small) should embed close together on the sphere *across* the batch; dissimilar-label transformations should be far apart. Dispersion is a detached diagnostic with no encoder gradient.

Phase 5 runs a six-way comparison on one fixed test sample: Tanimoto, real LOMAP, the independent hypersphere model, the independent MCS/GNN model, the hybrid+L2 ablation, and the hybrid+hypersphere combined model.

---

## 2. LOMAP baseline: from heuristic proxy to the real algorithm

The first version of the LOMAP baseline (`_lomap_like_score`) was a hand-rolled proxy: `1 - 0.5*(mcs_coverage + mw_similarity)`, using RDKit's generic `rdFMCS` search and molecular-weight ratio as a stand-in for LOMAP's actual rule-based penalties. Under scrutiny, this was found to diverge substantially from the real LOMAP algorithm, which uses exponential-decay rule terms (MCSR, MNACR) plus explicit penalties for ring-breaking, ring-size changes, hybridization changes, and charge changes — none of which the proxy captured.

This was replaced with the **real LOMAP algorithm**, via the actual `lomap` + `gufe` packages:

```python
comp_a = gufe.SmallMoleculeComponent(rdkit=ma)   # 3D-embedded RDKit mol
comp_b = gufe.SmallMoleculeComponent(rdkit=mb)
mapper = lomap.LomapAtomMapper(time=mcs_timeout, threed=False)
mapping = next(iter(mapper.suggest_mappings(comp_a, comp_b)), None)
score = lomap.default_lomap_score(mapping)   # 0 (terrible) .. 1 (great)
```

(`hybrid_topo_rbfe.py::_lomap_like_score`, shared by both `hybrid_topo_rbfe.py`'s and `hybrid_hypersphere_rbfe.py`'s Phase 4/5 comparisons.) Requires `lomap2` + `gufe` (conda-forge).

---

## 3. What each method actually does

Before the numbers: a precise description of how each method in the comparison table generates its prediction, since the names alone ("independent", "ablation", "combined") don't fully convey the mechanism.

| Method | Input processing | Embedding / score generation | Geometry | Training objective | Uses `\|SE\|` labels? |
|---|---|---|---|---|---|
| **Tanimoto (1 − sim)** | Ligand A and B each converted to a Morgan fingerprint (radius 2, 2048 bits) via RDKit, independently. | Score = 1 − TanimotoSimilarity(fp_A, fp_B). Pure bit-vector overlap; no atom mapping, no learned parameters. | N/A (bit-vector) | None — deterministic formula. | No |
| **LOMAP (real, 1 − sim)** | Ligand A and B each embedded into a 3D conformer (RDKit `EmbedMolecule` + MMFF), wrapped as `gufe.SmallMoleculeComponent`. | `lomap.LomapAtomMapper` finds the best maximum-common-substructure atom mapping between A and B; `lomap.default_lomap_score` combines exponential MCS-coverage penalties (MCSR/MNACR) with explicit penalties for ring-breaking, ring-size changes, hybridization changes, and charge changes. Score = 1 − LOMAP score. | N/A (rule-based) | None — deterministic algorithm, no training. | No |
| **Hypersphere (independent)** | Ligand A and B each converted to an atom/bond-featurized molecular graph and encoded **completely separately** — the encoder never sees both molecules together. | Each ligand's graph → MPNN message-passing → pooled to one vector per ligand → L2-normalized onto the unit hypersphere, giving two points `h_a`, `h_b`. A regression head combines `h_a`/`h_b` (via a `hypersphere_rbfe.py`-defined feature such as concatenation/difference/cosine) to predict `\|SE\|`. | Hyperspherical (unit-norm embeddings) | Continuous contrastive + regression use labels; dispersion is a detached diagnostic with no encoder gradient. | Yes — contrastive + regression; not dispersion |
| **MCS/GNN (independent)** | Ligand A and B are **merged into one hybrid-topology graph** via MCS: atoms tagged core (shared) / A-unique / B-unique. | The single merged graph → one MPNN → one pooled embedding vector representing the *whole transformation* (not either ligand individually) → regression head predicts `\|SE\|` directly from this one vector. This is `hybrid_topo_rbfe.py`'s original, independently-trained/validated model. | Flat Euclidean (no normalization) | Regression only (MSE on `\|SE\|`). | Yes (regression only) |
| **Hybrid + L2-norm (ablation)** | Same merged hybrid-topology graph construction as `MCS/GNN (independent)` above. | Same MPNN encoder architecture (`HybridGNN.encode()`), but the resulting pair embedding is **L2-normalized onto the hypersphere** before the regression head. Isolates the effect of hyperspherical geometry alone. | Hyperspherical | Regression only (MSE on `\|SE\|`) — no contrastive or dispersion pressure. | Yes (regression only) |
| **Hybrid + Hypersphere (combined)** | Same merged hybrid-topology graph construction as above. | Same L2-normalized single pair embedding as the ablation, but now trained with the full 3-component objective. Since there is only one embedding per pair (no separate `h_a`/`h_b` to compare), the contrastive term is reformulated across the batch: transformations `i,j` with similar observed `|SE|` labels (`\|SE_i − SE_j\|` small) are pulled close on the sphere; dissimilar-label transformations are pushed apart (`batch_pairwise_contrastive_loss`). Dispersion is a detached diagnostic with no encoder gradient. | Hyperspherical | Batch-pairwise contrastive + regression use labels; dispersion is a detached diagnostic with no encoder gradient. | Yes — contrastive + regression; not dispersion |

---

## 4. Historical six-way comparison (provisional)

The following table is preserved from an earlier report of a 20,000-pair
comparison using real LOMAP rather than the proxy. The run predates graph-pair
identity checks and benchmark-bound checkpoints, so alignment has not been
verified in this repair.

| Method | Pearson r | Spearman ρ | AUROC (top-25% hard) | AUPRC | MCC |
|---|---|---|---|---|---|
| Tanimoto (1 − sim) | −0.064 | −0.057 | 0.464 | 0.228 | −0.055 |
| **LOMAP (real, 1 − sim)** | 0.012 (p=0.19, n.s.) | −0.012 (n.s.) | 0.500 | 0.250 | −0.004 |
| Hypersphere (independent) | 0.910 | 0.864 | 0.962 | 0.917 | 0.785 |
| MCS/GNN (independent) | 0.523 | 0.388 | 0.743 | 0.551 | 0.332 |
| Hybrid + L2-norm (ablation) | 0.290 | 0.211 | 0.636 | 0.397 | 0.187 |
| Hybrid + Hypersphere (combined) | 0.604 | 0.505 | 0.796 | 0.613 | 0.446 |

These numbers are retained as a historical record only. They do not support
claims that a learned model beats LOMAP or that one model/loss configuration
is superior to another.

---

## 5. Historical ligand-A split vulnerability

### 5.1 Earlier split methodology

Earlier runs using `--split_method ligand_a` grouped on ligand A only. The
current default remains the scaffold split: both endpoint scaffolds are
assigned to one split, and pairs spanning splits are dropped. The following
describes the earlier ligand-A rule, not the current default:

1. Take the **unique set of ligand-A SMILES** across the whole dataset — not ligand B, not the pair.
2. Shuffle that list of unique ligand-A values with a fixed seed (`--seed`, default 42).
3. The last `test_frac` fraction (default 0.1 → 10%) becomes the **test** ligand-A group; the fraction before that (`val_frac`, default 0.1 → 10%) becomes the **val** ligand-A group. The remaining ~80% is **train**.
4. Every row (pair) is assigned to whichever split its ligand A falls into. All pairs sharing a given ligand A are therefore guaranteed to land in the same split together.

```python
groups = df[SMILES_A_COLUMN].unique()      # unique ligand-A values only
rng = np.random.default_rng(args.seed)
rng.shuffle(groups)
n = len(groups)
n_val  = max(1, int(n * args.val_frac))    # default 10%
n_test = max(1, int(n * args.test_frac))   # default 10%
val_g  = set(groups[n - n_val - n_test : n - n_test])
test_g = set(groups[n - n_test :])

df["split"] = "train"
df.loc[df[SMILES_A_COLUMN].isin(val_g),  "split"] = "val"
df.loc[df[SMILES_A_COLUMN].isin(test_g), "split"] = "test"
```

### 5.2 Why the earlier rule was insufficient

Every pair sharing a given ligand A is kept together in one split — but **ligand B is never grouped**. The same ligand-B molecule can appear across train, val, *and* test, paired with different A partners each time.

### Why this matters

The **independent hypersphere model** encodes ligand B with no knowledge of its transformation partner. If certain ligand-B molecules have a consistently high or low `|SE|` contribution across many different A-partners (plausible — some ligands are intrinsically harder to simulate regardless of what they're perturbed from/to), the encoder can simply **memorize "this specific ligand B → this SE value"** during training on many A-partners, then trivially recognize that same B molecule at test time paired with a novel A. That is not learning transformation difficulty — it's closer to a lookup table.

This was a plausible leakage mechanism for the earlier ligand-A runs, not
evidence that it explains any particular historical score:

| Model | How exposed is ligand-B identity? | Result |
|---|---|---|
| Hypersphere (independent) | B encoded in complete isolation | Historical r = 0.91 |
| Hybrid + Hypersphere (combined) | B entangled with A in one merged graph | Historical r = 0.60 |
| MCS/GNN (independent) | Same entanglement, no hyperspherical pressure | Historical r = 0.52 |
| Hybrid + L2-norm (ablation) | Same entanglement | Historical r = 0.29 |
| Real LOMAP / Tanimoto | No access to training labels | Historical r ≈ 0 |

The table must not be read as a reliable architecture ordering: split leakage
and unverified cache alignment are both possible confounders in the old runs.

---

## 6. Alignment repair and rerun requirements

The scaffold split remains the intended split rule. The earlier shared-loader fix addressed split-generation drift but left a separate graph-indexing defect: `mcs_precompute.py` stored graphs in the cleaned, pre-split row order, while `phase1_load` dropped straddling pairs and reset row indices. `_load_graphs` then used those reset indices against the original list. When this full-cache path was used after rows were dropped, graphs could receive another pair’s label. The cluster artifacts have not been inspected in this repair, so which historical runs used the affected path remains to be verified.

The pipelines now store versioned graph bundles keyed by exact ordered SMILES
pairs, validate data-cache manifests against input hashes and request
settings, and bind checkpoints to the ordered pairs, signed labels, and split
assignments. Legacy graph lists, unmanifested data caches, and checkpoints
without a matching benchmark identity are rejected. These safeguards apply
going forward and do not retroactively validate any historical results.

Rerun all comparisons from fresh directories using the commands in
[`living_network/README.md`](../living_network/README.md), including the
`graphs_hybrid_topo_v3`, `results_hybrid_topo_v3`,
`results_hypersphere_v3`, and `results_hybrid_hypersphere_v3` directories.
No training or baseline reruns are part of the artifact-alignment repair.

---

## 7. Week 1 run log (cluster, A40) — historical scaffold-split reports

The following numerical tables are kept as a run-history snapshot. Although
the runs were reported as using `--split_method scaffold` and `--seed 42`,
their caches/checkpoints predate provenance validation; “held-out” and
“leakage-free” labels in the original report are therefore not independently
verified here.

### 7.1 `hypersphere_rbfe.py` (independent encoder), standalone

- Phase 1 was reported as passing the go/no-go gate (6/10 descriptors) and retaining 2,414,293 pairs after scaffold filtering. The reported split counts were Train 2,399,294 / Val 7,477 / Test 7,522.
- Phase 3 was reported to early-stop at epoch 16; validation Spearman ρ peaked at epoch 6 (**0.232**). The reported loss magnitudes do not establish that regression was crowded out: in this implementation, dispersion is computed from cluster centers updated using detached embeddings, so the dispersion term has no gradient into the encoder. Any correction or loss redesign is deferred.
- Phase 4/5 was reported on a test set of n=7,522:

  | Method | Pearson r | AUROC (hard top-25%) |
  |---|---|---|
  | Tanimoto (1-sim) | -0.034 (p=0.0032) | 0.479 |
  | LOMAP-like (1-sim) | -0.034 (p=0.0031) | 0.480 |
  | NTM cross-cluster OOD | 0.100 | 0.549 |
  | NTM max-individual OOD | 0.141 | 0.517 |
  | NTM regression head | 0.080 | 0.542 |

  These are historical reported values, not certified estimates of performance or evidence that leakage caused the change from the older table.

  The independent hypersphere model constructs its own molecular graphs and
  does not use the affected hybrid graph-cache loader. Its standalone results
  are therefore not invalidated by this particular indexing defect; the
  earlier cross-script comparison still requires a matching split.

### 7.2 `hybrid_topo_rbfe.py` (merged MCS graph), standalone

Required a separate parallel MCS-precompute pass (`mcs_precompute.py`, multiprocessing) before training, since the script's own inline `--precompute` path is a serial loop and infeasible at 2.4M pairs.

  | Method | Pearson r | AUROC (hard top-25%) |
  |---|---|---|
  | Tanimoto (1-sim) | -0.019 (p=0.18, **n.s.**) | 0.483 |
  | LOMAP (real, 1-sim) | 0.023 (p=0.11, **n.s.**) | 0.499 |
  | Hybrid-topo GNN | **0.064** (p<0.0001) | 0.549 |

  The old report described the test scores as statistically significant/non-significant as shown; these results are provisional and do not establish superiority over the baselines.

The earlier report compared this value with an older ligand-A run and proposed several possible explanations. Until historical graph alignment and checkpoint benchmark identity are checked, that comparison does not establish leakage magnitude, a dataset ceiling, or an architecture effect.

### 7.3 `hybrid_hypersphere_rbfe.py` six-way comparison — historical report

First six-way run produced:

  | Method | Pearson r | AUROC (hard top-25%) |
  |---|---|---|
  | Tanimoto (1-sim) | -0.087 (p<0.0001) | 0.440 |
  | LOMAP (real, 1-sim) | 0.034 (p=0.0002) | 0.504 |
  | **Hypersphere (independent)** | **0.938** | **0.973** |
  | MCS/GNN (independent) | 0.014 (p=0.12, n.s.) | 0.528 |
  | Hybrid + L2-norm (ablation) | **0.149** | 0.594 |
  | Hybrid + Hypersphere (combined) | **0.117** | 0.580 |

The differing cleaning rules are a confirmed code discrepancy; the reported score change alone does not measure train/test overlap. The cluster artifacts have not been inspected here to quantify that overlap or determine graph alignment.

The reported ablation and combined scores in the table are also provisional:
sharing a Phase 1 loader corrected one split-generation discrepancy but did
not validate graph-cache pair order or prove that loaded checkpoints match
the comparison benchmark. Fresh, provenance-checked reruns are still needed.

### 7.4 Secondary finding: Tanimoto similarity correlates the "wrong" way

Historical tables reported negative Tanimoto-dissimilarity correlations and a
small positive raw-similarity correlation in the Phase 1 gate. These
unverified results do not establish a reproducible dataset property; revisit
the observation after aligned reruns.

### 7.5 Status / next actions

1. Rerun all models and comparisons using fresh v3 data caches, graph bundles, and checkpoints as documented in `living_network/README.md`.
2. Audit the outputs via each run's `benchmark.json`; do not compare checkpoints unless benchmark identities match.
3. After this repair, the next modeling stage is a descriptor-based baseline comparison against real LOMAP. That baseline is not implemented by this artifact-alignment work.
4. Any change to the dispersion-loss/gradient behavior is deferred; the historical loss magnitudes do not justify a claim that dispersion crowded out regression.
