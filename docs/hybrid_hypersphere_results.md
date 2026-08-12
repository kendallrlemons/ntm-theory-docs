
# Hybrid + Hypersphere Model: Results, Methodology, and the Ligand-B Leakage Issue

*Status: results below are provisional — a likely train/test leakage issue in the split design (Section 4) must be fixed before these numbers can be trusted.*

---

## 1. Background

Two encoders were developed and independently validated before this work:

| Script | Approach |
|---|---|
| `living_network/hybrid_topo_rbfe.py` | Merges ligand A + ligand B into **one** hybrid-topology graph via MCS (atoms tagged core / A-unique / B-unique), encodes with an MPNN → single embedding per pair → regression head predicts `\|SE\|`. Transformation-aware, flat Euclidean embedding, pure MSE loss. |
| `living_network/hypersphere_rbfe.py` | Encodes ligand A and ligand B **independently** (no knowledge of the transformation partner) with an MPNN, L2-normalizes each onto the unit hypersphere, trains with a 3-component loss (continuous contrastive + dispersion + regression). |

`living_network/hybrid_hypersphere_rbfe.py` combines both ideas: it reuses the hybrid-topology encoder's single transformation-aware embedding (via `HybridGNN.encode()`), L2-normalizes it onto the hypersphere, and trains two variants:

- **Phase 2 (ablation)** — hybrid encoder + L2-norm + regression-only loss. Isolates whether hyperspherical geometry *alone*, with no contrastive/dispersion pressure, helps over the flat-Euclidean baseline.
- **Phase 3 (combined)** — hybrid encoder + L2-norm + full 3-component loss.

Because `HybridGNN` produces only **one** embedding per pair (not two independent per-ligand embeddings), the original hypersphere contrastive loss — which compares `h_a` vs `h_b` *within* a pair — doesn't apply. It was reformulated as `batch_pairwise_contrastive_loss`: transformations `i, j` with similar predicted difficulty (`|SE_i - SE_j|` small) should embed close together on the sphere *across* the batch; dissimilar-difficulty transformations should be far apart. Dispersion and the regression head carry over unchanged.

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
| **Hypersphere (independent)** | Ligand A and B each converted to an atom/bond-featurized molecular graph and encoded **completely separately** — the encoder never sees both molecules together. | Each ligand's graph → MPNN message-passing → pooled to one vector per ligand → L2-normalized onto the unit hypersphere, giving two points `h_a`, `h_b`. A regression head combines `h_a`/`h_b` (via a `hypersphere_rbfe.py`-defined feature such as concatenation/difference/cosine) to predict `\|SE\|`. | Hyperspherical (unit-norm embeddings) | 3-component loss: continuous contrastive (`cos(h_a,h_b)` targets `cos(π·SE_norm)` — similar ligands pulled close, dissimilar pushed apart) + dispersion (spreads learned cluster centers across the sphere) + regression (MSE on `\|SE\|`). | Yes (all 3 loss terms) |
| **MCS/GNN (independent)** | Ligand A and B are **merged into one hybrid-topology graph** via MCS: atoms tagged core (shared) / A-unique / B-unique. | The single merged graph → one MPNN → one pooled embedding vector representing the *whole transformation* (not either ligand individually) → regression head predicts `\|SE\|` directly from this one vector. This is `hybrid_topo_rbfe.py`'s original, independently-trained/validated model. | Flat Euclidean (no normalization) | Regression only (MSE on `\|SE\|`). | Yes (regression only) |
| **Hybrid + L2-norm (ablation)** | Same merged hybrid-topology graph construction as `MCS/GNN (independent)` above. | Same MPNN encoder architecture (`HybridGNN.encode()`), but the resulting pair embedding is **L2-normalized onto the hypersphere** before the regression head. Isolates the effect of hyperspherical geometry alone. | Hyperspherical | Regression only (MSE on `\|SE\|`) — no contrastive or dispersion pressure. | Yes (regression only) |
| **Hybrid + Hypersphere (combined)** | Same merged hybrid-topology graph construction as above. | Same L2-normalized single pair embedding as the ablation, but now trained with the full 3-component objective. Since there is only one embedding per pair (no separate `h_a`/`h_b` to compare), the contrastive term is reformulated across the batch: transformations `i,j` with similar predicted difficulty (`\|SE_i − SE_j\|` small) are pulled close on the sphere; dissimilar-difficulty transformations are pushed apart (`batch_pairwise_contrastive_loss`). Dispersion and regression carry over unchanged. | Hyperspherical | 3-component loss: batch-pairwise contrastive + dispersion + regression (MSE on `\|SE\|`). | Yes (all 3 loss terms) |

---

## 4. Latest six-way comparison results

Evaluated on 20,000 shared test pairs, real LOMAP score in place of the proxy:

| Method | Pearson r | Spearman ρ | AUROC (top-25% hard) | AUPRC | MCC |
|---|---|---|---|---|---|
| Tanimoto (1 − sim) | −0.064 | −0.057 | 0.464 | 0.228 | −0.055 |
| **LOMAP (real, 1 − sim)** | 0.012 (p=0.19, n.s.) | −0.012 (n.s.) | 0.500 | 0.250 | −0.004 |
| Hypersphere (independent) | 0.910 | 0.864 | 0.962 | 0.917 | 0.785 |
| MCS/GNN (independent) | 0.523 | 0.388 | 0.743 | 0.551 | 0.332 |
| Hybrid + L2-norm (ablation) | 0.290 | 0.211 | 0.636 | 0.397 | 0.187 |
| Hybrid + Hypersphere (combined) | 0.604 | 0.505 | 0.796 | 0.613 | 0.446 |

### Key observations

1. **Both structural baselines are indistinguishable from noise.** Tanimoto and real LOMAP both sit at essentially zero correlation and AUROC ≈ 0.5. This is not an implementation artifact of the old heuristic — the *actual, validated* LOMAP algorithm performs identically to a naive fingerprint-similarity baseline on this dataset's `|SE|` target.
2. **All three learned models beat both baselines by a wide margin**, with a clear internal ordering: `Hypersphere (independent)` ≫ `Hybrid + Hypersphere (combined)` > `MCS/GNN (independent)` > `Hybrid + L2-norm (ablation)`.
3. **The 3-component loss helps within the hybrid-encoder family**: combining hyperspherical geometry with contrastive + dispersion losses (r=0.604) beats both the flat-Euclidean hybrid encoder (r=0.523, from the parent `hybrid_topo_rbfe.py`) and the L2-norm-only ablation (r=0.290) — L2-normalizing without the auxiliary losses actively hurts, but adding them back more than recovers the loss and provides a modest net gain.
4. **The independent hypersphere model's dominance (r=0.91) is the most important number to interpret carefully** — see Section 5.

---

## 5. Potential issue: train/test split does not group by ligand B

### The split code

`hto.phase1_load` (`hybrid_topo_rbfe.py:621-633`) builds the train/val/test split by grouping **only on ligand A**:

```python
groups = df[SMILES_A_COLUMN].unique()
rng.shuffle(groups)
...
df.loc[df[SMILES_A_COLUMN].isin(val_g),  "split"] = "val"
df.loc[df[SMILES_A_COLUMN].isin(test_g), "split"] = "test"
```

Every pair sharing a given ligand A is kept together in one split — but **ligand B is never grouped**. The same ligand-B molecule can appear across train, val, *and* test, paired with different A partners each time.

### Why this matters

The **independent hypersphere model** encodes ligand B with no knowledge of its transformation partner. If certain ligand-B molecules have a consistently high or low `|SE|` contribution across many different A-partners (plausible — some ligands are intrinsically harder to simulate regardless of what they're perturbed from/to), the encoder can simply **memorize "this specific ligand B → this SE value"** during training on many A-partners, then trivially recognize that same B molecule at test time paired with a novel A. That is not learning transformation difficulty — it's closer to a lookup table.

This mechanism predicts exactly the ordering observed:

| Model | How exposed is ligand-B identity? | Result |
|---|---|---|
| Hypersphere (independent) | B encoded in complete isolation — easiest to exploit | r = 0.91 (highest) |
| Hybrid + Hypersphere (combined) | B entangled with A in one merged graph — partially obscured | r = 0.60 |
| MCS/GNN (independent) | Same entanglement, no hyperspherical pressure | r = 0.52 |
| Hybrid + L2-norm (ablation) | Same entanglement | r = 0.29 |
| Real LOMAP / Tanimoto | No access to training labels at all — can't memorize | r ≈ 0 (lowest) |

The *less* transformation-aware / more B-isolated the architecture, the *better* it scores — the opposite of what should happen if the models were learning genuine structural-difficulty signal. This is a strong tell for leakage via ligand-B identity, and it means the current r=0.91 headline number should **not** be reported or relied on until this is ruled out.

---

## 6. Planned fix and path forward

### 6.1 Fix the split

Group by the union of ligand identities appearing in *either* column, so no ligand — A or B — appears in more than one split:

```python
all_ligands = pd.concat([df[SMILES_A_COLUMN], df[SMILES_B_COLUMN]]).unique()
# shuffle / partition all_ligands into train/val/test ligand sets, then
# assign a pair to a split only if BOTH endpoints fall in that split's ligand set
# (pairs straddling two splits get dropped)
```

This is a strictly harder generalization test (fewer eligible pairs, since both endpoints must independently land in the same split), so expect the usable dataset size to shrink somewhat.

### 6.2 Re-run the six-way comparison

Re-run Phase 3 (combined model retraining) and Phase 5 (comparison) under the corrected split.

**Interpretation guide for the re-run:**
- If the independent hypersphere model's advantage **collapses** toward the hybrid/MCS-GNN numbers → confirms leakage was the primary driver of the previous gap.
- If it **doesn't collapse much** → more defensible evidence of genuine generalizable signal, and a stronger result overall (though still worth double-checking other leakage vectors, e.g. near-duplicate ligands across splits that differ only trivially).

### 6.3 Other follow-ups noted during this work

- **Isolate whether the reformulated contrastive loss helps or hurts** the combined model specifically, by training with `--w_contrastive 0` (dispersion + regression only) and comparing to the current combined result (r=0.604) and the ablation (r=0.290).
- **Investigate the mechanism gap** between the independent hypersphere approach and the merged-graph hybrid approach in general — even after fixing the split, understanding *why* keeping A and B as two separate embeddings outperforms merging them into one transformation-aware graph is itself a useful modeling question.
