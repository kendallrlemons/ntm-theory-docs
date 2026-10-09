
# Hybrid + Hypersphere Model: Results, Methodology, and the Ligand-B Leakage Issue

*Status: the Section 5 leakage issue has been fixed (scaffold split) and re-run on the full dataset — see §7 for real, leakage-free numbers. A second, cross-script split-mismatch leakage bug was found and fixed during that re-run (§7.3); `hypersphere_rbfe.py` needs one more re-run under the fix before any new six-way table can be trusted.*

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

### 5.1 Current split methodology

Implemented in `hto.phase1_load` (`hybrid_topo_rbfe.py:621-633`), which `hybrid_hypersphere_rbfe.py` calls directly for Phase 1 (reusing it for exact index alignment with cached graphs). `hypersphere_rbfe.py` has its own separate implementation (`phase1_characterize`, lines ~492-509) that follows the same grouping strategy — group by unique `smi_a` only, same leakage pattern — but is not literally the same function call, so the two scripts' splits are independently generated (different random shuffles) even though the *design* is identical:

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

### 5.2 The gap this leaves

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

### 6.1 Fix the split — now implemented, splitting by Bemis-Murcko scaffold

Rather than grouping by raw ligand identity, we now group by **Bemis-Murcko scaffold**, computed across the union of ligand identities appearing in *either* column, so no scaffold — under ligand A or B — appears in more than one split:

```python
all_smiles = pd.concat([df[SMILES_A_COLUMN], df[SMILES_B_COLUMN]]).unique()
scaffold_of = {smi: MurckoScaffold.MurckoScaffoldSmiles(smi=smi) for smi in all_smiles}
df["scaffold_a"] = df[SMILES_A_COLUMN].map(scaffold_of)
df["scaffold_b"] = df[SMILES_B_COLUMN].map(scaffold_of)
# shuffle unique scaffolds, greedily assign to val/test buckets sized by
# PAIR COUNT (not scaffold count) so val_frac/test_frac reflect actual pair
# fractions, then assign a pair to a split only if BOTH endpoints' scaffolds
# fall in that split's scaffold set (pairs straddling two splits get dropped)
```

Scaffold-level grouping is stricter than raw-ligand grouping: molecules sharing a core scaffold but differing only in R-groups are treated as the same family, closing off a subtler leakage vector (near-duplicate ligands landing in different splits). Implemented in `hybrid_topo_rbfe.py::phase1_load` and `hypersphere_rbfe.py::phase1_characterize` behind a `--split_method {ligand_a,scaffold}` flag (default `scaffold`); `hybrid_hypersphere_rbfe.py` reuses `hto.phase1_load` and exposes the same flag.

This is a strictly harder generalization test (fewer eligible pairs, since both endpoints must independently land in the same split), so expect the usable dataset size to shrink somewhat — an early run on the full dataset dropped ~19-34% of pairs as straddling splits, depending on the pair-count-weighted target sizing.

### 6.2 Re-run the six-way comparison

Re-run Phase 3 (combined model retraining) and Phase 5 (comparison) under the corrected split.

**Interpretation guide for the re-run:**
- If the independent hypersphere model's advantage **collapses** toward the hybrid/MCS-GNN numbers → confirms leakage was the primary driver of the previous gap.
- If it **doesn't collapse much** → more defensible evidence of genuine generalizable signal, and a stronger result overall (though still worth double-checking other leakage vectors, e.g. near-duplicate ligands across splits that differ only trivially).

### 6.3 Other follow-ups noted during this work

- **Isolate whether the reformulated contrastive loss helps or hurts** the combined model specifically, by training with `--w_contrastive 0` (dispersion + regression only) and comparing to the current combined result (r=0.604) and the ablation (r=0.290).
- **Investigate the mechanism gap** between the independent hypersphere approach and the merged-graph hybrid approach in general — even after fixing the split, understanding *why* keeping A and B as two separate embeddings outperforms merging them into one transformation-aware graph is itself a useful modeling question.

---

## 7. Week 1 run log (cluster, A40) — actual scaffold-split results

All three runs below used `--split_method scaffold` (the now-default), `--seed 42`, and the full dataset (`compound_smiles_stderr_differences.csv`, 13.1M raw rows → 2,965,607 cleaned pairs). This section records the real numbers produced and a second leakage issue discovered (and fixed) while generating them.

### 7.1 `hypersphere_rbfe.py` (independent encoder), standalone

- Phase 1 go/no-go gate: **PASS** (6/10 descriptors matched chemical intuition, need ≥2). Scaffold split dropped 551,314 pairs (18.6%) straddling train/val/test. Train 2,399,294 / Val 7,477 / Test 7,522.
- Phase 3 training: early-stopped at epoch 16; val Spearman ρ peaked at epoch 6 (**0.232**) and drifted down afterward — loss-term imbalance (dispersion term `d≈0.27–0.28` dwarfing the regression term `r≈0.006`) is a plausible cause, flagged for the §6.3 `--w_contrastive 0` follow-up.
- Phase 4/5 (own honest held-out test, n=7,522):

  | Method | Pearson r | AUROC (hard top-25%) |
  |---|---|---|
  | Tanimoto (1-sim) | -0.034 (p=0.0032) | 0.479 |
  | LOMAP-like (1-sim) | -0.034 (p=0.0031) | 0.480 |
  | NTM cross-cluster OOD | 0.100 | 0.549 |
  | NTM max-individual OOD | 0.141 | 0.517 |
  | NTM regression head | 0.080 | 0.542 |

  **This is the real, leakage-free number for this architecture: r≈0.08–0.14, AUROC≈0.52–0.55 (barely above chance on hard-pair discrimination).** A ~90% relative collapse from the old r≈0.91 headline, confirming leakage was the dominant driver of that number.

### 7.2 `hybrid_topo_rbfe.py` (merged MCS graph), standalone

Required a separate parallel MCS-precompute pass (`mcs_precompute.py`, multiprocessing) before training, since the script's own inline `--precompute` path is a serial loop and infeasible at 2.4M pairs.

  | Method | Pearson r | AUROC (hard top-25%) |
  |---|---|---|
  | Tanimoto (1-sim) | -0.019 (p=0.18, **n.s.**) | 0.483 |
  | LOMAP (real, 1-sim) | 0.023 (p=0.11, **n.s.**) | 0.499 |
  | Hybrid-topo GNN | **0.064** (p<0.0001) | 0.549 |

  Notable: both baselines are statistically indistinguishable from noise on the honest test set, while the GNN's small effect size is nonetheless highly significant — real evidence that a learned representation captures something cheap heuristics don't, even though the absolute effect size is modest.

  **Open question for Week 4:** this architecture's relative collapse from its old leaky number (r=0.523 → 0.064, ~88% relative drop) is roughly the same magnitude as the independent hypersphere model's collapse (~85–91%), even though the hybrid design was specifically expected to be *less* exploitable by the diagnosed ligand-B-isolation leakage mechanism (§5.2–5.3). This suggests either (a) leakage in the old `ligand_a` split was broader than the specific mechanism diagnosed, inflating all architectures roughly proportionally, or (b) the real achievable ceiling on this dataset under an honest split is simply low (~r=0.1) and roughly architecture-independent — consistent with AUROC_hard landing at ~0.55 for every architecture tested so far regardless of design. Needs revisiting once the combined model's honest number (§7.3, re-run) is in.

### 7.3 `hybrid_hypersphere_rbfe.py` six-way comparison — cross-script split-mismatch leakage (found and fixed)

First six-way run produced:

  | Method | Pearson r | AUROC (hard top-25%) |
  |---|---|---|
  | Tanimoto (1-sim) | -0.087 (p<0.0001) | 0.440 |
  | LOMAP (real, 1-sim) | 0.034 (p=0.0002) | 0.504 |
  | **Hypersphere (independent)** | **0.938** | **0.973** |
  | MCS/GNN (independent) | 0.014 (p=0.12, n.s.) | 0.528 |
  | Hybrid + L2-norm (ablation) | **0.149** | 0.594 |
  | Hybrid + Hypersphere (combined) | **0.117** | 0.580 |

**The r=0.938 "Independent hypersphere" row is not real — it is a second, distinct leakage bug, not a resurrection of genuine signal.** Root cause: `hybrid_hypersphere_rbfe.py` and `hybrid_topo_rbfe.py` both split via the shared `hto.phase1_load`, so the "MCS/GNN independent" arm is evaluated on that same model's genuine held-out test set (and scores a consistent, low, honest r=0.014). But `hypersphere_rbfe.py` previously had its **own separate** `phase1_characterize` split implementation that was supposed to be algorithmically identical (same scaffold-grouping logic, same RNG) but silently omitted hto's `df[df.SE_abs > 0]` filter. That one missing filter changes the exact row set feeding the scaffold-degree RNG, so **the same `--seed` produced a different train/val/test partition in the two scripts** — meaning a `hypersphere_rbfe.py` checkpoint evaluated against `hybrid_hypersphere_rbfe.py`'s test set was partly being scored on its own training data. This is confirmed by the fact that the same checkpoint scored r≈0.08–0.14 on its own honest test set (§7.1) just hours earlier.

**Trustworthy numbers from this run** (both trained *and* evaluated under this script's own consistent `hto.phase1_load` split): **Hybrid + L2-norm ablation r=0.149**, **Hybrid + Hypersphere combined r=0.117** — both currently the best legitimate signal found on this dataset, and both clearly ahead of the baselines' near-null correlations.

**Fix applied:** `hypersphere_rbfe.py` now imports `hybrid_topo_rbfe` as `hto` and delegates its Phase 1 loading/cleaning/splitting entirely to `hto.phase1_load`, guaranteeing a byte-identical split across all three scripts for the same `--input/--seed/--split_method/--val_frac/--test_frac/--sample_size` — eliminating this class of leakage by construction rather than by keeping two hand-written implementations in sync. **`hypersphere_rbfe.py` needs to be re-run** (Phase 1 onward — the cached `phase1/cleaned_split_scaffold.pkl` is now stale and will be regenerated) before any new six-way table is produced or any checkpoint comparison is trusted.

### 7.4 Secondary finding: Tanimoto similarity correlates the "wrong" way

Across all three runs above, `Tanimoto (1-sim)` (dissimilarity) correlates **negatively** with `|SE|` (-0.034, -0.019, -0.087) — i.e., more *dissimilar* ligand pairs look slightly *easier* in this dataset, the opposite of the standard RBFE assumption. This is consistent with Phase 1's go/no-go gate, where raw `tanimoto_morgan` similarity showed r=+0.0385 with `|SE|` (also opposite the sign the gate's own documentation expects). The pattern has now replicated three times with growing significance as n increases, so it looks like a real, if counterintuitive, property of this dataset rather than noise — worth a dedicated look before Week 4's baseline writeup, since it may indicate a selection effect in how perturbation edges were originally drawn (e.g., edges are typically proposed between *similar* ligands by design, which could induce this kind of inversion).

### 7.5 Status / next actions

1. Re-run `hypersphere_rbfe.py` end-to-end under the fixed shared-split code.
2. Re-run `hybrid_hypersphere_rbfe.py`'s Phase 5 six-way comparison with the corrected checkpoint to get a trustworthy "Independent hypersphere" number.
3. Revisit the "why did every architecture collapse by roughly the same proportion" question (§7.2) once that number is in — it determines whether Week 4's ablation story is about *architecture* differences or a *shared data ceiling*.
4. Carry the `--w_contrastive 0` and loss-term-rebalancing follow-ups from §6.3 forward — current training curves (§7.1) suggest the regression signal may be getting crowded out by the dispersion term.
