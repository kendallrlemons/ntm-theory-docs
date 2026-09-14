# NTM Dissertation Sprint: Plan to Defense-Ready Results

*A schematic for closing out the PhD: from a leakage-contaminated headline number to a complete, defensible story — difficulty is learnable, it generalizes, and it can be used to allocate computational budget.*

---

## Table of Contents

1. [Guiding Narrative](#1-guiding-narrative)
2. [Timeline Overview (7 Weeks)](#2-timeline-overview-7-weeks)
3. [Week-by-Week Plan](#3-week-by-week-plan)
4. [Task Details](#4-task-details)
5. [Risk Register](#5-risk-register)
6. [Definition of Done](#6-definition-of-done)

---

## 1. Guiding Narrative

Everything below serves one dissertation arc:

1. **Transformation difficulty is learnable** from structural/topological pair information.
2. **NTM learns a useful representation/geometry** of that difficulty — better than hand-crafted similarity.
3. **It generalizes beyond training chemistry** — the representation isn't just memorized ligand identity.
4. **Predicted difficulty can guide resource allocation** — $\hat{s}_e$ informs $n_e$ per the proposal's $\sigma_e^2 = s_e^2/n_e$ relationship.
5. **NTM-informed allocation reduces the computational burden** of RBFE networks under a fixed budget.

Every task in this plan maps to exactly one of these five beats. If a task doesn't serve the narrative, it's out of scope for the next two months.

---

## 2. Timeline Overview (7 Weeks)

| Weeks | Focus | Narrative Beat |
|---|---|---|
| 1–2 | Fix leakage, establish real benchmark, statistical robustness | (1) |
| 3 | Out-of-distribution / cross-chemistry generalization | (2), (3) |
| 4 | Ablations + baseline lock-down | (2) |
| 5 | Difficulty → allocation formalism ($\hat{s}_e \to n_e$) | (4) |
| 6 | Definitive resource-allocation experiment | (4), (5) |
| 7 | Adaptive/online retrospective simulation | (5) |

This assumes ~1 week of buffer is already absorbed into each week's scope (i.e., treat each week as ~4 effective working days given compute queue times, debugging, etc.). Weeks 8+ are reserved for writing but not detailed here.

---

## 3. Week-by-Week Plan

### Week 1 — Leakage-Free Benchmark
- Finish scaffold-split retraining of all three models (`hybrid_topo_rbfe.py`, `hypersphere_rbfe.py`, `hybrid_hypersphere_rbfe.py`) using `--split_method scaffold` end-to-end on the full dataset.
- Regenerate the six-way comparison table under the corrected split.
- Update `docs/hybrid_hypersphere_results.md` to replace the contaminated r=0.91 headline with the corrected number.

### Week 2 — Statistical Robustness
- Re-run the winning architecture across **5 seeds × the scaffold split** (fixed split, varied init) and **5 independent scaffold-split draws** (varied split, fixed init) — 10 runs total minimum.
- Report mean ± 95% CI (or full distribution) for Pearson r, Spearman ρ, AUROC/AUPRC/MCC on the hard-pair subset.
- Decide the final architecture to carry forward into all later weeks based on this evidence (not a single lucky run).

### Week 3 — Out-of-Distribution Generalization
- Construct a **held-out-chemistry** test set: scaffolds (or targets, if target labels are recoverable from the dataset) entirely excluded from train/val.
- If multiple protein targets exist in the underlying data, run a **cross-target** experiment: train on target(s) A, evaluate zero-shot on target B.
- Report performance degradation curve: in-distribution test → same-target held-out-scaffold → cross-target held-out-scaffold.

### Week 4 — Ablations + Baselines
- Ablate: hybrid/MCS topology (on/off), hyperspherical L2-norm (on/off), contrastive loss weight (0 vs. tuned), dispersion loss weight (0 vs. tuned). Reuse the ablation/combined split already implemented in `hybrid_hypersphere_rbfe.py` (Phase 2 vs Phase 3) and extend with a `--w_contrastive 0` run as already noted in `docs/hybrid_hypersphere_results.md` §6.3.
- Finalize baseline table: Tanimoto, real LOMAP, simplest learned model (independent MPNN/GNN regression, no hypersphere) — all evaluated on the identical scaffold-split test set from Week 1.
- Produce the "why does NTM work" answer as a short table + 2–3 sentence explanation per ablation arm.

### Week 5 — Difficulty → Allocation Formalism
- Formalize the mapping from predicted $\hat{s}_e$ to allocated effort $n_e$, grounded in $\sigma_e^2 = s_e^2/n_e$: derive a concrete allocation rule (e.g., $n_e \propto \hat{s}_e^2$ for fixed total budget $N = \sum n_e$, or an optimization that minimizes total network variance subject to $\sum n_e \le N$).
- Implement this as a standalone allocation function/script, decoupled from the RBFE simulation itself so it can be tested against synthetic or historical $\hat{s}_e$ values first.
- Validate the formalism analytically/numerically (e.g., on a small synthetic network) before touching real simulation infrastructure.

### Week 6 — Resource-Allocation Experiment
- Under one fixed total computational budget, compare:
  - **Uniform allocation** (equal $n_e$ everywhere)
  - **Conventional/NetBFE-style allocation** (existing heuristic, e.g. LOMAP-informed)
  - **NTM-informed allocation** ($n_e$ from Week 5's rule using the model from Week 2/4)
  - **Oracle allocation** ($n_e$ from *true* stderr, upper bound)
- Key output figure: **uncertainty/error vs. computational effort**, all four strategies overlaid.
- This is the proposal's central success criterion — reduced computational resources at equal/better accuracy — so this figure is the centerpiece of the entire dissertation's empirical contribution.

### Week 7 — Adaptive / Online Retrospective Simulation
- Simulate a sequential campaign against historical/held-out data: NTM gives an initial $\hat{s}_e$-based allocation → "simulate" results arriving (using existing stderr values as ground truth, revealed incrementally) → re-estimate and redistribute remaining budget.
- Compare final network-wide uncertainty of the adaptive strategy vs. the static NTM-informed allocation from Week 6.
- This directly addresses Subaim 1.2 (adaptive allocation) without requiring a live production MD system — a retrospective/simulated replay is sufficient and defensible.

---

## 4. Task Details

### 4.1 Resolve data leakage / final benchmark
Already substantially implemented: `hybrid_topo_rbfe.py::phase1_load` and `hypersphere_rbfe.py::phase1_characterize` both support `--split_method scaffold` (Bemis-Murcko scaffold grouping across both ligand columns, pair-count-weighted val/test sizing, straddling pairs dropped). Remaining work is purely re-running training end-to-end and swapping the headline numbers in `docs/hybrid_hypersphere_results.md`.

### 4.2 Statistical robustness
No infrastructure exists yet for multi-seed/multi-split aggregation. Plan: wrap the existing training entrypoints in a loop (`--seed` sweep), collect the six-way comparison CSVs across runs, aggregate with a small pandas script into a mean ± CI table and a box/violin plot.

### 4.3 True OOD generalization
Depends on whether the dataset carries a recoverable target/protein identifier. If not present in the current CSV, this needs a data audit first — check `compound_smiles_stderr_differences.csv` for any latent target-grouping signal (e.g., a filename/batch column) before assuming a cross-target split is possible. If truly unavailable, the held-out-scaffold experiment (already partially achieved by the scaffold split itself, just made more extreme — e.g. bump `--test_frac` and use a *disjoint* scaffold family entirely) is the fallback.

### 4.4 Ablations
The combined/ablation model split already exists (`hybrid_hypersphere_rbfe.py` Phase 2 = ablation, Phase 3 = combined). Additional ablation arms (contrastive-off, dispersion-off, hybrid-topology-off i.e. plain independent MPNN) require small `--w_contrastive`/`--w_dispersion` CLI sweeps plus one more comparison arm using `hypersphere_rbfe.py`'s independent encoder without hybrid graph merging (already available as the "MCS/GNN independent" and "Hypersphere independent" arms in Phase 5 of `hybrid_hypersphere_rbfe.py`).

### 4.5 Baseline lock-down
Tanimoto and real-LOMAP baselines are already computed in Phase 5 of both `hybrid_hypersphere_rbfe.py` and `hypersphere_rbfe.py`. Just needs re-running under the corrected split and consolidating into one final table.

### 4.6 Difficulty → allocation
New work. No existing script computes $n_e$ from $\hat{s}_e$. This is a pure optimization/statistics task, independent of the neural network — can be prototyped in a notebook before becoming a script.

### 4.7 Resource-allocation experiment
New work, depends on 4.6. Requires simulated or historical stderr-vs-effort data to construct the uncertainty-vs-effort curves; if real MD budget-scaling data isn't available, this may need to be simulated using a stderr model (e.g., stderr $\propto 1/\sqrt{n_e}$, calibrated against a handful of real multi-length runs if any exist in the dataset).

### 4.8 Adaptive/online retrospective
New work, depends on 4.6/4.7. Purely a simulation/replay script over already-collected stderr values — no new MD required.

---

## 5. Risk Register

| Risk | Impact | Mitigation |
|---|---|---|
| Scaffold split drops too many pairs, shrinking train set below viable size | Delays Week 1 | Already observed 19–34% drop rates are tolerable; monitor and adjust `val_frac`/`test_frac` if a rerun shows an unacceptable drop |
| No target/protein identifier exists for a true cross-target OOD test | Weakens Week 3 | Fall back to more aggressive scaffold-family holdout; document this as a limitation rather than blocking on unavailable metadata |
| No real multi-effort-level stderr data for Week 6/7 experiments | Blocks the centerpiece figure | Use a simulated stderr-vs-effort model calibrated to whatever real multi-run data exists; state this assumption explicitly in the dissertation |
| Multi-seed runs (Week 2) are compute/time expensive at 13M-row scale | Delays timeline | Use the already-subsampled `--sample_size` path for the seed/split sweep; only the single final "headline" run needs the full dataset |

---

## 6. Definition of Done

By the end of Week 7, the empirical work should be able to show, in order:

1. One clean, leakage-free headline benchmark table (Week 1) with confidence intervals (Week 2).
2. A generalization result showing performance on chemistry not seen during training (Week 3).
3. An ablation table answering "why does NTM work" (Week 4) and a baseline table answering "why not just use Tanimoto/LOMAP" (Week 4).
4. A derived, justified rule mapping $\hat{s}_e \to n_e$ (Week 5).
5. The uncertainty-vs-effort figure comparing uniform / conventional / NTM-informed / oracle allocation (Week 6).
6. A retrospective adaptive-allocation simulation (Week 7).
