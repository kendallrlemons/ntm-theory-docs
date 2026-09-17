# NTM Project Sprint: Plan to Final Results

*A schematic for moving from a leakage-contaminated headline number to a complete, defensible story — difficulty is learnable, it generalizes, and it can be used to allocate computational budget.*

---

## Table of Contents

0. [Why This Work Matters](#0-why-this-work-matters)
1. [Guiding Narrative](#1-guiding-narrative)
2. [Timeline Overview (7 Weeks)](#2-timeline-overview-7-weeks)
3. [Week-by-Week Plan](#3-week-by-week-plan)
4. [Task Details](#4-task-details)
5. [Risk Register](#5-risk-register)
6. [Definition of Done](#6-definition-of-done)

---

## 0. Why This Work Matters

### The field-level problem

Relative Binding Free Energy (RBFE) calculations are the computational backbone of modern structure-based drug design, but they are expensive — each ligand-pair "edge" in a perturbation network can cost hours to days of GPU/CPU time to converge to a usable statistical error. In practice, most groups allocate simulation time **uniformly** across a network's edges, or rely on cheap structural heuristics (Tanimoto similarity, LOMAP atom-mapping scores) to decide which transformations are likely to be "hard." Neither approach is principled: uniform allocation wastes compute on easy edges and starves hard ones, and structural similarity is only a loose proxy for the thing that actually matters — how much a transformation perturbs the underlying free-energy landscape (i.e., how much statistical noise/dissipation it introduces).

### The proposal's central bet

The project proposal made a specific, falsifiable bet: transformation **difficulty** — operationalized as the standard error $s_e$ of an RBFE edge — has enough structure to be *predicted* from molecular representation alone, and that this predicted difficulty $\hat{s}_e$ can be plugged directly into the classical relationship

$$\sigma_e^2 = \frac{s_e^2}{n_e}$$

to *inform* how much simulation effort $n_e$ each edge receives. If true, this closes the loop from a purely predictive machine-learning result into a genuinely useful computational-chemistry tool: fewer total simulation-hours for the same (or better) network-wide precision. This is Subaim 1.1/1.2 of the proposal (predict difficulty, then adaptively allocate against it) and it is the concrete, practical payoff that justifies the more abstract Neural Thermodynamic Metric (NTM) framing described in `ntm_summary_and_experimental_plan.md`.

### Why this specific 7-week plan, and why now

All of the modeling work to date (`hybrid_topo_rbfe.py`, `hypersphere_rbfe.py`, `hybrid_hypersphere_rbfe.py`) has been aimed at beat (1)–(2) of the narrative below — showing that difficulty is learnable and that a hyperspherical/hybrid-topology representation captures it better than naive baselines. That work uncovered a **serious leakage bug** (ligand B never grouped in the original train/val/test split — see `docs/hybrid_hypersphere_results.md` §5), which means the headline result currently sitting in that document (r≈0.91) is not trustworthy and cannot be relied on. Before anything else can be built on top of it — generalization claims, ablations, and especially the resource-allocation story that is the proposal's actual novel contribution — that number has to be replaced with one computed under a split that cannot leak ligand identity across train/test.

Once that foundation is solid, the remaining weeks systematically build the rest of the argument: *is the result real and reproducible* (robustness), *does it transfer to chemistry the model has never seen* (generalization), *what part of the architecture is actually responsible for the signal* (ablations), *is a hand-crafted heuristic just as good* (baselines), and finally — the part that turns this from an ML paper into a computational-chemistry contribution — *can predicted difficulty actually save compute in a real allocation setting* (Weeks 5–7). Each of these is a standard objection that could be raised against this kind of result; this plan exists so that each objection has a pre-built, evidence-backed answer instead of a last-minute scramble.

---

## 1. Guiding Narrative

Everything below serves one overall research arc:

1. **Transformation difficulty is learnable** from structural/topological pair information.
2. **NTM learns a useful representation/geometry** of that difficulty — better than hand-crafted similarity.
3. **It generalizes beyond training chemistry** — the representation isn't just memorized ligand identity.
4. **Predicted difficulty can guide resource allocation** — $\hat{s}_e$ informs $n_e$ per the proposal's $\sigma_e^2 = s_e^2/n_e$ relationship.
5. **NTM-informed allocation reduces the computational burden** of RBFE networks under a fixed budget.

Every task in this plan maps to exactly one of these five beats. If a task doesn't serve the narrative, it's out of scope for the next two months.

**Why five beats and not just "the model works":** a single correlation number is not a complete result. Beat (1) establishes there's a signal at all; beat (2) establishes *your specific architecture* is the right way to capture it (vs. simpler alternatives — this is what the ablations in Week 4 are for); beat (3) rules out the most damaging alternative explanation for a good number, namely memorization (this is what leakage in §5 of `hybrid_hypersphere_results.md` already showed can happen, and why Week 1's fix and Week 3's OOD test both exist); beats (4)–(5) are what make this a *computational chemistry* contribution rather than just a *molecular property prediction* result — they are the direct payoff promised in the original proposal and the reason the NTM framework was proposed as more than a similarity heuristic in the first place.

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

**Why this matters:** every other week in this plan builds on top of whatever number comes out of Week 1. If the split still leaks, ablations in Week 4 will "explain" spurious signal, the OOD test in Week 3 will look like a bigger drop than it should (measuring a return to reality, not a real degradation), and the allocation experiments in Weeks 5–7 will be calibrated against a fictitious accuracy level. This week is not optional cleanup — it is the load-bearing wall for the rest of this project's empirical claims, and it is the direct fix for the specific mechanism already diagnosed in `docs/hybrid_hypersphere_results.md` §5.2–5.3 (ligand B never grouped, so the independent hypersphere encoder could memorize "this specific ligand B → this SE value" instead of learning transformation difficulty).

### Week 2 — Statistical Robustness
- Re-run the winning architecture across **5 seeds × the scaffold split** (fixed split, varied init) and **5 independent scaffold-split draws** (varied split, fixed init) — 10 runs total minimum.
- Report mean ± 95% CI (or full distribution) for Pearson r, Spearman ρ, AUROC/AUPRC/MCC on the hard-pair subset.
- Decide the final architecture to carry forward into all later weeks based on this evidence (not a single lucky run).

**Why this matters:** a single training run's correlation number is an anecdote, not a result — random seed and the specific random scaffold partition both introduce variance, and the first question after any single strong number will be "how do you know that wasn't luck?" Reporting a distribution (not a point estimate) is what converts Week 1's benchmark into a scientific claim you can stand behind, and it's also the practical gate for which architecture variant gets carried forward into the (expensive) Weeks 5–7 allocation experiments — you don't want to build the allocation story on top of an architecture choice you haven't stress-tested.

### Week 3 — Out-of-Distribution Generalization
- Construct a **held-out-chemistry** test set: scaffolds (or targets, if target labels are recoverable from the dataset) entirely excluded from train/val.
- If multiple protein targets exist in the underlying data, run a **cross-target** experiment: train on target(s) A, evaluate zero-shot on target B.
- Report performance degradation curve: in-distribution test → same-target held-out-scaffold → cross-target held-out-scaffold.

**Why this matters:** this is the direct empirical test of the proposal's original question — is transformation difficulty a *universal* property that transfers to new discovery settings, or is it something closer to a per-project lookup table that only works within the chemical series it was trained on? A model that performs well only in-distribution is scientifically interesting but practically useless for the proposal's stated goal, since in real drug-discovery campaigns you always want difficulty predictions for a *new* chemical series you haven't run FEP on yet. This week is also where the leakage story from Week 1 gets its final validation: if scaffold-holdout performance is reasonably close to in-distribution performance, that's strong independent confirmation that Week 1's fix actually worked (as opposed to just being a different flavor of leakage).

### Week 4 — Ablations + Baselines
- Ablate: hybrid/MCS topology (on/off), hyperspherical L2-norm (on/off), contrastive loss weight (0 vs. tuned), dispersion loss weight (0 vs. tuned). Reuse the ablation/combined split already implemented in `hybrid_hypersphere_rbfe.py` (Phase 2 vs Phase 3) and extend with a `--w_contrastive 0` run as already noted in `docs/hybrid_hypersphere_results.md` §6.3.
- Finalize baseline table: Tanimoto, real LOMAP, simplest learned model (independent MPNN/GNN regression, no hypersphere) — all evaluated on the identical scaffold-split test set from Week 1.
- Produce the "why does NTM work" answer as a short table + 2–3 sentence explanation per ablation arm.

**Why this matters:** "it works" is not a mechanistic claim by itself, and understanding *why* the architecture works is necessary for interpreting and trusting the result. Every architectural choice in the current pipeline — merging ligand A/B into one hybrid graph via MCS, projecting onto a hypersphere, adding a contrastive term, adding a dispersion term — was a design decision made for a reason, and each one needs its own piece of evidence that it earns its place in the final model, rather than being justified only by "it was in the design from the start." The baseline half of this week answers the complementary, equally important question the proposal is built around: is a NTM-learned geometric signal actually doing something that cheap, well-established heuristics (Tanimoto, LOMAP) cannot do? If a simple structural similarity score matched NTM's performance, the entire premise that "difficulty needs a learned representation" would collapse — so this comparison is not a formality, it is the central justification for the whole modeling approach existing at all.

### Week 5 — Difficulty → Allocation Formalism
- Formalize the mapping from predicted $\hat{s}_e$ to allocated effort $n_e$, grounded in $\sigma_e^2 = s_e^2/n_e$: derive a concrete allocation rule (e.g., $n_e \propto \hat{s}_e^2$ for fixed total budget $N = \sum n_e$, or an optimization that minimizes total network variance subject to $\sum n_e \le N$).
- Implement this as a standalone allocation function/script, decoupled from the RBFE simulation itself so it can be tested against synthetic or historical $\hat{s}_e$ values first.
- Validate the formalism analytically/numerically (e.g., on a small synthetic network) before touching real simulation infrastructure.

**Why this matters:** this is the pivot point of the entire project — the moment the story stops being "we predicted a number" and becomes "we used that number to do something useful." Everything in Weeks 1–4 exists to earn the right to trust $\hat{s}_e$; this week is where $\hat{s}_e$ is finally *used* for its stated purpose. Doing this analytically on a synthetic network first (rather than jumping straight into the full experiment in Week 6) is deliberate risk management: if the allocation rule has a bug or a degenerate edge case (e.g., what happens when $\hat{s}_e \to 0$, or when the budget is too small to give every edge a minimum viable sample size), it is far cheaper to discover that on a toy network than after burning a week of the real Week 6 experiment.

### Week 6 — Resource-Allocation Experiment
- Under one fixed total computational budget, compare:
  - **Uniform allocation** (equal $n_e$ everywhere)
  - **Conventional/NetBFE-style allocation** (existing heuristic, e.g. LOMAP-informed)
  - **NTM-informed allocation** ($n_e$ from Week 5's rule using the model from Week 2/4)
  - **Oracle allocation** ($n_e$ from *true* stderr, upper bound)
- Key output figure: **uncertainty/error vs. computational effort**, all four strategies overlaid.
- This is the proposal's central success criterion — reduced computational resources at equal/better accuracy — so this figure is the centerpiece of this project's empirical contribution.

**Why this matters:** this is the single figure the entire project is building toward, and it is the one that directly answers the proposal's stated success criterion (reduce computational resources while maintaining or improving accuracy). The four-way comparison is deliberately structured so every point on the spectrum is represented: uniform allocation is the naive floor everyone actually uses in practice today, the conventional/NetBFE heuristic is the state-of-the-art baseline it needs to be compared against, NTM-informed is the actual contribution, and the oracle is the theoretical ceiling that shows how much headroom is left even after NTM's improvement. Without the oracle curve, it's impossible to tell whether NTM closed most of the gap to "as good as it could possibly get" or only a small fraction of it — so all four arms are necessary, not just the top three.

### Week 7 — Adaptive / Online Retrospective Simulation
- Simulate a sequential campaign against historical/held-out data: NTM gives an initial $\hat{s}_e$-based allocation → "simulate" results arriving (using existing stderr values as ground truth, revealed incrementally) → re-estimate and redistribute remaining budget.
- Compare final network-wide uncertainty of the adaptive strategy vs. the static NTM-informed allocation from Week 6.
- This directly addresses Subaim 1.2 (adaptive allocation) without requiring a live production MD system — a retrospective/simulated replay is sufficient and defensible.

**Why this matters:** Week 6 shows NTM can inform a *static, one-shot* allocation decision made before any simulation starts. But the proposal's adaptive-allocation subaim (1.2) is a stronger and more practically valuable claim: as real simulation results start coming in, can the system *update* its difficulty estimates and *redistribute* remaining budget on the fly, the way an actual FEP campaign would want to operate in production? This is what separates a static screening tool from something resembling a real-time experimental-design system, and it's the natural bridge to future work / a follow-on paper. A retrospective replay over already-collected data is the pragmatic way to demonstrate this within a 1-week budget — building a live online MD-coupled system would be its own multi-month project and is explicitly out of scope here.

---

## 4. Task Details

### 4.1 Resolve data leakage / final benchmark
Already substantially implemented: `hybrid_topo_rbfe.py::phase1_load` and `hypersphere_rbfe.py::phase1_characterize` both support `--split_method scaffold` (Bemis-Murcko scaffold grouping across both ligand columns, pair-count-weighted val/test sizing, straddling pairs dropped). Remaining work is purely re-running training end-to-end and swapping the headline numbers in `docs/hybrid_hypersphere_results.md`.

*Context:* the original split only grouped by ligand A (`hypersphere_rbfe.py`'s legacy `phase1_characterize`), so the same ligand-B molecule could appear across train, val, and test paired with different A partners. Because the independent hypersphere encoder sees ligand B in complete isolation from its transformation partner, it could trivially memorize a per-molecule difficulty value rather than learning anything about the transformation itself — this is precisely why the "independent" hypersphere arm scored implausibly higher than every more transformation-aware architecture in the original six-way table. The scaffold-based fix closes this by ensuring no scaffold (not just no exact ligand) appears on both sides of any split.

### 4.2 Statistical robustness
No infrastructure exists yet for multi-seed/multi-split aggregation. Plan: wrap the existing training entrypoints in a loop (`--seed` sweep), collect the six-way comparison CSVs across runs, aggregate with a small pandas script into a mean ± CI table and a box/violin plot.

*Context:* varying only the random seed measures *optimization* variance (did training converge to a good local minimum); varying the scaffold-split draw measures *sampling* variance (was this particular train/test partition unusually favorable or unfavorable). Both need to be reported separately, because either source could otherwise be individually responsible for inflating the headline number.

### 4.3 True OOD generalization
Depends on whether the dataset carries a recoverable target/protein identifier. If not present in the current CSV, this needs a data audit first — check `compound_smiles_stderr_differences.csv` for any latent target-grouping signal (e.g., a filename/batch column) before assuming a cross-target split is possible. If truly unavailable, the held-out-scaffold experiment (already partially achieved by the scaffold split itself, just made more extreme — e.g. bump `--test_frac` and use a *disjoint* scaffold family entirely) is the fallback.

*Context:* a cross-target experiment (train on one protein's ligand series, evaluate zero-shot on a completely different protein's series) is a strictly stronger generalization claim than a scaffold holdout within the same dataset, because it also tests robustness to shifts in binding-pocket-driven chemistry, not just shifts in scaffold chemistry. It's worth the data audit up front rather than assuming it's infeasible, since this is the single strongest piece of evidence available for the proposal's "transfers to new discovery settings" claim.

### 4.4 Ablations
The combined/ablation model split already exists (`hybrid_hypersphere_rbfe.py` Phase 2 = ablation, Phase 3 = combined). Additional ablation arms (contrastive-off, dispersion-off, hybrid-topology-off i.e. plain independent MPNN) require small `--w_contrastive`/`--w_dispersion` CLI sweeps plus one more comparison arm using `hypersphere_rbfe.py`'s independent encoder without hybrid graph merging (already available as the "MCS/GNN independent" and "Hypersphere independent" arms in Phase 5 of `hybrid_hypersphere_rbfe.py`).

*Context:* the goal is not exhaustive ablation coverage but the minimum set needed to answer, component by component: does merging A/B into one hybrid graph help (vs. encoding them independently)? does the hyperspherical L2-norm geometry help (vs. flat Euclidean)? does the contrastive term help (vs. dispersion + regression alone)? does the dispersion term help (vs. contrastive + regression alone)? Four targeted yes/no comparisons, each isolating one design decision, is enough to construct a defensible mechanistic story without turning this into an open-ended architecture search.

### 4.5 Baseline lock-down
Tanimoto and real-LOMAP baselines are already computed in Phase 5 of both `hybrid_hypersphere_rbfe.py` and `hypersphere_rbfe.py`. Just needs re-running under the corrected split and consolidating into one final table.

*Context:* Tanimoto and LOMAP are the two heuristics actually used in practice today for network planning, so they are the necessary "why not just use the existing tool" comparison — not a formality, but the direct justification for why a learned model is worth the added complexity at all.

### 4.6 Difficulty → allocation
New work. No existing script computes $n_e$ from $\hat{s}_e$. This is a pure optimization/statistics task, independent of the neural network — can be prototyped in a notebook before becoming a script.

*Context:* this is intentionally decoupled from the GNN/hypersphere machinery so it can be validated on its own terms (does the allocation rule behave sensibly given *any* set of difficulty estimates) before being coupled to a specific model's predictions, which keeps debugging tractable — an allocation-rule bug and a model-prediction problem are much harder to disentangle if tested together for the first time.

### 4.7 Resource-allocation experiment
New work, depends on 4.6. Requires simulated or historical stderr-vs-effort data to construct the uncertainty-vs-effort curves; if real MD budget-scaling data isn't available, this may need to be simulated using a stderr model (e.g., stderr $\propto 1/\sqrt{n_e}$, calibrated against a handful of real multi-length runs if any exist in the dataset).

*Context:* this experiment is the empirical proof of the proposal's central quantitative claim ($\sigma_e^2 = s_e^2/n_e$-motivated allocation reduces compute at fixed accuracy). If real multi-length-run data doesn't exist for a meaningful subset of pairs, using a calibrated $1/\sqrt{n_e}$ scaling law is a standard, well-justified statistical assumption (it's the textbook scaling for standard error of a mean under i.i.d. sampling) — the key requirement is stating this assumption explicitly rather than presenting the resulting curve as if it were purely empirical.

### 4.8 Adaptive/online retrospective
New work, depends on 4.6/4.7. Purely a simulation/replay script over already-collected stderr values — no new MD required.

*Context:* "retrospective" here specifically means no new simulations are run — the script replays already-known stderr values in a simulated time order to demonstrate the update-and-redistribute mechanism. This is a standard, accepted way to demonstrate an online/adaptive algorithm's behavior without the cost of an actual live deployment, and is sufficient to support the claim as long as it's framed as a retrospective simulation rather than a live-system result.

---

## 5. Risk Register

| Risk | Impact | Mitigation |
|---|---|---|
| Scaffold split drops too many pairs, shrinking train set below viable size | Delays Week 1 | Already observed 19–34% drop rates are tolerable; monitor and adjust `val_frac`/`test_frac` if a rerun shows an unacceptable drop |
| No target/protein identifier exists for a true cross-target OOD test | Weakens Week 3 | Fall back to more aggressive scaffold-family holdout; document this as a limitation rather than blocking on unavailable metadata |
| No real multi-effort-level stderr data for Week 6/7 experiments | Blocks the centerpiece figure | Use a simulated stderr-vs-effort model calibrated to whatever real multi-run data exists; state this assumption explicitly when reporting results |
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
