# NTM Framework: Repository Summary, Theoretical Basis, and Experimental Roadmap

*A comprehensive reference document for the Neural Thermodynamic Metric (NTM) project.*

---

## Table of Contents

1. [Repository Summary](#1-repository-summary)
2. [Thought Process and Reasoning Arc](#2-thought-process-and-reasoning-arc)
3. [Theoretical Foundations](#3-theoretical-foundations)
4. [Connection to Committor Functions](#4-connection-to-committor-functions)
5. [Experimental Validation Program](#5-experimental-validation-program)
6. [Committor-Based Experiments in Detail](#6-committor-based-experiments-in-detail)
7. [Recommended Next Steps](#7-recommended-next-steps)

---

## 1. Repository Summary

### What It Is

A theoretical framework + computational pipeline for **Neural Thermodynamic Metrics (NTM)** — learned Riemannian metrics over molecular embedding spaces that predict the *difficulty* of alchemical transformations used in Free Energy Perturbation (FEP) calculations.

The repo has two layers.

### Layer 1: Theoretical Notebooks (`notebooks/`)

Documents the mathematical reasoning for colleagues and thesis defense.

| Notebook | Content |
|---|---|
| `01_theoretical_foundations.ipynb` | Information geometry, Fisher metric, Riemannian structure on molecular manifolds |
| `02_model_architecture.ipynb` | GNN encoder + learned metric tensor $\mathbf{M}$ via Cholesky factorization |
| `03_geodesics_and_paths.ipynb` | Shortest paths on the learned manifold ↔ minimum-dissipation alchemical protocols |
| `04_energy_landscapes.ipynb` | Connection to free energy surfaces and transition regions |
| `05_experimental_validation.ipynb` | Proposed validation framework |
| `06_interpretability_analysis.ipynb` | Eigendecomposition of $\mathbf{M}$ → hard/easy directions |
| `07_generative_path_optimization.ipynb` | NTM as steering signal for generative models (VAE/Diffusion/RL) to propose intermediates |

### Layer 2: Computational Pipeline (`scripts/`)

Executes models on the 13M-row SMILES-pair dataset.

```
shared_utils.py                  — featurization, dataset, batching
00_preprocess_data.py            — dedup, smart subsample, train/val/test split
01_lomap_baseline.py             — RDKit + GBM baseline
02_mpnn_model.py                 — message-passing GNN
03_gat_model.py                  — graph attention
04_ntm_model.py                  — learned metric tensor (the core model)
05_transformer_model.py          — SMILES-based attention
06_difficulty_decomposition.py   — eigenstructure analysis
07_evaluate_and_compare.py       — side-by-side benchmarking
run_all.sh                       — master pipeline
```

---

## 2. Thought Process and Reasoning Arc

### The Starting Problem

FEP calculations are expensive and often fail. Some molecule pairs are "easy" to transform, others "hard," and we can't predict this a priori. The field uses heuristics (Tanimoto similarity, LOMAP atom-mapping scores) that don't capture the underlying physics.

### The Core Idea

What if "transformation difficulty" has a **geometric structure**? Specifically:

1. Molecules live on a **manifold** (the embedding space learned by a GNN).
2. That manifold has a **Riemannian metric** $g_{ij}$ that measures how "costly" it is to move in different directions.
3. This metric should approximate the **Fisher information matrix** of the underlying Boltzmann distribution — which is mathematically equivalent to the Hessian of free energy.
4. If true, geodesic distance on this manifold = **thermodynamic length**, which by Crooks' identity is the minimum dissipation of any protocol connecting two states.

### Why This Is Non-Trivial

- It's not just "fancy similarity." The metric is learned from dissipation-like signals (stderr differences) rather than structural descriptors.
- Eigenstructure of $\mathbf{M}$ is interpretable: hard eigendirections should correspond to physically difficult transformations (ring opening, charge flipping, stereocenter inversion).
- The framework extends naturally to **generation**: if you know which directions are hard, you can design intermediates that break the path into easy pieces.

### The Gap

Until now, the framework is **predictive but not mechanistically validated**. It learns correlations with stderr, but nobody has shown that the learned $\mathbf{M}$ is actually isomorphic to the physical Fisher metric. That is what experimental validation via committor functions can fix.

---

## 3. Theoretical Foundations

NTM sits at the intersection of four established theories.

### 3.1 Information Geometry (Amari, Chentsov)

The Fisher information matrix $\mathcal{I}(\theta)$ defines a Riemannian metric on statistical manifolds. For a parameterized family $p(x|\theta)$:

$$\mathcal{I}_{ij}(\theta) = \mathbb{E}\!\left[\frac{\partial \log p}{\partial \theta_i}\frac{\partial \log p}{\partial \theta_j}\right]$$

In the alchemical setting, $\theta$ is the coupling parameter and $p$ is the equilibrium distribution $p_\theta(x) \propto e^{-\beta U_\theta(x)}$. NTM's learned $\mathbf{M}$ is, in the best case, a **learned approximation to this Fisher metric** on the molecular embedding space.

### 3.2 Thermodynamic Length (Weinhold 1975; Ruppeiner 1979; Crooks 2007)

For a quasistatic protocol $\theta(t)$ between two equilibrium states, the **thermodynamic length** is:

$$\mathcal{L} = \int_0^1 \sqrt{\dot\theta^T \mathcal{I}(\theta)\, \dot\theta}\, dt$$

Crooks (2007) proved the key identity linking this to dissipation:

$$\langle W_{\text{diss}} \rangle \geq \frac{\mathcal{L}^2}{2\tau}$$

where $\tau$ is protocol duration. **Short thermodynamic length ⇒ lower dissipation ⇒ lower variance in free-energy estimates.** This is the rigorous root of the claim that "transformation difficulty" is metric-like.

### 3.3 Optimal Transport / Wasserstein Geometry

The Wasserstein-2 metric between probability distributions has a Riemannian structure (Otto calculus). FEP convergence quality is closely tied to phase-space overlap $\sim \int \sqrt{p_A p_B}$ (Bhattacharyya), which is itself a Riemannian distance on the simplex of distributions.

### 3.4 Transition Path Theory (E and Vanden-Eijnden)

Provides the committor function, which is where NTM becomes mechanistically testable.

---

## 4. Connection to Committor Functions

### 4.1 Definition

The committor $q(x)$ is the probability that a trajectory initiated at configuration $x$ reaches product state $B$ before returning to reactant state $A$:

$$q(x) = \Pr[\tau_B < \tau_A \mid X_0 = x]$$

It satisfies the backward Kolmogorov equation $\mathcal{L} q = 0$ with boundary conditions $q|_A = 0,\ q|_B = 1$.

### 4.2 Why It Matters for NTM

| Property | Committor | NTM Analog |
|---|---|---|
| Optimal reaction coordinate | By construction | NTM geodesic direction $\mathbf{M}^{-1}(h_B - h_A)$ |
| Transition state | Isosurface $q = 0.5$ | NTM midpoint along geodesic |
| Measures "progress" | Between 0 and 1 | Normalized NTM distance |
| Defines barrier | $-\log q$ near $A$ | Integrated curvature along path |

### 4.3 The Formal Link

For overdamped Langevin dynamics with potential $U$, the committor barrier height is related to:

$$\Delta G^\ddagger \sim -\log\langle q(1-q)\rangle$$

and the **mean first-passage time** scales as:

$$\langle \tau \rangle \sim \exp(\beta \Delta G^\ddagger)$$

If the NTM metric is correctly learned, then **NTM distance² should scale with the committor barrier height**, because both are controlled by the Fisher/Hessian structure near the transition region. The hard eigendirections of $\mathbf{M}$ should align with directions along which the committor changes rapidly — i.e., the reactive modes.

### 4.4 The Variational Bridge

The connection is via the **variational principle for the committor**:

$$q = \arg\min_{\phi}\, \int \rho(x) |\nabla \phi(x)|_{D(x)}^2\, dx$$

subject to boundary conditions, where $D(x)$ is the diffusion tensor.

**Claim:** The NTM metric $\mathbf{M}$, when trained on dissipation-like signals, approximates (up to projection onto the embedding space):

$$\mathbf{M}_{ij}(h) \approx \rho(h) \cdot D^{-1}_{ij}(h)$$

If this is true:

- **NTM geodesics** = integral curves of $\nabla q$ in the embedding space
- **NTM distance** = $\int |\nabla q|\, ds$ along the geodesic
- **Top eigendirections of $\mathbf{M}$** = directions of largest committor gradient = reactive modes

This is the precise claim that makes NTM more than a similarity heuristic.

---

## 5. Experimental Validation Program

### 5.1 Infrastructure Prerequisites

1. **Reference dataset with known physics**
   - Schrödinger public FEP benchmark, Wang et al. 2015 JACS dataset, or similar
   - For each pair: known ΔG, known stderr, MBAR/BAR overlap statistics
   - ~100–200 pairs is sufficient

2. **MD simulation capability**
   - GROMACS, OpenMM, or AMBER
   - Forward and reverse alchemical pulling (for Jarzynski/Crooks statistics)
   - Typical: 50–100 ns per endpoint, ~1–10 ns per λ window

3. **Analysis stack**
   - MBAR (`pymbar`) for unbiased free energy estimates
   - Jarzynski equality + Crooks fluctuation theorem
   - Committor estimator (shooting, TIS, or ML-based)

### 5.2 Tier 1 — Statistical Consistency (Necessary)

**Goal:** Show $d_M$ correlates with physically meaningful dissipation measures.

| ID | Experiment | Measurement | Success Criterion |
|---|---|---|---|
| E1.1 | Thermodynamic length | Compute $\mathcal{L}$ from MBAR overlap | $\text{Corr}(d_M, \mathcal{L}) > 0.7$ |
| E1.2 | Work variance | Forward/reverse pulling; compute $\sigma_W^2$ | $\text{Corr}(d_M^2, \sigma_W^2) > 0.7$ (Crooks) |
| E1.3 | Bennett overlap | $\hat{\pi} = \langle\min(1, e^{-\beta\Delta U})\rangle$ | $d_M$ predicts $-\log\hat{\pi}$ |
| E1.4 | Convergence rate | Time to reach stderr < threshold | Negative correlation with $d_M$ |

Tier 1 alone would justify using NTM as a **screening tool** for alchemical network design.

### 5.3 Tier 2 — Committor Validation (Sufficient for strong claim)

See [Section 6](#6-committor-based-experiments-in-detail).

### 5.4 Tier 3 — Causal / Prospective (Gold Standard)

**Goal:** Blind test — propose intermediates for hard pairs, see if they actually help.

| ID | Experiment | Protocol | Success Criterion |
|---|---|---|---|
| E3.1 | Decomposed paths | For 50 hard pairs, generate NTM-guided intermediates; run FEP on $A{\to}I{\to}B$ vs $A{\to}B$ | ≥30% reduction in combined stderr |
| E3.2 | Ablation | Compare NTM vs LOMAP vs Tanimoto vs random intermediates | NTM beats all three significantly |
| E3.3 | Cross-force-field | Train NTM on OPLS, evaluate on AMBER | Eigenstructure >70% preserved |

### 5.5 Falsifiability Conditions

To be rigorous, the theory is falsified if:

- $d_M$ correlates with target stderr but **not with $\mathcal{L}$ computed from MBAR** → NTM is a similarity regressor with thermodynamic aesthetics, not physics.
- NTM-geodesic midpoints do **not** concentrate near $q = 0.5$ isosurfaces → the metric-as-committor-curvature claim fails.
- Hard eigendirections do **not** correspond to physically interpretable substructure changes → the eigenstructure is artifactual.

---

## 6. Committor-Based Experiments in Detail

### 6.1 Alchemical Committor Definition

In molecular transitions, the committor is defined on configuration space. In an **alchemical** context, we reinterpret:

- States $A$ and $B$ are the two endpoint Hamiltonians $H_A, H_B$
- The "reaction coordinate" is the coupling parameter $\lambda$, extended to full phase space $(x, \lambda)$
- The **alchemical committor** $q(x, \lambda)$: probability that an MD trajectory at configuration $x$ with coupling $\lambda$ equilibrates to state $B$'s distribution before state $A$'s, under the relevant dynamical rule

Formally:

$$\mathcal{L} q = 0, \quad q|_{\lambda=0,\text{eq}} = 0, \quad q|_{\lambda=1,\text{eq}} = 1$$

where $\mathcal{L}$ is the backward generator.

### 6.2 Phase A — Model System (Proof of Concept)

**Candidate systems:**

1. **Butane dihedral isomerization** (gauche ↔ trans): 1D committor, analytical reference
2. **Alanine dipeptide** φ/ψ transition: classic TPT benchmark, 2D committor
3. **Simple alchemical transformation:** methane → water (or similar small endpoint pair)

**For each system:**

| Step | What to Do | Output |
|---|---|---|
| A.1 | Run long MD to equilibrate both endpoints | Reference ensembles |
| A.2 | Compute committor via shooting method (or TIS, or ML committor network) | $q(x)$ on grid |
| A.3 | Train NTM on pairs sampled from this system | Learned $\mathbf{M}$ |
| A.4 | Project committor gradient onto NTM embedding space | $\nabla q$ in embedding coords |
| A.5 | **Compute alignment** of top NTM eigenvectors vs $\nabla q$ | Cosine similarity |
| A.6 | **Check geodesic midpoints** vs $q = 0.5$ isosurface | Distance between them |

**Success criteria:**

- Alignment cosine > 0.7 for top eigendirection
- Geodesic midpoints within $\sigma$ of the $q = 0.5$ surface

### 6.3 Phase B — Drug-Relevant Pairs (Scaling Up)

**B.1 ML-based committor estimation**

Exhaustive shooting is infeasible for drug-sized molecules. Use:

- **Committor networks** (Jung, Covino, Noé 2023): train a neural network $q_\theta$ on short trajectories
- **Van Koten–Weare schemes:** variational committor via Dirichlet energy
- **Diffusion maps:** approximate committor via eigenfunctions of the generator

**B.2 Paired FEP + committor on 20–30 pairs from Wang et al. 2015**

- Run FEP → get stderr (already present in dataset)
- Run committor network on intermediate λ states → get $q(x, \lambda)$
- Compare: does $\nabla q$ concentrate in the same embedding directions as the top NTM eigenvectors?

**B.3 The killer experiment — intermediate design**

For a hard pair $(A, B)$ with known committor structure:

1. Compute NTM geodesic midpoint $h_{\text{mid}}$
2. Decode $h_{\text{mid}}$ into an actual molecule $I$ (via the generative model)
3. Run MD on $I$ at $\lambda = 0.5$
4. Measure committor $q(I_{\text{config}}) \overset{?}{\approx} 0.5$

If $q(I) \approx 0.5$, NTM literally recovers the **transition state ensemble** — a publishable result.

### 6.4 Phase C — Practical Deployment

**C.1 Automatic alchemical network planning**

- Replace LOMAP scoring with NTM distance
- Use committor-validated intermediates to design $n$-hop paths
- Benchmark on held-out protein–ligand systems

**C.2 Adaptive FEP**

- Use NTM + committor to identify which λ windows need more sampling
- Predicted: hard regions correspond to $q \approx 0.5$ where the committor gradient is steep

### 6.5 Minimum Viable Thesis Contribution

A compact, defensible body of work:

- **Phase A fully completed** (toy systems): shows the principle works
- **Phase B.1 + B.2 on 5–10 pairs:** shows it scales to real molecules
- **Phase B.3 on 2–3 hard pairs:** shows it recovers transition states

The generative extension (Phase C) is the sequel paper.

---

## 7. Recommended Next Steps

### 7.1 New Notebook: `08_committor_connection.ipynb`

Proposed structure:

1. **Section 1: Mathematical bridge**
   - Derive the NTM ↔ committor correspondence formally
   - State approximation assumptions explicitly

2. **Section 2: Toy system worked example**
   - 2D double-well potential
   - Train an NTM on synthetic pair data
   - Compute committor analytically
   - Show the eigendirection–gradient alignment

3. **Section 3: Experimental protocol**
   - Phase A/B/C plan with specific systems, metrics, thresholds
   - Falsifiability conditions

4. **Section 4: Implementation sketch**
   - How to compute a committor network in the codebase
   - How to project onto NTM embeddings

### 7.2 Script Extensions

- `scripts/08_committor_network.py` — train an ML committor on trajectory data
- `scripts/09_ntm_committor_alignment.py` — measure cosine alignment and geodesic-vs-isosurface distance
- `scripts/10_intermediate_validation.py` — decode NTM midpoint to a molecule, run MD, compute empirical committor

### 7.3 Partnering / Resources Needed

- Access to an MD-capable cluster or partnership with a simulation group
- `pymbar`, `OpenMM`/`GROMACS`, `openmmtools`, possibly `openpathsampling` for TIS
- One benchmark protein–ligand system with published FEP data for Phase B

---

## Bottom Line

NTM is **theoretically well-founded** as a learned approximation to the Fisher / thermodynamic metric.

- **Tier 1 experiments** justify NTM as a screening tool (correlation with physics).
- **Tier 2 experiments** (committor alignment) justify NTM as a *mechanistic* model and are the sharpest test of the theory.
- **Tier 3 experiments** (prospective intermediate design) deliver practical value and close the loop to the generative framework in `07_generative_path_optimization.ipynb`.

The committor function is the linchpin: it turns NTM from a predictor with thermodynamic aesthetics into a falsifiable claim about the geometry of alchemical transitions.
