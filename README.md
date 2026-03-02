# Neural Thermodynamic Metric (NTM) for RBFE Uncertainty Prediction

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/kendallrlemons/ntm-theory-docs/main?filepath=notebooks%2F01_theoretical_foundations.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

A comprehensive theoretical and practical guide to understanding and implementing the Neural Thermodynamic Metric approach for predicting Relative Binding Free Energy (RBFE) calculation difficulty.

## Important: This Is a Learned Surrogate

**NTM is NOT a thermodynamic calculation.** It is a machine learning model that *predicts* transformation difficulty from molecular structure alone, without running MD simulations.

| Aspect | True Thermodynamic Length | NTM (This Model) |
|--------|--------------------------|------------------|
| **Input** | MD trajectories at each λ | Molecular graphs (SMILES) |
| **Computes** | Phase space overlap, ⟨∂H/∂λ⟩ variance | Learned embedding distance |
| **Requires** | Expensive simulations | Only inference (fast) |
| **Nature** | Ground truth measurement | Surrogate prediction |

We use "thermodynamic" in the name because:
1. Training signal comes from FEP calculation outcomes
2. The Riemannian framework mirrors thermodynamic length theory
3. The quantity predicted (difficulty) is thermodynamically meaningful

**But the model learns correlations from data—it does not compute physics.**

## Overview

This repository provides educational materials explaining how machine learning can predict which molecular transformations will be computationally challenging in free energy perturbation (FEP) calculations. The core innovation is learning a **Riemannian metric** on molecular embedding space where **geodesic distance** correlates with transformation difficulty.

## Repository Structure

```
ntm-theory-docs/
├── notebooks/
│   ├── 01_theoretical_foundations.ipynb    # Mathematical background + surrogate disclaimer
│   ├── 02_model_architecture.ipynb         # GNN and metric tensor design
│   ├── 03_geodesics_and_paths.ipynb        # Computing optimal paths
│   ├── 04_energy_landscapes.ipynb          # Visualizing the manifold
│   ├── 05_practical_workflow.ipynb         # End-to-end tutorial
│   ├── 06_experimental_validation.ipynb    # PhD thesis experiments
│   ├── 07_interpretability_analysis.ipynb  # Feature importance & explanations
│   └── 08_generative_path_optimization.ipynb # Generative models for intermediates
├── src/
│   └── ntm_core.py                         # Example of What Core implementation Looks Like
├── figures/
│   └── (generated figures)
└── README.md
```

## Key Concepts

### 1. The RBFE Problem
Relative Binding Free Energy calculations predict how changing a ligand affects its binding affinity to a target protein. Some transformations converge quickly; others require extensive sampling—but predicting which is difficult *a priori*.

### 2. Neural Thermodynamic Metric
Instead of using Euclidean distance in molecular descriptor space, we learn a **Riemannian metric tensor** that warps the space according to transformation difficulty:

$$d_M(A, B) = \sqrt{(h_B - h_A)^T \mathbf{M} (h_B - h_A)}$$

where $\mathbf{M}$ is a learned positive-definite matrix that encodes which molecular features matter most for predicting difficulty.

### 3. Geodesics as Optimal Paths
The **geodesic** (shortest path under the learned metric) represents the minimum-difficulty transformation pathway. Regions of high metric curvature indicate transformation barriers.

### 4. Energy Landscape Visualization
The metric induces an "energy landscape" where:
- **Valleys** = stable molecular states (easy to reach)
- **Peaks** = transition barriers (hard to cross)
- **Saddle points** = optimal transition pathways

## Getting Started

### Option 1: Run Online (No Installation Required)
Click the **Binder badge** above to launch interactive notebooks in your browser. RDKit and all dependencies are pre-installed.

### Option 2: Local Installation

#### Prerequisites
```bash
# Clone the repository
git clone https://github.com/kendallrlemons/ntm-theory-docs.git
cd ntm-theory-docs

# Using conda (recommended for RDKit)
conda env create -f environment.yml
conda activate ntm-theory

# Or using pip (RDKit must be installed separately)
pip install -r requirements.txt
```

#### Running the Notebooks
```bash
jupyter notebook notebooks/
```

#### Running the Streamlit Visualizer
```bash
streamlit run ntm_visualizer_app.py
```

## Theoretical Background

The notebooks cover:

1. **Riemannian Geometry Basics** - Metric tensors, geodesics, curvature
2. **Graph Neural Networks** - Message passing, molecular embeddings
3. **Thermodynamic Interpretation** - Connection to free energy surfaces
4. **Optimization** - Computing geodesics via path optimization

## Experimental Validation (Notebooks 06-07)

For PhD thesis rigor, we provide comprehensive experimental validation:

| Experiment | Question Addressed |
|------------|-------------------|
| **Surrogate vs True Thermodynamic Length** | Does NTM correlate with actual dU/dλ variance? |
| **Comparison to LOMAP, FEP+, etc.** | Does NTM outperform existing tools? |
| **Ablation Studies** | Which components (GNN, metric, residual) matter? |
| **Generalization Across Targets** | Does it work on unseen proteins? |
| **Uncertainty Calibration** | Are predictions well-calibrated? |
| **Computational Efficiency** | How much compute does NTM-guided selection save? |
| **Interpretability** | What molecular features drive predictions? |



### What We Are Actually Approximating

True thermodynamic length requires computing:

$$\mathcal{L}_{thermo} = \int_0^1 \sqrt{\text{Var}_\lambda\left(\frac{\partial H}{\partial \lambda}\right)} \, d\lambda$$

This measures how much the potential energy derivative fluctuates along the alchemical path—computable only *after* running expensive MD.

Our surrogate hypothesis: **Structural dissimilarity in a learned embedding space correlates with this thermodynamic quantity.** We train on historical FEP outcomes to learn this correlation.
