"""
hypersphere_rbfe.py
===================
All-in-one pipeline for the Living Network hyperspherical RBFE difficulty model.

Predicts the 'Difference in first_pass_free_energy_stderr' (SE) column of the
compound-pair CSV using a hyperspherical GNN encoder trained with a hybrid loss
(continuous contrastive + dispersion + regression), then validates the learned
geometry and compares against LOMAP-style and Tanimoto baselines.

Phases
------
1. Data characterization: confirm SE correlates with chemical descriptors
   (heavy atoms, rings, charges, etc.) before using it as a training signal.
2. Molecular graph construction: atom/bond featurization for GNN input.
3. Hyperspherical encoder training: GNN + L2 norm + hybrid loss (Option C).
4. Geometric validation: angular distance ~ SE, OOD score ~ SE, visualization,
   flat-Euclidean ablation.
5. Baseline comparison: LOMAP-style, Tanimoto, vs learned OOD + SE prediction.

Usage
-----
    python hypersphere_rbfe.py

Defaults are hardcoded in the CONFIG block below. Edit those to point at the
CSV, set output directory, sample size, seed, etc. All CLI flags still work
and override the defaults (e.g. --phases 1,3 to run a subset of phases).
Artifacts are saved under --output_dir/phase{N}/ so later phases can resume.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import sys
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, rdFMCS, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger

from scipy.stats import pearsonr, spearmanr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    average_precision_score,
    matthews_corrcoef,
    mean_absolute_error,
    roc_auc_score,
)

RDLogger.DisableLog("rdApp.*")

# =============================================================================
# Constants & config
# =============================================================================

SE_COLUMN = "Difference in first_pass_free_energy_stderr"
SMILES_A_COLUMN = "Compound Smiles 1"
SMILES_B_COLUMN = "Compound Smiles 2"

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print(f"[device] CUDA available — using {torch.cuda.get_device_name(0)} "
          f"(device 0 of {torch.cuda.device_count()})")
else:
    DEVICE = torch.device("cpu")
    _reasons = []
    if not torch.backends.cuda.is_built():
        _reasons.append("PyTorch was not built with CUDA support")
    else:
        _reasons.append("PyTorch has CUDA support but no GPU was found")
        try:
            import subprocess, shutil
            if shutil.which("nvidia-smi"):
                _smi = subprocess.run(
                    ["nvidia-smi", "--query-gpu=name,driver_version",
                     "--format=csv,noheader"],
                    capture_output=True, text=True, timeout=5,
                )
                if _smi.returncode == 0 and _smi.stdout.strip():
                    _reasons.append(
                        f"nvidia-smi sees: {_smi.stdout.strip()!r} "
                        "(driver/CUDA version mismatch?)"
                    )
                else:
                    _reasons.append("nvidia-smi found but reported no GPUs")
            else:
                _reasons.append("nvidia-smi not found on PATH (no NVIDIA driver installed?)")
        except Exception as _e:
            _reasons.append(f"could not run nvidia-smi ({_e})")
    print(f"[device] CUDA unavailable — falling back to CPU. Reason(s): "
          + "; ".join(_reasons))


# =============================================================================
# CONFIG — hardcoded defaults (override via CLI if desired)
# =============================================================================

DEFAULT_INPUT = "/Users/lemonsk/Downloads/compound_smiles_stderr_differences.csv"
DEFAULT_OUTPUT_DIR = "./results"
DEFAULT_SAMPLE_SIZE = 0             # rows to use after cleaning (0 = keep all)
DEFAULT_SEED = 42
DEFAULT_PHASES = "1,2,3,4,5"
DEFAULT_NUM_CLUSTERS = 32          # K for dispersion loss / cross-cluster OOD


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser()
    # I/O
    p.add_argument("--input", default=DEFAULT_INPUT, help="Raw CSV path")
    p.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR,
                   help="Output directory")
    p.add_argument("--phases", default=DEFAULT_PHASES,
                   help="Comma-separated list of phases to run (1-5)")
    # Data
    p.add_argument("--sample_size", type=int, default=DEFAULT_SAMPLE_SIZE,
                   help="Rows to use after cleaning (0 = keep all)")
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--test_frac", type=float, default=0.1)
    p.add_argument("--split_method", choices=["ligand_a", "scaffold"], default="scaffold",
                   help="'ligand_a' (legacy): group only by ligand-A identity — "
                        "ligand B can leak across splits. "
                        "'scaffold' (default): group by Bemis-Murcko scaffold "
                        "across both ligands, sized by pair count, dropping "
                        "any pair whose two ligands' scaffolds land in "
                        "different splits.")
    # Model
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--embed_dim", type=int, default=128)
    p.add_argument("--num_clusters", type=int, default=DEFAULT_NUM_CLUSTERS,
                   help="K for dispersion loss")
    p.add_argument("--dropout", type=float, default=0.1)
    # Training
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--num_workers", type=int, default=4)
    # Loss weights
    p.add_argument("--w_contrastive", type=float, default=1.0)
    p.add_argument("--w_dispersion", type=float, default=0.5)
    p.add_argument("--w_regression", type=float, default=1.0)
    p.add_argument("--cluster_ema", type=float, default=0.9,
                   help="EMA decay for cluster centers")
    # Ablation
    p.add_argument("--flat_ablation", action="store_true",
                   help="Run flat Euclidean ablation in Phase 4")
    # Misc
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# =============================================================================
# PHASE 1 — Data Characterization
# =============================================================================
# =============================================================================

@dataclass
class PairDescriptors:
    """Structural deltas between a pair of molecules."""
    delta_heavy_atoms: int
    delta_rings: int
    delta_aromatic_rings: int
    delta_rotatable_bonds: int
    delta_formal_charge: int
    delta_hbd: int
    delta_hba: int
    delta_molwt: float
    tanimoto_morgan: float
    mcs_coverage: float  # fraction of atoms shared via MCS


def compute_pair_descriptors(smi_a: str, smi_b: str) -> Optional[PairDescriptors]:
    """Compute structural difference descriptors for a pair."""
    mol_a = Chem.MolFromSmiles(smi_a)
    mol_b = Chem.MolFromSmiles(smi_b)
    if mol_a is None or mol_b is None:
        return None

    # Morgan fingerprint Tanimoto
    fp_a = AllChem.GetMorganFingerprintAsBitVect(mol_a, 2, nBits=2048)
    fp_b = AllChem.GetMorganFingerprintAsBitVect(mol_b, 2, nBits=2048)
    tanimoto = DataStructs.TanimotoSimilarity(fp_a, fp_b)

    # MCS coverage (fraction of min atoms)
    try:
        mcs = rdFMCS.FindMCS(
            [mol_a, mol_b],
            timeout=2,
            atomCompare=rdFMCS.AtomCompare.CompareElements,
            bondCompare=rdFMCS.BondCompare.CompareOrder,
        )
        mcs_atoms = mcs.numAtoms
        min_atoms = min(mol_a.GetNumHeavyAtoms(), mol_b.GetNumHeavyAtoms())
        mcs_coverage = mcs_atoms / max(min_atoms, 1)
    except Exception:
        mcs_coverage = 0.0

    def _formal_charge(m):
        return sum(a.GetFormalCharge() for a in m.GetAtoms())

    return PairDescriptors(
        delta_heavy_atoms=abs(mol_a.GetNumHeavyAtoms() - mol_b.GetNumHeavyAtoms()),
        delta_rings=abs(rdMolDescriptors.CalcNumRings(mol_a)
                        - rdMolDescriptors.CalcNumRings(mol_b)),
        delta_aromatic_rings=abs(rdMolDescriptors.CalcNumAromaticRings(mol_a)
                                 - rdMolDescriptors.CalcNumAromaticRings(mol_b)),
        delta_rotatable_bonds=abs(rdMolDescriptors.CalcNumRotatableBonds(mol_a)
                                  - rdMolDescriptors.CalcNumRotatableBonds(mol_b)),
        delta_formal_charge=abs(_formal_charge(mol_a) - _formal_charge(mol_b)),
        delta_hbd=abs(rdMolDescriptors.CalcNumHBD(mol_a)
                      - rdMolDescriptors.CalcNumHBD(mol_b)),
        delta_hba=abs(rdMolDescriptors.CalcNumHBA(mol_a)
                      - rdMolDescriptors.CalcNumHBA(mol_b)),
        delta_molwt=abs(Descriptors.MolWt(mol_a) - Descriptors.MolWt(mol_b)),
        tanimoto_morgan=tanimoto,
        mcs_coverage=mcs_coverage,
    )


def phase1_characterize(input_csv: str, output_dir: str,
                        sample_size: int, seed: int,
                        val_frac: float = 0.1, test_frac: float = 0.1,
                        split_method: str = "scaffold") -> pd.DataFrame:
    """
    Phase 1: Load, clean, compute descriptors, validate SE as a difficulty signal.

    Returns a cleaned DataFrame with columns:
        smi_a, smi_b, SE, SE_abs, [descriptor columns], split
    """
    print("\n" + "=" * 70)
    print("PHASE 1 — Data Characterization")
    print("=" * 70)

    phase_dir = os.path.join(output_dir, "phase1")
    os.makedirs(phase_dir, exist_ok=True)

    # ---- Load & clean ----
    print(f"\n[1.1] Loading {input_csv} ...")
    df = pd.read_csv(input_csv, usecols=[SMILES_A_COLUMN, SMILES_B_COLUMN, SE_COLUMN])
    df = df.rename(columns={
        SMILES_A_COLUMN: "smi_a",
        SMILES_B_COLUMN: "smi_b",
        SE_COLUMN: "SE",
    })
    n_raw = len(df)
    df = df.dropna().drop_duplicates(subset=["smi_a", "smi_b"])
    df["SE_abs"] = df["SE"].abs()
    n_clean = len(df)
    print(f"    Raw rows: {n_raw:,}  |  After dedup + NaN drop: {n_clean:,}")

    # ---- Subsample ----
    try:
        sample_size = int(sample_size)
    except (TypeError, ValueError) as e:
        raise TypeError(
            f"sample_size must be an integer, got {type(sample_size).__name__!r}: "
            f"{sample_size!r}. "
            "If you set DEFAULT_SAMPLE_SIZE, use underscores (500_000), not commas (500,000)."
        ) from e
    if sample_size > 0 and sample_size < n_clean:
        df = df.sample(n=sample_size, random_state=seed).reset_index(drop=True)
        print(f"    Subsampled to {len(df):,} rows (seed={seed})")

    # ---- SE distribution ----
    print("\n[1.2] SE distribution:")
    se = df["SE_abs"].values
    for q in [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]:
        print(f"      q{int(q*100):>2}: {np.quantile(se, q):.6f}")
    print(f"      mean: {se.mean():.6f}  std: {se.std():.6f}")
    print(f"      skew: {pd.Series(se).skew():.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(se, bins=80, color="steelblue", alpha=0.85)
    axes[0].set_xlabel("|SE|")
    axes[0].set_ylabel("Count")
    axes[0].set_title("SE magnitude distribution")
    axes[1].hist(np.log10(se + 1e-8), bins=80, color="steelblue", alpha=0.85)
    axes[1].set_xlabel("log10(|SE|)")
    axes[1].set_title("Log-scale SE distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(phase_dir, "se_distribution.png"), dpi=140)
    plt.close()

    # ---- Structural descriptors (subset for speed; MCS is slow) ----
    print("\n[1.3] Computing structural descriptors on sub-sample (for validation)")
    n_desc = min(20_000, len(df))
    desc_idx = np.random.RandomState(seed).choice(len(df), n_desc, replace=False)
    desc_rows = []
    t0 = time.time()
    for i, idx in enumerate(desc_idx):
        row = df.iloc[idx]
        d = compute_pair_descriptors(row.smi_a, row.smi_b)
        if d is not None:
            rec = asdict(d)
            rec["SE_abs"] = row["SE_abs"]
            rec["idx"] = int(idx)
            desc_rows.append(rec)
        if (i + 1) % 2000 == 0:
            eta = (time.time() - t0) / (i + 1) * (n_desc - i - 1)
            print(f"      [{i+1:>6}/{n_desc}]  ETA {eta/60:.1f} min")
    desc_df = pd.DataFrame(desc_rows)
    desc_df.to_csv(os.path.join(phase_dir, "pair_descriptors_sample.csv"), index=False)
    print(f"    Computed descriptors for {len(desc_df):,} pairs")

    # ---- Correlation with SE ----
    print("\n[1.4] Correlation of descriptors with |SE| (go/no-go gate):")
    descriptor_cols = [c for c in desc_df.columns if c not in ("SE_abs", "idx")]
    corr_rows = []
    for col in descriptor_cols:
        x = desc_df[col].values
        y = desc_df["SE_abs"].values
        # Guard: pearsonr/spearmanr are undefined when a variable has zero
        # variance (e.g. all pairs have the same formal-charge delta).
        if np.nanstd(x) < 1e-12 or np.nanstd(y) < 1e-12:
            pr, pp, sr, sp = np.nan, np.nan, np.nan, np.nan
            constant_note = "  [constant input; correlation undefined]"
        else:
            with np.errstate(invalid="ignore"):
                pr, pp = pearsonr(x, y)
                sr, sp = spearmanr(x, y)
            constant_note = ""
        corr_rows.append({
            "descriptor": col,
            "pearson_r": pr, "pearson_p": pp,
            "spearman_r": sr, "spearman_p": sp,
            "note": constant_note.strip(),
        })
    corr_df = pd.DataFrame(corr_rows).set_index("descriptor")
    print(corr_df.round(4).to_string())
    corr_df.to_csv(os.path.join(phase_dir, "descriptor_se_correlations.csv"))

    # Expected signs: more structural change → higher SE
    #   delta_* and SE_abs: positive correlation expected
    #   tanimoto and mcs_coverage: negative correlation expected (more similar → lower SE)
    expected_positive = [c for c in descriptor_cols if c.startswith("delta_")]
    expected_negative = ["tanimoto_morgan", "mcs_coverage"]

    def _sr(col):
        v = corr_df.loc[col, "spearman_r"]
        return v if not pd.isna(v) else 0.0

    def _sp(col):
        v = corr_df.loc[col, "spearman_p"]
        return float(v) if not pd.isna(v) else 1.0

    # Gate criterion: correct sign AND statistically significant (p < 0.05).
    # We use significance rather than effect size because RBFE congeneric series
    # datasets have inherently compressed structural variance — correlations are
    # small by construction, not because SE is uninformative.
    signs_ok = 0
    for col in expected_positive:
        if _sr(col) > 0 and _sp(col) < 0.05:
            signs_ok += 1
    for col in expected_negative:
        if _sr(col) < 0 and _sp(col) < 0.05:
            signs_ok += 1

    gate = {
        "n_descriptors_checked": len(descriptor_cols),
        "n_consistent_with_intuition": int(signs_ok),
        "gate_threshold": 2,
        "gate_criterion": "correct sign AND p < 0.05 (significance, not effect size)",
        "passed": signs_ok >= 2,
    }
    with open(os.path.join(phase_dir, "go_no_go_gate.json"), "w") as f:
        json.dump(gate, f, indent=2)

    print(f"\n    Go/no-go: {signs_ok}/{len(descriptor_cols)} descriptors "
          f"behave as chemically expected (need ≥ 2)")
    if not gate["passed"]:
        print("    WARNING: SE does not behave like a difficulty signal. "
              "Proceeding, but interpret results with caution.")
    else:
        print("    PASS: SE correlates with chemical intuition. "
              "Continuing to Phase 2.")

    # ---- Visualize key descriptor vs SE ----
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    key_descs = ["tanimoto_morgan", "mcs_coverage", "delta_heavy_atoms",
                 "delta_rings", "delta_formal_charge", "delta_rotatable_bonds"]
    for ax, col in zip(axes.flat, key_descs):
        ax.scatter(desc_df[col], desc_df["SE_abs"], s=2, alpha=0.25)
        sr = corr_df.loc[col, "spearman_r"]
        sr_str = f"{sr:.3f}" if not pd.isna(sr) else "n/a"
        ax.set_xlabel(col)
        ax.set_ylabel("|SE|")
        ax.set_title(f"{col}  (Spearman ρ = {sr_str})")
    plt.tight_layout()
    plt.savefig(os.path.join(phase_dir, "descriptors_vs_se.png"), dpi=140)
    plt.close()

    # ---- Plotly: interactive Spearman correlation bar chart ----
    print("\n[1.6] Writing interactive Plotly visualizations")
    corr_sorted = corr_df.sort_values("spearman_r")
    colors = ["#d62728" if r < 0 else "#1f77b4" for r in corr_sorted["spearman_r"]]
    fig_bar = go.Figure(go.Bar(
        x=corr_sorted["spearman_r"],
        y=corr_sorted.index,
        orientation="h",
        marker_color=colors,
        text=[f"ρ={r:+.3f}  r={pr:+.3f}  p={pp:.1e}"
              for r, pr, pp in zip(corr_sorted["spearman_r"],
                                   corr_sorted["pearson_r"],
                                   corr_sorted["pearson_p"])],
        textposition="auto",
        hovertemplate=("<b>%{y}</b><br>"
                       "Spearman ρ = %{x:+.4f}<br>"
                       "%{text}<extra></extra>"),
    ))
    fig_bar.add_vline(x=0, line_width=1, line_color="black")
    fig_bar.add_vline(x=0.05, line_width=1, line_dash="dot", line_color="gray")
    fig_bar.add_vline(x=-0.05, line_width=1, line_dash="dot", line_color="gray")
    fig_bar.update_layout(
        title=(f"Phase 1: Descriptor vs |SE| Spearman Correlations<br>"
               f"<sub>Go/no-go gate: {signs_ok}/{len(descriptor_cols)} "
               f"descriptors match chemical intuition (need ≥2)</sub>"),
        xaxis_title="Spearman ρ with |SE|",
        yaxis_title="Descriptor",
        height=500, width=900,
        template="plotly_white",
    )
    fig_bar.write_html(os.path.join(phase_dir, "spearman_correlations.html"))

    # ---- Plotly: interactive scatter grid with SMILES tooltips ----
    # Subsample for HTML size
    plot_n = min(3000, len(desc_df))
    plot_idx = np.random.RandomState(seed).choice(len(desc_df), plot_n, replace=False)
    pdf = desc_df.iloc[plot_idx].reset_index(drop=True)
    pdf["smi_a"] = df.iloc[pdf["idx"].values]["smi_a"].values
    pdf["smi_b"] = df.iloc[pdf["idx"].values]["smi_b"].values

    fig_scatter = make_subplots(
        rows=2, cols=3,
        subplot_titles=[
            (f"{c}  (ρ = {corr_df.loc[c, 'spearman_r']:+.3f})"
             if not pd.isna(corr_df.loc[c, "spearman_r"])
             else f"{c}  (ρ = n/a)")
            for c in key_descs
        ],
    )
    for idx, col in enumerate(key_descs):
        r, c = idx // 3 + 1, idx % 3 + 1
        fig_scatter.add_trace(
            go.Scattergl(
                x=pdf[col], y=pdf["SE_abs"],
                mode="markers",
                marker=dict(size=4, opacity=0.4, color=pdf["SE_abs"],
                            colorscale="Viridis", showscale=False),
                customdata=pdf[["smi_a", "smi_b"]].values,
                hovertemplate=(f"<b>{col}</b> = %{{x:.3f}}<br>"
                               "|SE| = %{y:.4f}<br>"
                               "A: %{customdata[0]}<br>"
                               "B: %{customdata[1]}<extra></extra>"),
                showlegend=False,
            ),
            row=r, col=c,
        )
        fig_scatter.update_xaxes(title_text=col, row=r, col=c)
        fig_scatter.update_yaxes(title_text="|SE|", row=r, col=c)
    fig_scatter.update_layout(
        title=f"Phase 1: Structural descriptors vs |SE| (n={plot_n:,} sampled pairs)",
        height=800, width=1200, template="plotly_white",
    )
    fig_scatter.write_html(os.path.join(phase_dir, "descriptor_scatter_interactive.html"))

    # ---- Split train/val/test ----
    if split_method == "scaffold":
        print("\n[1.5] Splitting train/val/test (Bemis-Murcko scaffold, "
              "grouping by both ligands to avoid leakage)")

        all_smiles = pd.unique(pd.concat([df["smi_a"], df["smi_b"]], ignore_index=True))
        scaffold_of = {}
        for smi in all_smiles:
            try:
                scaffold_of[smi] = MurckoScaffold.MurckoScaffoldSmiles(smi=smi)
            except Exception:
                scaffold_of[smi] = smi  # fall back to full SMILES if scaffold fails

        df["scaffold_a"] = df["smi_a"].map(scaffold_of)
        df["scaffold_b"] = df["smi_b"].map(scaffold_of)

        # Size val/test by PAIR COUNT, not scaffold count (see
        # hybrid_topo_rbfe.py::phase1_load for the same logic/rationale).
        degree = pd.concat([df["scaffold_a"], df["scaffold_b"]]).value_counts()
        scaffolds = degree.index.to_numpy()
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(scaffolds))
        scaffolds = scaffolds[order]
        degrees = degree.values[order]

        n_pairs = len(df)
        target_val = val_frac * n_pairs
        target_test = test_frac * n_pairs

        val_s, test_s, train_s = set(), set(), set()
        val_deg = test_deg = 0
        for s, d in zip(scaffolds, degrees):
            if val_deg < target_val:
                val_s.add(s)
                val_deg += d
            elif test_deg < target_test:
                test_s.add(s)
                test_deg += d
            else:
                train_s.add(s)

        def _scaffold_split(s):
            if s in val_s:
                return "val"
            if s in test_s:
                return "test"
            if s in train_s:
                return "train"
            return None

        df["split_a"] = df["scaffold_a"].map(_scaffold_split)
        df["split_b"] = df["scaffold_b"].map(_scaffold_split)

        n_before = len(df)
        df = df[df["split_a"] == df["split_b"]].reset_index(drop=True)
        n_dropped = n_before - len(df)
        df["split"] = df["split_a"]
        df = df.drop(columns=["scaffold_a", "scaffold_b", "split_a", "split_b"])

        print(f"    Dropped {n_dropped:,} pairs straddling scaffold splits "
              f"({n_dropped / n_before:.1%})")
    else:
        print("\n[1.5] Splitting train/val/test (grouped by anchor smi_a to avoid leakage)")
        unique_a = df["smi_a"].unique()
        rng = np.random.RandomState(seed)
        rng.shuffle(unique_a)
        n_a = len(unique_a)
        n_test = int(n_a * test_frac)
        n_val = int(n_a * val_frac)
        test_anchors = set(unique_a[:n_test])
        val_anchors = set(unique_a[n_test:n_test + n_val])

        def assign(smi_a):
            if smi_a in test_anchors:
                return "test"
            if smi_a in val_anchors:
                return "val"
            return "train"
        df["split"] = df["smi_a"].map(assign)

    print(f"    Train: {(df.split=='train').sum():,}  "
          f"Val: {(df.split=='val').sum():,}  "
          f"Test: {(df.split=='test').sum():,}")

    # Save split as pickle (avoids pyarrow/fastparquet dependency).
    df.to_pickle(os.path.join(phase_dir, f"cleaned_split_{split_method}.pkl"))
    return df


# =============================================================================
# =============================================================================
# PHASE 2 — Molecular Graph Construction
# =============================================================================
# =============================================================================

ATOM_VOCAB = {
    "atomic_num": list(range(1, 120)),
    "degree": list(range(0, 7)),
    "formal_charge": [-2, -1, 0, 1, 2],
    "hybridization": [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
    ],
    "num_hs": list(range(0, 5)),
}

BOND_VOCAB = {
    "bond_type": [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ],
}


def _one_hot(v, vocab):
    """One-hot encode v with an extra OOV bin at the end."""
    x = [0] * (len(vocab) + 1)
    if v in vocab:
        x[vocab.index(v)] = 1
    else:
        x[-1] = 1  # OOV bin
    return x


def atom_features(atom) -> List[float]:
    return (
        _one_hot(atom.GetAtomicNum(), ATOM_VOCAB["atomic_num"])
        + _one_hot(atom.GetDegree(), ATOM_VOCAB["degree"])
        + _one_hot(atom.GetFormalCharge(), ATOM_VOCAB["formal_charge"])
        + _one_hot(atom.GetHybridization(), ATOM_VOCAB["hybridization"])
        + _one_hot(atom.GetTotalNumHs(), ATOM_VOCAB["num_hs"])
        + [int(atom.GetIsAromatic()),
           int(atom.IsInRing()),
           float(atom.GetNumRadicalElectrons())]
    )


def bond_features(bond) -> List[float]:
    return (
        _one_hot(bond.GetBondType(), BOND_VOCAB["bond_type"])
        + [int(bond.GetIsConjugated()),
           int(bond.IsInRing()),
           int(bond.GetStereo() != Chem.rdchem.BondStereo.STEREONONE)]
    )


ATOM_DIM = (
    len(ATOM_VOCAB["atomic_num"]) + 1
    + len(ATOM_VOCAB["degree"]) + 1
    + len(ATOM_VOCAB["formal_charge"]) + 1
    + len(ATOM_VOCAB["hybridization"]) + 1
    + len(ATOM_VOCAB["num_hs"]) + 1
    + 3  # aromatic, in_ring, radicals
)
BOND_DIM = len(BOND_VOCAB["bond_type"]) + 1 + 3  # conjugated, in_ring, stereo


def smiles_to_graph(smi: str):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    nf = [atom_features(a) for a in mol.GetAtoms()]
    if not nf:
        return None
    ei, ef = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bf = bond_features(bond)
        ei.extend([[i, j], [j, i]])
        ef.extend([bf, bf])
    if not ei:
        ei = [[0, 0]]
        ef = [[0.0] * BOND_DIM]
    return {
        "node_feats": torch.tensor(nf, dtype=torch.float32),
        "edge_index": torch.tensor(ei, dtype=torch.long).T,
        "edge_feats": torch.tensor(ef, dtype=torch.float32),
        "num_nodes": len(nf),
    }


def batch_graphs(graphs):
    nf, ei, ef, bv = [], [], [], []
    off = 0
    for i, g in enumerate(graphs):
        nf.append(g["node_feats"])
        ei.append(g["edge_index"] + off)
        ef.append(g["edge_feats"])
        bv.append(torch.full((g["num_nodes"],), i, dtype=torch.long))
        off += g["num_nodes"]
    return {
        "node_feats": torch.cat(nf, dim=0),
        "edge_index": torch.cat(ei, dim=1),
        "edge_feats": torch.cat(ef, dim=0),
        "batch": torch.cat(bv, dim=0),
    }


class PairDataset(Dataset):
    """Precompute and cache graphs; reuse anchor graphs efficiently."""

    def __init__(self, df: pd.DataFrame):
        self.smi_a = df["smi_a"].tolist()
        self.smi_b = df["smi_b"].tolist()
        self.se = df["SE_abs"].astype(np.float32).tolist()
        self._cache: Dict[str, dict] = {}
        self.valid = []
        for i in range(len(self.smi_a)):
            if self._get(self.smi_a[i]) and self._get(self.smi_b[i]):
                self.valid.append(i)

    def _get(self, smi):
        if smi not in self._cache:
            self._cache[smi] = smiles_to_graph(smi)
        return self._cache[smi]

    def __len__(self):
        return len(self.valid)

    def __getitem__(self, i):
        r = self.valid[i]
        return (self._get(self.smi_a[r]),
                self._get(self.smi_b[r]),
                torch.tensor(self.se[r], dtype=torch.float32))


def collate_pairs(batch):
    ga, gb, y = zip(*batch)
    return batch_graphs(ga), batch_graphs(gb), torch.stack(y)


# =============================================================================
# =============================================================================
# PHASE 3 — Hyperspherical Encoder (Option C: hybrid loss)
# =============================================================================
# =============================================================================

class MPNNLayer(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim):
        super().__init__()
        self.msg = nn.Sequential(
            nn.Linear(node_dim + edge_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.upd = nn.GRUCell(hidden_dim, node_dim)

    def forward(self, x, edge_index, edge_attr):
        src, dst = edge_index
        m = self.msg(torch.cat([x[src], edge_attr], dim=-1))
        agg = torch.zeros(x.size(0), m.size(1), device=x.device)
        agg.index_add_(0, dst, m)
        return self.upd(agg, x)


class HypersphereEncoder(nn.Module):
    """GNN encoder → projection → optional L2 norm (hypersphere)."""

    def __init__(self, atom_dim, bond_dim, hidden_dim, num_layers,
                 embed_dim, dropout, l2_normalize: bool = True):
        super().__init__()
        self.node_embed = nn.Linear(atom_dim, hidden_dim)
        self.edge_embed = nn.Linear(bond_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [MPNNLayer(hidden_dim, hidden_dim, hidden_dim) for _ in range(num_layers)]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.l2_normalize = l2_normalize

    def forward(self, graph):
        x = self.node_embed(graph["node_feats"])
        e = self.edge_embed(graph["edge_feats"])
        ei = graph["edge_index"]
        for layer, norm in zip(self.layers, self.norms):
            x = norm(x + self.dropout(layer(x, ei, e)))
        # mean pool
        b = graph["batch"]
        n = int(b.max().item()) + 1
        pooled = torch.zeros(n, x.size(1), device=x.device)
        cnt = torch.zeros(n, 1, device=x.device)
        pooled.index_add_(0, b, x)
        cnt.index_add_(0, b, torch.ones(x.size(0), 1, device=x.device))
        pooled = pooled / cnt.clamp(min=1)
        h = self.proj(pooled)
        if self.l2_normalize:
            h = F.normalize(h, p=2, dim=-1)
        return h


class RegressionHead(nn.Module):
    """Predicts |SE| from (h_A, h_B) geometry."""

    def __init__(self, embed_dim, hidden=64):
        super().__init__()
        # Inputs: [|h_A - h_B|, h_A * h_B, cos_sim]  → 2*embed_dim + 1
        self.net = nn.Sequential(
            nn.Linear(2 * embed_dim + 1, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, h_a, h_b):
        diff = (h_a - h_b).abs()
        prod = h_a * h_b
        cos = (h_a * h_b).sum(-1, keepdim=True)
        return self.net(torch.cat([diff, prod, cos], dim=-1)).squeeze(-1)


class ClusterCenters(nn.Module):
    """
    EMA-maintained cluster centers on the hypersphere.
    Centers are initialized by K-means on a warmup batch, then updated with
    each batch via EMA of assigned embeddings (re-normalized to unit sphere).
    """

    def __init__(self, num_clusters: int, embed_dim: int, ema: float = 0.9):
        super().__init__()
        self.num_clusters = num_clusters
        self.ema = ema
        self.register_buffer("centers", F.normalize(torch.randn(num_clusters, embed_dim),
                                                    p=2, dim=-1))
        self.register_buffer("initialized", torch.tensor(False))

    @torch.no_grad()
    def initialize(self, embeddings: torch.Tensor):
        emb = embeddings.detach().cpu().numpy()
        if len(emb) < self.num_clusters:
            return
        km = KMeans(n_clusters=self.num_clusters, n_init=5, random_state=0).fit(emb)
        c = torch.tensor(km.cluster_centers_, dtype=torch.float32,
                         device=self.centers.device)
        self.centers.copy_(F.normalize(c, p=2, dim=-1))
        self.initialized.fill_(True)

    @torch.no_grad()
    def update(self, embeddings: torch.Tensor):
        if not self.initialized:
            self.initialize(embeddings)
            return
        sims = embeddings @ self.centers.T
        assign = sims.argmax(dim=-1)
        new_centers = self.centers.clone()
        for k in range(self.num_clusters):
            mask = assign == k
            if mask.any():
                mean_k = embeddings[mask].mean(dim=0)
                new_centers[k] = self.ema * self.centers[k] + (1 - self.ema) * mean_k
        self.centers.copy_(F.normalize(new_centers, p=2, dim=-1))

    def dispersion_loss(self) -> torch.Tensor:
        """Mean off-diagonal cosine similarity between centers (minimize)."""
        sims = self.centers @ self.centers.T
        K = self.num_clusters
        mask = ~torch.eye(K, dtype=torch.bool, device=sims.device)
        return sims[mask].mean()

    def ood_score(self, embeddings: torch.Tensor) -> torch.Tensor:
        """1 - max cosine similarity to any center. Higher = more OOD."""
        sims = embeddings @ self.centers.T
        return 1.0 - sims.max(dim=-1).values


# -----------------------------------------------------------------------------
# Hybrid loss
# -----------------------------------------------------------------------------

def continuous_contrastive_loss(h_a: torch.Tensor, h_b: torch.Tensor,
                                se: torch.Tensor, se_scale: float) -> torch.Tensor:
    """
    Target cosine similarity = cos(pi * SE_norm).
      SE_norm = 0  →  target_cos = 1   (identical embeddings)
      SE_norm = 1  →  target_cos = -1  (antipodal)
    """
    se_norm = (se / max(se_scale, 1e-8)).clamp(0, 1)
    target_cos = torch.cos(np.pi * se_norm)
    pred_cos = (h_a * h_b).sum(dim=-1)
    return F.mse_loss(pred_cos, target_cos)


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------

def _build_loader(df, args, shuffle):
    ds = PairDataset(df)
    print(f"    Built dataset with {len(ds):,} valid pairs "
          f"(cache hit rate: {len(ds._cache)} unique mols)")
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=shuffle,
        collate_fn=collate_pairs, num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return ds, loader


def phase3_train(df: pd.DataFrame, args, output_dir: str,
                 run_name: str = "hypersphere") -> Dict:
    """Train the hypersphere encoder (Option C hybrid). Returns ckpt paths."""
    print("\n" + "=" * 70)
    print(f"PHASE 3 — Hypersphere Encoder Training ({run_name})")
    print("=" * 70)

    phase_dir = os.path.join(output_dir, f"phase3_{run_name}")
    os.makedirs(phase_dir, exist_ok=True)

    train_df = df[df.split == "train"].reset_index(drop=True)
    val_df = df[df.split == "val"].reset_index(drop=True)

    print("\n[3.1] Building data loaders")
    _, train_loader = _build_loader(train_df, args, shuffle=True)
    _, val_loader = _build_loader(val_df, args, shuffle=False)

    se_scale = float(np.quantile(train_df["SE_abs"].values, 0.99))
    print(f"    SE scale (q99): {se_scale:.6f}")

    # Model
    l2 = (run_name != "flat_ablation")
    encoder = HypersphereEncoder(
        ATOM_DIM, BOND_DIM, args.hidden_dim, args.num_layers,
        args.embed_dim, args.dropout, l2_normalize=l2,
    ).to(DEVICE)
    head = RegressionHead(args.embed_dim).to(DEVICE)
    clusters = ClusterCenters(args.num_clusters, args.embed_dim,
                              ema=args.cluster_ema).to(DEVICE)
    print(f"    Params: encoder={sum(p.numel() for p in encoder.parameters()):,}  "
          f"head={sum(p.numel() for p in head.parameters()):,}  "
          f"L2-norm={l2}")

    optim = torch.optim.Adam(
        list(encoder.parameters()) + list(head.parameters()),
        lr=args.lr, weight_decay=1e-5,
    )

    history = []
    best_val = float("inf")
    best_ckpt = os.path.join(phase_dir, "best.pt")
    patience = 0

    print("\n[3.2] Training")
    for epoch in range(args.epochs):
        encoder.train(); head.train()
        t0 = time.time()
        sums = {"total": 0.0, "contrast": 0.0, "disp": 0.0, "reg": 0.0, "n": 0}

        for ga, gb, y in train_loader:
            ga = {k: v.to(DEVICE) for k, v in ga.items()}
            gb = {k: v.to(DEVICE) for k, v in gb.items()}
            y = y.to(DEVICE)

            h_a = encoder(ga)
            h_b = encoder(gb)

            # Contrastive (on hypersphere)
            if l2:
                L_c = continuous_contrastive_loss(h_a, h_b, y, se_scale)
            else:
                # Flat Euclidean analogue: target distance = SE_norm, predicted = ||h_a - h_b||
                se_norm = (y / max(se_scale, 1e-8)).clamp(0, 1)
                pred_d = (h_a - h_b).norm(dim=-1)
                L_c = F.mse_loss(pred_d / (pred_d.detach().max() + 1e-6), se_norm)

            # Update cluster centers with a mixed pool of embeddings
            with torch.no_grad():
                pool = torch.cat([h_a.detach(), h_b.detach()], dim=0)
                if l2:
                    clusters.update(pool)

            # Dispersion (only meaningful on hypersphere)
            L_d = clusters.dispersion_loss() if l2 else torch.tensor(0.0, device=DEVICE)

            # Regression head
            pred_se = head(h_a, h_b)
            L_r = F.mse_loss(pred_se, y)

            L = (args.w_contrastive * L_c
                 + args.w_dispersion * L_d
                 + args.w_regression * L_r)

            optim.zero_grad()
            L.backward()
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(head.parameters()), 5.0)
            optim.step()

            bs = y.size(0)
            sums["total"] += L.item() * bs
            sums["contrast"] += L_c.item() * bs
            sums["disp"] += float(L_d.item()) * bs
            sums["reg"] += L_r.item() * bs
            sums["n"] += bs

        # ---- Validation ----
        encoder.eval(); head.eval()
        val_mse = 0.0; val_n = 0
        val_preds = []; val_true = []
        with torch.no_grad():
            for ga, gb, y in val_loader:
                ga = {k: v.to(DEVICE) for k, v in ga.items()}
                gb = {k: v.to(DEVICE) for k, v in gb.items()}
                y = y.to(DEVICE)
                h_a = encoder(ga); h_b = encoder(gb)
                p = head(h_a, h_b)
                val_mse += F.mse_loss(p, y, reduction="sum").item()
                val_n += y.size(0)
                val_preds.append(p.cpu().numpy())
                val_true.append(y.cpu().numpy())
        val_mse /= max(val_n, 1)
        val_preds = np.concatenate(val_preds)
        val_true = np.concatenate(val_true)
        val_spear = spearmanr(val_preds, val_true).correlation

        dt = time.time() - t0
        n = max(sums["n"], 1)
        rec = {
            "epoch": epoch + 1,
            "train_total": sums["total"] / n,
            "train_contrast": sums["contrast"] / n,
            "train_disp": sums["disp"] / n,
            "train_reg": sums["reg"] / n,
            "val_mse": val_mse,
            "val_spearman": float(val_spear),
            "time_sec": dt,
        }
        history.append(rec)
        print(f"  ep {epoch+1:3d}  "
              f"train {rec['train_total']:.4f} "
              f"(c={rec['train_contrast']:.3f} d={rec['train_disp']:.3f} "
              f"r={rec['train_reg']:.4f})  "
              f"val_mse {val_mse:.4f}  val_ρ {val_spear:.3f}  "
              f"({dt:.0f}s)")

        if val_mse < best_val:
            best_val = val_mse
            torch.save({
                "encoder": encoder.state_dict(),
                "head": head.state_dict(),
                "clusters": clusters.state_dict(),
                "args": vars(args),
                "se_scale": se_scale,
                "l2_normalize": l2,
            }, best_ckpt)
            patience = 0
        else:
            patience += 1
            if patience >= args.patience:
                print(f"  Early stop at epoch {epoch+1}")
                break

    with open(os.path.join(phase_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    return {
        "ckpt": best_ckpt,
        "phase_dir": phase_dir,
        "se_scale": se_scale,
        "l2_normalize": l2,
    }


# =============================================================================
# =============================================================================
# PHASE 4 — Geometric Validation
# =============================================================================
# =============================================================================

def _load_model(ckpt_path: str, args):
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    l2 = ckpt["l2_normalize"]
    encoder = HypersphereEncoder(
        ATOM_DIM, BOND_DIM, args.hidden_dim, args.num_layers,
        args.embed_dim, args.dropout, l2_normalize=l2,
    ).to(DEVICE)
    encoder.load_state_dict(ckpt["encoder"])
    encoder.eval()
    head = RegressionHead(args.embed_dim).to(DEVICE)
    head.load_state_dict(ckpt["head"])
    head.eval()
    clusters = ClusterCenters(args.num_clusters, args.embed_dim,
                              ema=args.cluster_ema).to(DEVICE)
    clusters.load_state_dict(ckpt["clusters"])
    return encoder, head, clusters, ckpt["se_scale"], l2


@torch.no_grad()
def _encode_all(encoder, head, clusters, loader):
    H_a, H_b, Y, PRED = [], [], [], []
    for ga, gb, y in loader:
        ga = {k: v.to(DEVICE) for k, v in ga.items()}
        gb = {k: v.to(DEVICE) for k, v in gb.items()}
        h_a = encoder(ga); h_b = encoder(gb)
        p = head(h_a, h_b)
        H_a.append(h_a.cpu().numpy())
        H_b.append(h_b.cpu().numpy())
        Y.append(y.numpy())
        PRED.append(p.cpu().numpy())
    return (np.concatenate(H_a), np.concatenate(H_b),
            np.concatenate(Y), np.concatenate(PRED))


def phase4_validate(df: pd.DataFrame, args, ckpt_info: Dict,
                    output_dir: str, name: str = "hypersphere") -> Dict:
    print("\n" + "=" * 70)
    print(f"PHASE 4 — Geometric Validation ({name})")
    print("=" * 70)

    phase_dir = os.path.join(output_dir, f"phase4_{name}")
    os.makedirs(phase_dir, exist_ok=True)

    encoder, head, clusters, se_scale, l2 = _load_model(ckpt_info["ckpt"], args)

    test_df = df[df.split == "test"].reset_index(drop=True)
    _, test_loader = _build_loader(test_df, args, shuffle=False)

    print("\n[4.1] Encoding test set")
    H_a, H_b, Y, pred = _encode_all(encoder, head, clusters, test_loader)

    # Angular distance (or Euclidean if flat)
    if l2:
        cos_sim = (H_a * H_b).sum(axis=-1)
        angular_dist = np.arccos(np.clip(cos_sim, -1, 1))
        dist_name = "angular_distance"
    else:
        angular_dist = np.linalg.norm(H_a - H_b, axis=-1)
        dist_name = "euclidean_distance"

    # OOD scores
    centers = clusters.centers.detach().cpu().numpy()
    sims_a = H_a @ centers.T
    sims_b = H_b @ centers.T
    ood_a = 1.0 - sims_a.max(axis=-1)
    ood_b = 1.0 - sims_b.max(axis=-1)
    ood_max = np.maximum(ood_a, ood_b)  # original: max of individual OOD

    # Cross-cluster OOD: angular distance between the cluster centers that A and B
    # each belong to. Hard pairs span two different clusters (large angle); easy
    # pairs share the same cluster (angle = 0). This is the correct pair-level OOD
    # signal — it penalizes cross-boundary transformations directly.
    k_a = sims_a.argmax(axis=-1)           # nearest cluster for each A
    k_b = sims_b.argmax(axis=-1)           # nearest cluster for each B
    c_ka = centers[k_a]                    # (N, D) cluster center vectors for A
    c_kb = centers[k_b]                    # (N, D) cluster center vectors for B
    cross_cos = (c_ka * c_kb).sum(axis=-1) # cosine between assigned cluster centers
    ood_cross = np.arccos(np.clip(cross_cos, -1.0, 1.0))  # 0 = same cluster, π = opposite
    ood = ood_cross  # use cross-cluster OOD as the primary pair OOD score

    def _corr(x, y, label):
        pr = pearsonr(x, y)
        sr = spearmanr(x, y)
        print(f"    {label:>30}:  Pearson r={pr.statistic:+.3f} (p={pr.pvalue:.1e})  "
              f"Spearman ρ={sr.correlation:+.3f} (p={sr.pvalue:.1e})")
        return {"pearson_r": float(pr.statistic), "pearson_p": float(pr.pvalue),
                "spearman_r": float(sr.correlation), "spearman_p": float(sr.pvalue)}

    print("\n[4.2] Correlations with |SE|:")
    results = {
        dist_name: _corr(angular_dist, Y, dist_name),
        "ood_cross_cluster": _corr(ood_cross, Y, "ood_cross_cluster"),
        "ood_max_individual": _corr(ood_max, Y, "ood_max_individual"),
        "regression_head": _corr(pred, Y, "regression_head_pred"),
    }
    results["test_mae"] = float(mean_absolute_error(Y, pred))
    results["l2_normalize"] = l2

    # Save scores
    np.savez(os.path.join(phase_dir, "scores.npz"),
             H_a=H_a, H_b=H_b, Y=Y, pred=pred,
             dist=angular_dist, ood_cross=ood_cross, ood_max=ood_max)

    with open(os.path.join(phase_dir, "correlations.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Visualization: PCA of embeddings colored by |SE| (per-ligand OOD)
    print("\n[4.3] Visualizing embedding geometry (PCA 2D + 3D sphere + stereographic)")
    all_embed = np.vstack([H_a, H_b])
    pca = PCA(n_components=2, random_state=0).fit(all_embed)
    emb_2d = pca.transform(all_embed)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    # Left: colored by OOD
    ax = axes[0]
    ood_all = np.concatenate([ood_a, ood_b])
    sc = ax.scatter(emb_2d[:, 0], emb_2d[:, 1], c=ood_all, s=2, alpha=0.4,
                    cmap="viridis")
    centers_2d = pca.transform(centers)
    ax.scatter(centers_2d[:, 0], centers_2d[:, 1], c="red", s=120, marker="*",
               edgecolor="black", linewidth=0.8, label="cluster centers")
    plt.colorbar(sc, ax=ax, label="OOD score")
    ax.set_title(f"Embedding space — colored by OOD ({name})")
    ax.legend()

    # Right: colored by SE of the pair (average per-ligand SE involvement)
    ax = axes[1]
    # Map SE of each pair onto both endpoints
    ligand_se = np.concatenate([Y, Y])
    sc = ax.scatter(emb_2d[:, 0], emb_2d[:, 1], c=ligand_se, s=2, alpha=0.4,
                    cmap="magma",
                    vmin=np.quantile(ligand_se, 0.05),
                    vmax=np.quantile(ligand_se, 0.95))
    ax.scatter(centers_2d[:, 0], centers_2d[:, 1], c="cyan", s=120, marker="*",
               edgecolor="black", linewidth=0.8)
    plt.colorbar(sc, ax=ax, label="|SE|")
    ax.set_title(f"Embedding space — colored by |SE| ({name})")

    plt.tight_layout()
    plt.savefig(os.path.join(phase_dir, "embedding_visualization.png"), dpi=140)
    plt.close()

    # Scatter of predictions
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (x, xl, r) in zip(
            axes,
            [(angular_dist, dist_name, results[dist_name]["spearman_r"]),
             (ood_cross, "OOD cross-cluster", results["ood_cross_cluster"]["spearman_r"]),
             (pred, "Predicted |SE|", results["regression_head"]["spearman_r"])]):
        ax.scatter(x, Y, s=2, alpha=0.2)
        ax.set_xlabel(xl)
        ax.set_ylabel("|SE|")
        ax.set_title(f"{xl} vs |SE|  (ρ = {r:+.3f})")
    plt.tight_layout()
    plt.savefig(os.path.join(phase_dir, "score_vs_se_scatter.png"), dpi=140)
    plt.close()

    # ---- True hypersphere visualizations ----
    if l2:
        _visualize_hypersphere(H_a, H_b, Y, ood_a, ood_b, centers,
                               phase_dir, name)
    else:
        print("    (skipping 3D sphere viz: flat Euclidean ablation)")

    return results


def _visualize_hypersphere(H_a, H_b, Y, ood_a, ood_b, centers,
                           phase_dir, name):
    """
    3D visualization of the unit hypersphere.

    Approach: 3D PCA of the embeddings, then re-normalize each point back onto
    a unit sphere. This preserves angular relationships approximately and gives
    a true spherical picture (vs. flat 2D PCA which collapses the manifold).
    Also produces a stereographic projection from the antipode of the data
    centroid for a static conformal figure.
    """
    all_embed = np.vstack([H_a, H_b])
    ood_all = np.concatenate([ood_a, ood_b])
    se_all = np.concatenate([Y, Y])

    # ---- 3D PCA + re-normalize to unit sphere ----
    pca3 = PCA(n_components=3, random_state=0).fit(all_embed)
    emb_3d = pca3.transform(all_embed)
    emb_3d = emb_3d / (np.linalg.norm(emb_3d, axis=1, keepdims=True) + 1e-9)
    centers_3d = pca3.transform(centers)
    centers_3d = centers_3d / (np.linalg.norm(centers_3d, axis=1, keepdims=True) + 1e-9)

    # Subsample for HTML size
    n_plot = min(8000, len(emb_3d))
    idx = np.random.RandomState(0).choice(len(emb_3d), n_plot, replace=False)
    pts = emb_3d[idx]
    pt_ood = ood_all[idx]
    pt_se = se_all[idx]

    # Wireframe unit sphere for visual anchor
    u, v = np.mgrid[0:2 * np.pi:40j, 0:np.pi:20j]
    sx = np.cos(u) * np.sin(v)
    sy = np.sin(u) * np.sin(v)
    sz = np.cos(v)

    def _sphere_traces(color, cbar_title):
        traces = [
            # Translucent sphere surface
            go.Surface(
                x=sx, y=sy, z=sz,
                opacity=0.08, showscale=False,
                colorscale=[[0, "lightgray"], [1, "lightgray"]],
                hoverinfo="skip",
            ),
            # Embedding points
            go.Scatter3d(
                x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                mode="markers",
                marker=dict(size=2.5, color=color, colorscale="Viridis",
                            opacity=0.75, showscale=True,
                            colorbar=dict(title=cbar_title)),
                name="ligand embeddings",
                hovertemplate=("PC1=%{x:.2f}<br>PC2=%{y:.2f}<br>PC3=%{z:.2f}<br>"
                               f"{cbar_title}=%{{marker.color:.3f}}<extra></extra>"),
            ),
            # Cluster centers
            go.Scatter3d(
                x=centers_3d[:, 0], y=centers_3d[:, 1], z=centers_3d[:, 2],
                mode="markers+text",
                marker=dict(size=10, color="red", symbol="diamond",
                            line=dict(color="black", width=2)),
                text=[f"c{i}" for i in range(len(centers_3d))],
                textposition="top center",
                name="cluster centers",
            ),
        ]
        return traces

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=(f"Colored by OOD score", f"Colored by |SE|"),
        horizontal_spacing=0.05,
    )
    for t in _sphere_traces(pt_ood, "OOD"):
        fig.add_trace(t, row=1, col=1)
    for t in _sphere_traces(np.clip(pt_se, 0, np.quantile(pt_se, 0.95)), "|SE|"):
        fig.add_trace(t, row=1, col=2)

    fig.update_layout(
        title=(f"Hypersphere Embedding ({name}) — 3D PCA re-normalized to S²<br>"
               f"<sub>Ligand embeddings on the learned unit hypersphere; "
               f"red diamonds = cluster centers</sub>"),
        height=700, width=1400, template="plotly_white",
        scene=dict(aspectmode="cube"),
        scene2=dict(aspectmode="cube"),
    )
    fig.write_html(os.path.join(phase_dir, "sphere_3d_interactive.html"))

    # ---- Stereographic projection (static matplotlib figure) ----
    # Project from the antipode of the data centroid so the densest region is
    # well-separated on the 2D plane.
    centroid = emb_3d.mean(axis=0)
    centroid /= np.linalg.norm(centroid) + 1e-9
    pole = -centroid  # project from the antipode

    def stereographic(pts_3d, pole):
        # Rotate so `pole` is at the south pole (0, 0, -1), then project.
        # Rotation: align `pole` with -z via Rodrigues' formula.
        target = np.array([0.0, 0.0, -1.0])
        v = np.cross(pole, target)
        s = np.linalg.norm(v)
        c = np.dot(pole, target)
        if s < 1e-9:
            R = np.eye(3) if c > 0 else -np.eye(3)
        else:
            vx = np.array([[0, -v[2], v[1]],
                           [v[2], 0, -v[0]],
                           [-v[1], v[0], 0]])
            R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s ** 2))
        rot = pts_3d @ R.T
        # Stereographic from south pole -> project to plane z = 0
        denom = 1.0 - rot[:, 2] + 1e-9
        x = rot[:, 0] / denom
        y = rot[:, 1] / denom
        return x, y

    sx_flat, sy_flat = stereographic(emb_3d, pole)
    sxc, syc = stereographic(centers_3d, pole)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, color, cbar in zip(axes,
                                [ood_all, np.clip(se_all, 0, np.quantile(se_all, 0.95))],
                                ["OOD score", "|SE|"]):
        sc = ax.scatter(sx_flat, sy_flat, c=color, s=2, alpha=0.4, cmap="viridis")
        ax.scatter(sxc, syc, c="red", s=140, marker="*",
                   edgecolor="black", linewidth=0.8, label="cluster centers")
        ax.set_xlabel("stereographic x")
        ax.set_ylabel("stereographic y")
        ax.set_title(f"Stereographic projection — colored by {cbar}")
        ax.set_aspect("equal", "box")
        plt.colorbar(sc, ax=ax, label=cbar)
        # Clip to finite viewing window
        lim = np.quantile(np.sqrt(sx_flat ** 2 + sy_flat ** 2), 0.98)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    axes[0].legend()
    plt.suptitle(f"Hypersphere Embedding ({name}) — Stereographic Projection",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(phase_dir, "sphere_stereographic.png"), dpi=140)
    plt.close()

    print(f"    wrote sphere_3d_interactive.html and sphere_stereographic.png")


# =============================================================================
# =============================================================================
# PHASE 5 — Baseline Comparison
# =============================================================================
# =============================================================================

def _tanimoto(smi_a, smi_b):
    ma = Chem.MolFromSmiles(smi_a); mb = Chem.MolFromSmiles(smi_b)
    if ma is None or mb is None:
        return np.nan
    fa = AllChem.GetMorganFingerprintAsBitVect(ma, 2, nBits=2048)
    fb = AllChem.GetMorganFingerprintAsBitVect(mb, 2, nBits=2048)
    return DataStructs.TanimotoSimilarity(fa, fb)


def _lomap_like_score(smi_a, smi_b):
    """Simple LOMAP-style score: MCS coverage weighted by molecular weight similarity.
    Higher score = easier (similar to LOMAP original definition)."""
    ma = Chem.MolFromSmiles(smi_a); mb = Chem.MolFromSmiles(smi_b)
    if ma is None or mb is None:
        return np.nan
    try:
        mcs = rdFMCS.FindMCS([ma, mb], timeout=2)
        min_atoms = min(ma.GetNumHeavyAtoms(), mb.GetNumHeavyAtoms())
        mcs_cov = mcs.numAtoms / max(min_atoms, 1)
    except Exception:
        mcs_cov = 0.0
    mw_a, mw_b = Descriptors.MolWt(ma), Descriptors.MolWt(mb)
    mw_sim = 1.0 - abs(mw_a - mw_b) / max(mw_a + mw_b, 1.0)
    return mcs_cov * mw_sim


def phase5_baselines(df: pd.DataFrame, args, ckpt_info: Dict,
                     output_dir: str) -> Dict:
    print("\n" + "=" * 70)
    print("PHASE 5 — Baseline Comparison")
    print("=" * 70)

    phase_dir = os.path.join(output_dir, "phase5")
    os.makedirs(phase_dir, exist_ok=True)

    test_df = df[df.split == "test"].reset_index(drop=True)
    # For speed, cap at 20k test pairs for the baseline structural calcs
    n_cap = min(20_000, len(test_df))
    test_df = test_df.sample(n=n_cap, random_state=args.seed).reset_index(drop=True)
    print(f"    Using {len(test_df):,} test pairs for baseline comparison")

    # Baselines: compute Tanimoto and LOMAP-like for each pair
    print("\n[5.1] Computing Tanimoto + LOMAP-like baselines")
    tani = np.zeros(len(test_df)); lomap = np.zeros(len(test_df))
    for i in range(len(test_df)):
        row = test_df.iloc[i]
        tani[i] = _tanimoto(row.smi_a, row.smi_b)
        lomap[i] = _lomap_like_score(row.smi_a, row.smi_b)
        if (i + 1) % 2000 == 0:
            print(f"      {i+1}/{len(test_df)}")
    # Both tani and lomap are "easy-ness" scores; invert for difficulty
    tani_diff = 1.0 - tani
    lomap_diff = 1.0 - lomap

    # Learned model scores on the same subset
    print("\n[5.2] Computing NTM OOD + regression scores")
    encoder, head, clusters, se_scale, l2 = _load_model(ckpt_info["ckpt"], args)
    ds = PairDataset(test_df)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        collate_fn=collate_pairs, num_workers=args.num_workers)
    H_a, H_b, Y, pred = _encode_all(encoder, head, clusters, loader)
    centers = clusters.centers.detach().cpu().numpy()
    sims_a = H_a @ centers.T
    sims_b = H_b @ centers.T
    ood_max = np.maximum(1.0 - sims_a.max(-1), 1.0 - sims_b.max(-1))
    k_a = sims_a.argmax(axis=-1)
    k_b = sims_b.argmax(axis=-1)
    cross_cos = (centers[k_a] * centers[k_b]).sum(axis=-1)
    ood_cross = np.arccos(np.clip(cross_cos, -1.0, 1.0))

    # Valid mask (drop NaN baseline rows and mismatched lengths)
    valid = ~(np.isnan(tani) | np.isnan(lomap))
    # Align by length (ds.valid may filter some pairs)
    if len(Y) != len(test_df):
        print(f"    Note: encoder kept {len(Y)}/{len(test_df)} pairs; aligning.")
        keep = np.array(ds.valid)
        tani_diff = tani_diff[keep]
        lomap_diff = lomap_diff[keep]
        Y_true = Y  # already aligned with the encoder output
    else:
        Y_true = Y
    valid = ~(np.isnan(tani_diff) | np.isnan(lomap_diff))
    tani_diff = tani_diff[valid]; lomap_diff = lomap_diff[valid]
    Y_true = Y_true[valid]; pred = pred[valid]
    ood_max = ood_max[valid]; ood_cross = ood_cross[valid]

    # Binary labels: top 25% SE = hard
    thresh = np.quantile(Y_true, 0.75)
    y_bin = (Y_true >= thresh).astype(int)

    def _evaluate(score, name):
        pr = pearsonr(score, Y_true)
        sr = spearmanr(score, Y_true)
        try:
            auroc = roc_auc_score(y_bin, score)
            auprc = average_precision_score(y_bin, score)
            # Threshold score at same prevalence as y_bin (top 25%) for MCC
            score_thresh = np.quantile(score, 0.75)
            y_pred_bin = (score >= score_thresh).astype(int)
            mcc = matthews_corrcoef(y_bin, y_pred_bin)
        except Exception:
            auroc = np.nan; auprc = np.nan; mcc = np.nan
        return {
            "method": name,
            "pearson_r": float(pr.statistic), "pearson_p": float(pr.pvalue),
            "spearman_r": float(sr.correlation), "spearman_p": float(sr.pvalue),
            "auroc_hard_top25": float(auroc),
            "auprc_hard_top25": float(auprc),
            "mcc_hard_top25": float(mcc),
        }

    rows = [
        _evaluate(tani_diff, "Tanimoto (1-sim)"),
        _evaluate(lomap_diff, "LOMAP-like (1-sim)"),
        _evaluate(ood_cross, "NTM cross-cluster OOD"),
        _evaluate(ood_max, "NTM max-individual OOD"),
        _evaluate(pred, "NTM regression head"),
    ]
    comp_df = pd.DataFrame(rows).set_index("method")
    print("\n[5.3] Baseline vs NTM comparison:")
    print(comp_df.round(4).to_string())
    comp_df.to_csv(os.path.join(phase_dir, "baseline_comparison.csv"))

    # Bar chart — one panel per metric
    _metrics = [
        ("spearman_r",        "Spearman ρ"),
        ("auroc_hard_top25",  "AUROC (top 25% hard)"),
        ("auprc_hard_top25",  "AUPRC (top 25% hard)"),
        ("mcc_hard_top25",    "MCC (top 25% hard)"),
    ]
    n_methods = len(comp_df)
    _colors = ["gray"] * 2 + ["steelblue", "darkorange", "seagreen"]
    _colors = (_colors + ["mediumpurple"] * n_methods)[:n_methods]  # pad if needed
    fig, axes = plt.subplots(1, len(_metrics), figsize=(20, 5))
    for ax, (metric, title) in zip(axes, _metrics):
        vals = comp_df[metric].values
        ax.bar(range(n_methods), vals, color=_colors, edgecolor="black")
        ax.set_xticks(range(n_methods))
        ax.set_xticklabels(comp_df.index, rotation=25, ha="right", fontsize=8)
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.axhline(y=0, color="black", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(phase_dir, "baseline_comparison.png"), dpi=140)
    plt.close()

    return {"comparison": comp_df.to_dict(orient="index"),
            "n_test": int(len(Y_true)),
            "hard_threshold": float(thresh)}


# =============================================================================
# Main dispatcher
# =============================================================================

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)
    phases = [int(p) for p in args.phases.split(",") if p.strip()]

    print(f"\nDevice: {DEVICE}")
    print(f"Phases requested: {phases}")
    print(f"Output dir: {args.output_dir}")

    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    df = None
    ckpt_info = None

    # Phase 1
    if 1 in phases:
        df = phase1_characterize(args.input, args.output_dir,
                                  args.sample_size, args.seed,
                                  val_frac=args.val_frac, test_frac=args.test_frac,
                                  split_method=args.split_method)
    else:
        p1_pkl = os.path.join(args.output_dir, "phase1",
                               f"cleaned_split_{args.split_method}.pkl")
        if os.path.exists(p1_pkl):
            print(f"Loading Phase 1 artifact from {p1_pkl}")
            df = pd.read_pickle(p1_pkl)

    # Phase 2 is implicit (graph featurization happens inside Dataset construction).
    if 2 in phases:
        print("\n" + "=" * 70)
        print("PHASE 2 — Molecular Graph Construction")
        print("=" * 70)
        if df is None:
            raise RuntimeError("Phase 2 requires Phase 1 artifacts.")
        sample = df.sample(min(500, len(df)), random_state=args.seed)
        n_ok = 0
        for _, row in sample.iterrows():
            ga = smiles_to_graph(row.smi_a)
            gb = smiles_to_graph(row.smi_b)
            if ga is not None and gb is not None:
                n_ok += 1
        print(f"    Featurization works on {n_ok}/{len(sample)} sampled pairs. "
              f"ATOM_DIM={ATOM_DIM}, BOND_DIM={BOND_DIM}")

    # Phase 3
    if 3 in phases:
        if df is None:
            raise RuntimeError("Phase 3 requires Phase 1 artifacts.")
        ckpt_info = phase3_train(df, args, args.output_dir, run_name="hypersphere")
        if args.flat_ablation:
            phase3_train(df, args, args.output_dir, run_name="flat_ablation")
    else:
        ck = os.path.join(args.output_dir, "phase3_hypersphere", "best.pt")
        if os.path.exists(ck):
            ckpt_info = {"ckpt": ck, "phase_dir": os.path.dirname(ck),
                         "l2_normalize": True, "se_scale": None}

    # Phase 4
    if 4 in phases:
        if ckpt_info is None:
            raise RuntimeError("Phase 4 requires a trained checkpoint from Phase 3.")
        phase4_validate(df, args, ckpt_info, args.output_dir, name="hypersphere")
        if args.flat_ablation:
            flat_ck = os.path.join(args.output_dir, "phase3_flat_ablation", "best.pt")
            if os.path.exists(flat_ck):
                phase4_validate(df, args,
                                {"ckpt": flat_ck, "phase_dir": os.path.dirname(flat_ck)},
                                args.output_dir, name="flat_ablation")

    # Phase 5
    if 5 in phases:
        if ckpt_info is None:
            raise RuntimeError("Phase 5 requires a trained checkpoint from Phase 3.")
        phase5_baselines(df, args, ckpt_info, args.output_dir)

    print("\nAll requested phases complete.")


if __name__ == "__main__":
    main()
