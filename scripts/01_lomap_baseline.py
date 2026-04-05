"""
01_lomap_baseline.py
====================
LOMAP-style baseline: compute molecular similarity features between pairs,
then train a simple regression model to predict stderr difference.

Uses RDKit fingerprints + molecular descriptors as a heuristic baseline.

Usage:
    python 01_lomap_baseline.py \
        --data_dir ../data \
        --output_dir ../results/lomap \
        --seed 42
"""

import argparse
import os
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Descriptors, rdMolDescriptors
from rdkit import RDLogger
from tqdm import tqdm
import pickle
import json

RDLogger.DisableLog("rdApp.*")


# =========================================================================
# Feature computation
# =========================================================================

def mol_from_smiles(smi: str):
    """Safely parse SMILES."""
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            Chem.SanitizeMol(mol)
        return mol
    except Exception:
        return None


def compute_fingerprint(mol, radius=2, nbits=2048):
    """Morgan (ECFP-like) fingerprint as numpy array."""
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
    arr = np.zeros(nbits, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def compute_pair_features(smi_a: str, smi_b: str) -> np.ndarray:
    """
    Compute pairwise features between two molecules.

    Features:
      - Tanimoto similarity (Morgan FP)
      - Dice similarity
      - Cosine similarity of FP vectors
      - Difference in molecular descriptors (MW, LogP, HBA, HBD, TPSA, etc.)
      - Number of heavy atom difference
      - Ring count difference
      - Aromatic ring difference
    """
    mol_a = mol_from_smiles(smi_a)
    mol_b = mol_from_smiles(smi_b)

    if mol_a is None or mol_b is None:
        return None

    # Fingerprints
    fp_a = AllChem.GetMorganFingerprintAsBitVect(mol_a, 2, nBits=2048)
    fp_b = AllChem.GetMorganFingerprintAsBitVect(mol_b, 2, nBits=2048)

    tanimoto = DataStructs.TanimotoSimilarity(fp_a, fp_b)
    dice = DataStructs.DiceSimilarity(fp_a, fp_b)

    # Numpy FP for cosine
    arr_a = np.zeros(2048, dtype=np.float32)
    arr_b = np.zeros(2048, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp_a, arr_a)
    DataStructs.ConvertToNumpyArray(fp_b, arr_b)
    cos_sim = np.dot(arr_a, arr_b) / (np.linalg.norm(arr_a) * np.linalg.norm(arr_b) + 1e-8)

    # Molecular descriptors
    def desc(mol):
        return np.array([
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.TPSA(mol),
            Descriptors.NumRotatableBonds(mol),
            mol.GetNumHeavyAtoms(),
            Descriptors.NumAromaticRings(mol),
            Descriptors.RingCount(mol),
            Descriptors.FractionCSP3(mol),
            rdMolDescriptors.CalcNumBridgeheadAtoms(mol),
            Descriptors.NumHeteroatoms(mol),
        ], dtype=np.float32)

    desc_a = desc(mol_a)
    desc_b = desc(mol_b)

    # Feature vector: similarities + descriptor differences + absolute descriptors
    features = np.concatenate([
        [tanimoto, dice, cos_sim],           # 3 similarity features
        desc_b - desc_a,                      # 12 descriptor differences
        np.abs(desc_b - desc_a),             # 12 absolute differences
        (desc_a + desc_b) / 2,               # 12 pair averages
    ])

    return features  # 39 features total


def featurize_dataset(df: pd.DataFrame, desc: str = ""):
    """Compute features for all pairs in a DataFrame."""
    col_a, col_b, col_t = df.columns
    features = []
    targets = []
    skipped = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Featurizing {desc}"):
        feat = compute_pair_features(row[col_a], row[col_b])
        if feat is not None:
            features.append(feat)
            targets.append(row[col_t])
        else:
            skipped += 1

    if skipped > 0:
        print(f"  Skipped {skipped} pairs with invalid SMILES")

    return np.array(features), np.array(targets)


# =========================================================================
# Evaluation
# =========================================================================

def evaluate(y_true, y_pred, prefix=""):
    """Compute regression metrics."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    pearson_r, pearson_p = pearsonr(y_true, y_pred)
    spearman_r, spearman_p = spearmanr(y_true, y_pred)

    metrics = {
        f"{prefix}rmse": rmse,
        f"{prefix}mae": mae,
        f"{prefix}r2": r2,
        f"{prefix}pearson_r": pearson_r,
        f"{prefix}pearson_p": pearson_p,
        f"{prefix}spearman_r": spearman_r,
        f"{prefix}spearman_p": spearman_p,
    }

    print(f"  RMSE:      {rmse:.4f}")
    print(f"  MAE:       {mae:.4f}")
    print(f"  R²:        {r2:.4f}")
    print(f"  Pearson r: {pearson_r:.4f} (p={pearson_p:.2e})")
    print(f"  Spearman ρ:{spearman_r:.4f} (p={spearman_p:.2e})")

    return metrics


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="../data")
    parser.add_argument("--output_dir", default="../results/lomap")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    np.random.seed(args.seed)

    # Load splits
    print("Loading data...")
    df_train = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
    df_val = pd.read_csv(os.path.join(args.data_dir, "val.csv"))
    df_test = pd.read_csv(os.path.join(args.data_dir, "test.csv"))
    print(f"  Train: {len(df_train):,}  Val: {len(df_val):,}  Test: {len(df_test):,}")

    # Featurize
    print("\nComputing LOMAP-style features...")
    X_train, y_train = featurize_dataset(df_train, "train")
    X_val, y_val = featurize_dataset(df_val, "val")
    X_test, y_test = featurize_dataset(df_test, "test")

    print(f"\nFeature matrix shapes:")
    print(f"  Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")

    # Train models
    models = {
        "Ridge": Ridge(alpha=1.0),
        "RandomForest": RandomForestRegressor(
            n_estimators=500, max_depth=15, n_jobs=-1, random_state=args.seed
        ),
        "GBM": GradientBoostingRegressor(
            n_estimators=500, max_depth=6, learning_rate=0.05, random_state=args.seed
        ),
    }

    all_results = {}

    for name, model in models.items():
        print(f"\n{'='*60}")
        print(f"Training: {name}")
        print(f"{'='*60}")

        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t0
        print(f"  Training time: {train_time:.1f}s")

        # Validate
        print(f"\nValidation metrics ({name}):")
        y_val_pred = model.predict(X_val)
        val_metrics = evaluate(y_val, y_val_pred, prefix="val_")

        # Test
        print(f"\nTest metrics ({name}):")
        y_test_pred = model.predict(X_test)
        test_metrics = evaluate(y_test, y_test_pred, prefix="test_")

        all_results[name] = {
            **val_metrics,
            **test_metrics,
            "train_time_s": train_time,
        }

        # Save model
        model_path = os.path.join(args.output_dir, f"{name.lower()}_model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        # Save predictions
        preds_path = os.path.join(args.output_dir, f"{name.lower()}_test_preds.npz")
        np.savez(preds_path, y_true=y_test, y_pred=y_test_pred)

    # Feature importance (GBM)
    if "GBM" in models:
        feat_names = (
            ["tanimoto", "dice", "cosine"]
            + [f"diff_{n}" for n in ["MW", "LogP", "HBA", "HBD", "TPSA",
                                      "RotBonds", "HeavyAtoms", "AromaticRings",
                                      "RingCount", "FracCSP3", "Bridgehead", "Heteroatoms"]]
            + [f"abs_diff_{n}" for n in ["MW", "LogP", "HBA", "HBD", "TPSA",
                                          "RotBonds", "HeavyAtoms", "AromaticRings",
                                          "RingCount", "FracCSP3", "Bridgehead", "Heteroatoms"]]
            + [f"avg_{n}" for n in ["MW", "LogP", "HBA", "HBD", "TPSA",
                                     "RotBonds", "HeavyAtoms", "AromaticRings",
                                     "RingCount", "FracCSP3", "Bridgehead", "Heteroatoms"]]
        )
        importances = models["GBM"].feature_importances_
        sorted_idx = np.argsort(importances)[::-1]
        print(f"\n{'='*60}")
        print("GBM Feature Importance (top 15):")
        print(f"{'='*60}")
        for i in sorted_idx[:15]:
            print(f"  {feat_names[i]:30s} {importances[i]:.4f}")

    # Save summary
    results_path = os.path.join(args.output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
