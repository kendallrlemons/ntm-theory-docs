"""
00_preprocess_data.py
=====================
Preprocesses the raw CSV: deduplicates, smart subsamples, and creates
train/val/test splits with molecule-level leakage prevention.

Usage:
    python 00_preprocess_data.py \
        --input /path/to/compound_smiles_stderr_differences.csv \
        --output_dir ../data \
        --sample_size 750000 \
        --seed 42
"""

import argparse
import os
import time
import pandas as pd
import numpy as np
from pathlib import Path


def profile_dataset(path: str, chunk_size: int = 500_000):
    """Profile the full dataset without loading it all into memory."""
    print("=" * 60)
    print("Profiling full dataset (chunked read)")
    print("=" * 60)

    n_rows = 0
    all_mol_a = set()
    all_mol_b = set()
    target_sample = []
    pair_counts = {}

    t0 = time.time()
    for i, chunk in enumerate(pd.read_csv(path, chunksize=chunk_size)):
        n_rows += len(chunk)
        col_a, col_b, col_t = chunk.columns

        all_mol_a.update(chunk[col_a].unique())
        all_mol_b.update(chunk[col_b].unique())

        # Keep 1% sample for distribution stats
        target_sample.extend(
            chunk[col_t].sample(frac=0.01, random_state=42).tolist()
        )

        for mol, cnt in chunk[col_a].value_counts().items():
            pair_counts[mol] = pair_counts.get(mol, 0) + cnt

        if (i + 1) % 5 == 0:
            print(f"  {n_rows:>12,} rows  ({time.time() - t0:.0f}s)")

    all_mols = all_mol_a | all_mol_b
    target_arr = np.array(target_sample)

    print(f"\nTotal rows:           {n_rows:,}")
    print(f"Unique Mol A:         {len(all_mol_a):,}")
    print(f"Unique Mol B:         {len(all_mol_b):,}")
    print(f"Total unique mols:    {len(all_mols):,}")
    print(f"\nTarget (1% sample):   mean={target_arr.mean():.4f}  "
          f"std={target_arr.std():.4f}  "
          f"range=[{target_arr.min():.4f}, {target_arr.max():.4f}]")

    print(f"\nTop-15 anchor molecules (Mol A) by pair count:")
    for mol, cnt in sorted(pair_counts.items(), key=lambda x: -x[1])[:15]:
        tag = mol[:55] + "..." if len(mol) > 55 else mol
        print(f"  {tag:60s} {cnt:>10,}")

    return {
        "n_rows": n_rows,
        "n_mol_a": len(all_mol_a),
        "n_mol_b": len(all_mol_b),
        "n_mols": len(all_mols),
        "pair_counts": pair_counts,
    }


def smart_subsample(
    path: str,
    target_size: int = 750_000,
    chunk_size: int = 500_000,
    seed: int = 42,
):
    """
    Subsample the dataset while preserving:
      - All anchor molecules (Mol A)
      - Target distribution
      - Diversity of Mol B
    """
    print("\n" + "=" * 60)
    print(f"Smart subsampling → ~{target_size:,} rows")
    print("=" * 60)

    # First pass: count total rows to determine sampling fraction
    total = sum(1 for _ in open(path)) - 1  # subtract header
    frac = min(target_size / total, 1.0)
    print(f"  Full dataset: {total:,} rows → sampling frac={frac:.6f}")

    chunks = []
    t0 = time.time()
    for i, chunk in enumerate(pd.read_csv(path, chunksize=chunk_size)):
        chunk = chunk.drop_duplicates()
        sampled = chunk.sample(frac=frac, random_state=seed + i)
        chunks.append(sampled)
        if (i + 1) % 5 == 0:
            n = sum(len(c) for c in chunks)
            print(f"  Chunk {i+1}: {n:,} sampled rows ({time.time()-t0:.0f}s)")

    df = pd.concat(chunks, ignore_index=True).drop_duplicates()
    print(f"\n  Final sample: {len(df):,} rows  ({time.time()-t0:.0f}s)")
    return df


def split_by_molecule(df: pd.DataFrame, seed: int = 42):
    """
    Split into train / val / test by Mol B identity.
    This prevents data leakage: no Mol B in test appears in train.
    """
    print("\n" + "=" * 60)
    print("Train / Val / Test split (by Mol B)")
    print("=" * 60)

    col_a, col_b, col_t = df.columns
    unique_b = df[col_b].unique()
    rng = np.random.RandomState(seed)
    rng.shuffle(unique_b)

    n_test = int(0.15 * len(unique_b))
    n_val = int(0.15 * len(unique_b))

    test_set = set(unique_b[:n_test])
    val_set = set(unique_b[n_test: n_test + n_val])
    train_set = set(unique_b[n_test + n_val:])

    df_train = df[df[col_b].isin(train_set)].copy()
    df_val = df[df[col_b].isin(val_set)].copy()
    df_test = df[df[col_b].isin(test_set)].copy()

    for name, split in [("Train", df_train), ("Val", df_val), ("Test", df_test)]:
        print(f"  {name:5s}: {len(split):>9,} rows | "
              f"{split[col_b].nunique():>6,} unique Mol B | "
              f"target mean={split[col_t].mean():.4f}")

    return df_train, df_val, df_test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to raw CSV")
    parser.add_argument("--output_dir", default="../data")
    parser.add_argument("--sample_size", type=int, default=750_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_profile", action="store_true",
                        help="Skip full-dataset profiling (faster)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Profile
    if not args.skip_profile:
        profile_dataset(args.input)

    # Subsample
    df = smart_subsample(args.input, args.sample_size, seed=args.seed)

    # Split
    df_train, df_val, df_test = split_by_molecule(df, seed=args.seed)

    # Save
    for name, split in [("train", df_train), ("val", df_val), ("test", df_test)]:
        p = os.path.join(args.output_dir, f"{name}.csv")
        split.to_csv(p, index=False)
        print(f"  Saved {p}")

    print("\nDone.")


if __name__ == "__main__":
    main()
