"""Compound-level uncertainty model evaluated on held-out scaffold pairs.

The label-structure diagnostic showed y(A, B) ~= c + u_A + v_B, where u and v
are per-compound values for the first and second column. This script:

1. Recovers u, v, c from TRAINING edges only (two-slot least squares).
2. Trains structure -> u and structure -> v models on training compounds.
3. Selects model capacity on VALIDATION pairs (signed Spearman) only.
4. Predicts test pairs as c + u_hat(A) + v_hat(B), and compares |prediction|
   with corrected real LOMAP on the same pairs from the existing v4
   descriptor run (no LOMAP is recomputed here).

The test-split two-slot fit is an evaluation-only ceiling: it never trains or
selects a model.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from rdkit import DataStructs
from scipy.sparse.linalg import lsqr
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

import descriptor_rbfe as desc
import hybrid_topo_rbfe as hto
import label_structure_diagnostic as labels
import rbfe_artifacts as artifacts


MODEL_VERSION = 1
FP_BITS = 256
FEATURE_NAMES = [name for name, _ in desc.DESCRIPTORS] + [f"morgan_{i}" for i in range(FP_BITS)]
LOMAP = "LOMAP (real, 1-sim)"
REFERENCE_METHODS = [LOMAP, "Tanimoto (1-sim)", "descriptors", "train_mean"]


def fit_two_slot(frame):
    """Least-squares y = c + u_first + v_second; u/v share one gauge freedom."""
    codes, names = pd.factorize(pd.concat([frame[artifacts.A], frame[artifacts.B]], ignore_index=True))
    m = len(frame)
    src, dst, n = codes[:m].astype(np.int64), codes[m:].astype(np.int64), len(names)
    y = frame[artifacts.Y].to_numpy(dtype=float)
    if not np.isfinite(y).all():
        raise ValueError("Signed labels must be finite.")
    c = float(y.mean())
    coef = lsqr(labels.design(src, dst, n, "two_slot"), y - c,
                atol=1e-10, btol=1e-10, iter_lim=10_000)[0]
    table = pd.DataFrame({
        "smiles": np.asarray(names), "u": coef[:n], "v": coef[n:],
        "n_first": np.bincount(src, minlength=n), "n_second": np.bincount(dst, minlength=n),
    })
    return table, c, labels.r_squared(y, c + coef[src] + coef[n + dst])


def compound_features(smiles):
    X = np.full((len(smiles), len(FEATURE_NAMES)), np.nan)
    for i, s in enumerate(smiles):
        features = desc.molecule_features(s)
        if features is None:
            continue
        descriptors, fingerprint = features
        bits = np.zeros(fingerprint.GetNumBits())
        DataStructs.ConvertToNumpyArray(fingerprint, bits)
        X[i] = np.r_[descriptors, bits.reshape(FP_BITS, -1).max(axis=1)]
    return X


def pair_predict(models, c, features, frame):
    u_model, v_model = models
    return (c + u_model.predict(features.loc[frame[artifacts.A]].to_numpy())
            + v_model.predict(features.loc[frame[artifacts.B]].to_numpy()))


def signed_metrics(y, prediction):
    return {"r2": labels.r_squared(y, prediction),
            "pearson": float(pearsonr(y, prediction).statistic),
            "spearman": float(spearmanr(y, prediction).statistic)}


def select_models(table, features, c, val, min_edges, seed):
    """Fixed small search; only validation pairs select capacity."""
    first = table[table.n_first >= min_edges]
    second = table[table.n_second >= min_edges]
    if min(len(first), len(second)) < 10:
        raise ValueError("Too few identifiable training compounds for the compound model.")
    y_val = val[artifacts.Y].to_numpy(dtype=float)
    history, best, best_rho = [], None, -np.inf
    for leaves in (7, 15):
        for iterations in (50, 100, 200):
            models = []
            for subset, target in ((first, "u"), (second, "v")):
                model = HistGradientBoostingRegressor(
                    max_leaf_nodes=leaves, max_iter=iterations, learning_rate=.05,
                    l2_regularization=1., min_samples_leaf=20, early_stopping=False,
                    random_state=seed,
                )
                models.append(model.fit(features.loc[subset.smiles].to_numpy(), subset[target].to_numpy()))
            metrics = signed_metrics(y_val, pair_predict(models, c, features, val))
            history.append({"leaves": leaves, "iterations": iterations, **metrics})
            if np.isfinite(metrics["spearman"]) and metrics["spearman"] > best_rho:
                best_rho, best = metrics["spearman"], models
    if best is None:
        raise ValueError("No candidate produced a finite validation correlation.")
    return best, history, {"u_training_compounds": len(first), "v_training_compounds": len(second)}


def load_reference(reference_dir, test, benchmark):
    """Corrected-LOMAP test scores from the v4 descriptor run, aligned by pair."""
    with open(os.path.join(reference_dir, "benchmark.json")) as stream:
        artifacts.require_benchmark({"benchmark": benchmark}, json.load(stream))
    with open(os.path.join(reference_dir, "features.pkl.manifest.json")) as stream:
        version = json.load(stream)["request"].get("lomap_scoring_version")
    if version != artifacts.LOMAP_SCORING_VERSION:
        raise ValueError(f"Reference LOMAP scoring version {version!r} is not the corrected scorer.")
    reference = pd.read_csv(os.path.join(reference_dir, "test_predictions.csv"))
    keys = artifacts.pair_keys(reference)
    if set(keys) != set(artifacts.pair_keys(test)) or len(keys) != len(test):
        raise ValueError("Reference predictions do not cover exactly the test pairs.")
    reference = reference.set_index([artifacts.A, artifacts.B]).loc[artifacts.pair_keys(test)]
    if not np.allclose(reference.SE_abs.to_numpy(), test.SE_abs.to_numpy()):
        raise ValueError("Reference labels differ from the benchmark test labels.")
    missing = [m for m in REFERENCE_METHODS if m not in reference]
    if missing:
        raise ValueError(f"Reference predictions missing columns: {missing}")
    return {m: reference[m].to_numpy(dtype=float) for m in REFERENCE_METHODS}


def compound_agreement(models, features, table_test, min_edges):
    """Rank agreement with test-only recovered values (offset-invariant)."""
    result = {}
    for model, target, count in ((models[0], "u", "n_first"), (models[1], "v", "n_second")):
        subset = table_test[table_test[count] >= min_edges]
        if len(subset) < 3:
            result[target] = {"compounds": len(subset)}
            continue
        predicted = model.predict(features.loc[subset.smiles].to_numpy())
        result[target] = {"compounds": len(subset),
                          "spearman": float(spearmanr(predicted, subset[target]).statistic)}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--reference_dir", required=True, help="Completed results_hybrid_topo_v4 directory")
    parser.add_argument("--lomap_reference_dir", required=True, help="Completed v4 descriptor run with corrected LOMAP")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--min_edges", type=int, default=10)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if args.min_edges < 1 or args.threads < 1:
        parser.error("min_edges and threads must be positive")
    outputs = {os.path.realpath(args.output_dir)}
    if outputs & {os.path.realpath(args.reference_dir), os.path.realpath(args.lomap_reference_dir)}:
        parser.error("Use a fresh output directory.")
    args.seed, args.sample_size = 42, 0
    args.split_method, args.val_frac, args.test_frac = "scaffold", .1, .1
    with open(os.path.join(args.reference_dir, "benchmark.json")) as stream:
        expected = json.load(stream)
    df = hto.phase1_load(args, args.reference_dir)
    benchmark = artifacts.benchmark_identity(df)
    artifacts.require_benchmark({"benchmark": benchmark}, expected)
    train, val, test = (df[df.split == s].reset_index(drop=True) for s in ("train", "val", "test"))
    reference = load_reference(args.lomap_reference_dir, test, benchmark)

    table, c, train_fit_r2 = fit_two_slot(train)
    smiles = pd.unique(pd.concat([df[artifacts.A], df[artifacts.B]]))
    features = pd.DataFrame(compound_features(smiles), index=smiles, columns=FEATURE_NAMES)
    if features.isna().any(axis=None):
        raise ValueError("Compound features failed for some SMILES.")
    with threadpool_limits(limits=args.threads):
        models, history, counts = select_models(table, features, c, val, args.min_edges, args.seed)
        prediction = pair_predict(models, c, features, test)
    # Evaluation only: never used for fitting or selection above.
    table_test, c_test, test_ceiling_r2 = fit_two_slot(test)
    y = test[artifacts.Y].to_numpy(dtype=float)
    scores = {"compound_model": np.abs(prediction), **reference}
    common = np.all([np.isfinite(v) for v in scores.values()], axis=0)
    if common.sum() < 2:
        raise ValueError("Insufficient common test predictions.")
    magnitude = desc.metric_rows(np.abs(y)[common], {k: v[common] for k, v in scores.items()},
                                 {"compound_model", "descriptors", "train_mean"})
    summary = {
        "model_version": MODEL_VERSION, "benchmark": benchmark, "min_edges": args.min_edges,
        "train_two_slot_fit_r2": train_fit_r2, "train_intercept": c, **counts,
        "test_signed": signed_metrics(y, prediction),
        "test_signed_ceiling_r2_from_test_only_fit": test_ceiling_r2,
        "test_compound_rank_agreement": compound_agreement(models, features, table_test, args.min_edges),
        "test_common_pairs": int(common.sum()),
    }
    os.makedirs(args.output_dir, exist_ok=True)
    desc.json_dump(summary, os.path.join(args.output_dir, "summary.json"))
    desc.json_dump(history, os.path.join(args.output_dir, "validation_search.json"))
    pd.DataFrame(magnitude).to_csv(os.path.join(args.output_dir, "comparison.csv"), index=False)
    table.to_csv(os.path.join(args.output_dir, "train_compound_values.csv"), index=False)
    output = test[[artifacts.A, artifacts.B, artifacts.Y, "SE_abs"]].copy()
    output["compound_model_signed"] = prediction
    for name, values in scores.items():
        output[name] = values
    output["common_eligible"] = common
    output.to_csv(os.path.join(args.output_dir, "test_predictions.csv"), index=False)
    print(json.dumps({k: v for k, v in summary.items() if k != "benchmark"}, indent=2))
    print(pd.DataFrame(magnitude).to_string(index=False))
    print("Point estimates only; run the paired group bootstrap before claiming superiority.")


if __name__ == "__main__":
    main()
