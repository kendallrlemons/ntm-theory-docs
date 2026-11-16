"""Fixed-model, validation-only descriptor learning curve for the v4 split.

No LOMAP scores, graph caches, test features, test predictions, or test metrics
are computed. The complete split is read only to verify benchmark identity;
test labels do not enter fitting, validation, or sample-size selection.
"""

import argparse
import json
import os
import pickle
import time

import numpy as np
import pandas as pd
import sklearn
from rdkit import rdBase
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

import descriptor_rbfe as desc
import hybrid_topo_rbfe as hto
import rbfe_artifacts as artifacts


CURVE_VERSION = 1
MODEL_SETTINGS = {
    "max_leaf_nodes": 15, "max_iter": 50, "learning_rate": .05,
    "l2_regularization": 1., "min_samples_leaf": 30,
    "early_stopping": False, "random_state": 42,
}


def nested_rows(df, sizes, seed=42):
    """One shuffled training prefix; each smaller sample is nested in larger.

    Check that the first cap reproduces the old pandas sample operation.
    Features follow shuffled prefix order; fitting restores source row order
    to match descriptor_rbfe.select_rows.
    """
    if not sizes or sorted(set(sizes)) != list(sizes) or min(sizes) < 1:
        raise ValueError("Training sizes must be unique, positive, and increasing.")
    train = df.loc[df.split == "train"]
    val = df.loc[df.split == "val"]
    if sizes[-1] > len(train):
        raise ValueError(f"Requested {sizes[-1]} training pairs, but only {len(train)} available.")
    selected = train.sample(n=sizes[-1], random_state=seed)
    previous = train.sample(n=sizes[0], random_state=seed)
    if not selected.index[:sizes[0]].equals(previous.index):
        raise ValueError("Installed pandas sampling does not preserve the expected nested prefix.")
    if len(val) < 2:
        raise ValueError("Need at least two validation rows.")
    if not df.index.is_unique:
        raise ValueError("Source row indices must be unique.")
    return selected, val


def save_pickle(path, value):
    temporary = str(path) + ".tmp"
    with open(temporary, "wb") as stream:
        pickle.dump(value, stream)
    os.replace(temporary, path)


def feature_matrix(frame, path, request):
    """Cache descriptors only, with exact row identities and configuration."""
    if os.path.exists(path):
        artifacts.require_cache_manifest(path, request)
        with open(path, "rb") as stream:
            payload = pickle.load(stream)
        if payload["pair_keys"] != artifacts.pair_keys(frame):
            raise ValueError("Learning-curve feature identities mismatch.")
        X = payload["X"]
        if X.shape != (len(frame), len(desc.FEATURE_NAMES)):
            raise ValueError("Learning-curve feature shape mismatch.")
        return X
    pairs = artifacts.pair_keys(frame)
    X = np.empty((len(pairs), len(desc.FEATURE_NAMES)), dtype=np.float32)
    for i, pair in enumerate(pairs):
        X[i] = desc.pair_features(pair)
        if (i+1) % 50_000 == 0:
            print(f"Descriptors {i+1:,}/{len(pairs):,}", flush=True)
    X[~np.isfinite(X)] = np.nan
    save_pickle(path, {"pair_keys": pairs, "X": X})
    artifacts.write_cache_manifest(path, request)
    return X


def fit_point(X_train, y_train, X_val, y_val, seed=42):
    """Independent fixed-capacity fit; no warm start from the smaller sample."""
    train_valid = np.isfinite(X_train).all(axis=1) & np.isfinite(y_train)
    val_valid = np.isfinite(X_val).all(axis=1) & np.isfinite(y_val)
    if train_valid.sum() < 2 or val_valid.sum() < 2:
        raise ValueError("Insufficient descriptor-valid training or validation rows.")
    model = HistGradientBoostingRegressor(**{**MODEL_SETTINGS, "random_state": seed})
    started = time.monotonic()
    model.fit(X_train[train_valid], y_train[train_valid])
    seconds = time.monotonic() - started
    pred = model.predict(X_val[val_valid])
    mean = float(y_train[train_valid].mean())
    metrics = desc.metric_rows(
        y_val[val_valid],
        {"descriptors": pred, "train_mean": np.full(len(pred), mean)},
        {"descriptors", "train_mean"},
    )
    record = {
        "train_selected": len(y_train), "train_eligible": int(train_valid.sum()),
        "val_selected": len(y_val), "val_eligible": int(val_valid.sum()),
        "fit_seconds": seconds, "train_mean": mean, "metrics": metrics,
    }
    all_pred = np.full(len(y_val), np.nan)
    all_pred[val_valid] = pred
    return model, record, all_pred, val_valid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--reference_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--train_sizes", default="100000,500000")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    try:
        sizes = [int(value) for value in args.train_sizes.split(",")]
    except ValueError:
        parser.error("train_sizes must be comma-separated integers")
    if args.threads < 1:
        parser.error("threads must be positive")
    if os.path.realpath(args.reference_dir) == os.path.realpath(args.output_dir):
        parser.error("Use a separate learning-curve output directory.")
    args.seed, args.sample_size = 42, 0
    args.split_method, args.val_frac, args.test_frac = "scaffold", .1, .1
    with open(os.path.join(args.reference_dir, "benchmark.json")) as stream:
        expected = json.load(stream)
    df = hto.phase1_load(args, args.reference_dir)
    benchmark = artifacts.benchmark_identity(df)
    artifacts.require_benchmark({"benchmark": benchmark}, expected)
    train, val = nested_rows(df, sizes, args.seed)
    # Test membership/labels are used above only for provenance. They are not
    # passed to feature extraction or model fitting below.
    request = {
        "curve_version": CURVE_VERSION, "benchmark": benchmark,
        "train_sizes": sizes, "seed": args.seed, "feature_version": desc.FEATURE_VERSION,
        "feature_names": desc.FEATURE_NAMES, "model_settings": MODEL_SETTINGS,
        "rdkit_version": rdBase.rdkitVersion, "sklearn_version": sklearn.__version__,
        "pandas_version": pd.__version__,
        "train_order_sha256": artifacts._hash_rows(artifacts.pair_keys(train)),
        "val_order_sha256": artifacts._hash_rows(artifacts.pair_keys(val)),
    }
    os.makedirs(args.output_dir, exist_ok=True)
    manifest = os.path.join(args.output_dir, "learning_curve_manifest.json")
    if os.path.exists(manifest):
        with open(manifest) as stream:
            if json.load(stream) != request:
                raise ValueError("Learning-curve settings changed; use a fresh output directory.")
    else:
        desc.json_dump(request, manifest)
    desc.json_dump(benchmark, os.path.join(args.output_dir, "benchmark.json"))
    X_train = feature_matrix(train, os.path.join(args.output_dir, "train_features.pkl"),
                             {**request, "partition": "train"})
    X_val = feature_matrix(val, os.path.join(args.output_dir, "val_features.pkl"),
                           {**request, "partition": "val"})
    y_train, y_val = train.SE_abs.to_numpy(), val.SE_abs.to_numpy()
    records, table = [], []
    with threadpool_limits(limits=args.threads):
        for size in sizes:
            path = os.path.join(args.output_dir, f"model_{size}.pkl")
            if os.path.exists(path):
                with open(path, "rb") as stream:
                    saved = pickle.load(stream)
                if saved["request"] != request or saved["train_size"] != size:
                    raise ValueError("Cached learning-curve model settings mismatch.")
                record, pred, valid = saved["record"], saved["predictions"], saved["valid"]
            else:
                # Prefix selection is nested; source-order restoration matches
                # the earlier 100k experiment's fit ordering.
                order = np.argsort(train.index.to_numpy()[:size])
                print(f"Training fixed 15-leaf / 50-iteration model on {size:,} pairs", flush=True)
                model, record, pred, valid = fit_point(
                    X_train[:size][order], y_train[:size][order], X_val, y_val, args.seed)
                saved = {"request": request, "train_size": size, "model": model,
                         "record": record, "predictions": pred, "valid": valid,
                         "benchmark": benchmark}
                save_pickle(path, saved)
            records.append(record)
            output = val[[artifacts.A, artifacts.B, "SE_abs"]].reset_index(drop=True).copy()
            output["descriptors"] = pred
            output["train_mean"] = record["train_mean"]
            output["eligible"] = valid
            output.to_csv(os.path.join(args.output_dir, f"validation_predictions_{size}.csv"), index=False)
            for metrics in record["metrics"]:
                table.append({**{k: v for k, v in record.items() if k != "metrics"}, **metrics})
            pd.DataFrame(table).to_csv(os.path.join(args.output_dir, "validation_learning_curve.csv"), index=False)
            desc.json_dump(records, os.path.join(args.output_dir, "learning_curve.json"))
            print(pd.DataFrame(record["metrics"]).to_string(index=False), flush=True)
    print("Complete: validation only. No LOMAP calls or test evaluation.")


if __name__ == "__main__":
    main()
