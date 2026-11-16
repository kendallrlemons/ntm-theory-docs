"""Cluster-only frozen validation selection and test comparison on Fah tasks."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from importlib.metadata import version
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rdkit import rdBase
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from threadpoolctl import threadpool_limits

from descriptor_rbfe import pair_features, lomap_worker
from rbfe_artifacts import file_hash, LOMAP_SCORING_VERSION

SEED = 42


def atomic_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def weighted_rank(x, w):
    _, inverse = np.unique(x, return_inverse=True)
    mass = np.bincount(inverse, weights=w)
    return (np.cumsum(mass) - mass / 2)[inverse]


def correlation(x, y, w):
    # Subtracting a rounded weighted mean can leave tiny residuals for a
    # constant predictor. Such correlations are undefined, not near zero.
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    x = x - np.average(x, weights=w)
    y = y - np.average(y, weights=w)
    den = np.sqrt(np.sum(w * x * x) * np.sum(w * y * y))
    return float(np.sum(w * x * y) / den) if den > 0 else None


def metrics(y, p, w=None):
    if w is None:
        w = np.ones(len(y))
    mask = np.isfinite(y) & np.isfinite(p) & (w > 0)
    y, p, w = y[mask], p[mask], w[mask]
    if len(y) < 3 or w.sum() == 0:
        return {"pearson": None, "spearman": None}
    return {"pearson": correlation(y, p, w),
            "spearman": correlation(weighted_rank(y, w), weighted_rank(p, w), w)}


def bootstrap(frame, columns, comparator, repetitions=1000):
    """Paired dyadic group bootstrap, shared endpoint sampled once for self-group edges."""
    a_col, b_col = (("resample_group_a", "resample_group_b")
                    if "resample_group_a" in frame else ("group_a", "group_b"))
    groups = sorted(set(frame[a_col]) | set(frame[b_col]))
    mapping = {g: i for i, g in enumerate(groups)}
    a, b = frame[a_col].map(mapping).values, frame[b_col].map(mapping).values
    rng = np.random.default_rng(SEED)
    samples = {(c, metric): [] for c in columns if c != comparator
               for metric in ("pearson", "spearman")}
    y = frame.target.values
    for _ in range(repetitions):
        counts = rng.multinomial(len(groups), np.full(len(groups), 1 / len(groups)))
        w = counts[a] * np.where(a == b, 1, counts[b])
        base = metrics(y, frame[comparator].values, w)
        for col in columns:
            if col == comparator:
                continue
            score = metrics(y, frame[col].values, w)
            for metric in base:
                if score[metric] is not None and base[metric] is not None:
                    samples[col, metric].append(score[metric] - base[metric])
    result = {}
    for (col, metric), values in samples.items():
        result[col + "_minus_" + comparator + "_" + metric] = {
            "valid_replicates": len(values),
            "ci95": np.quantile(values, [.025, .975]).tolist() if len(values) >= 200 else None}
    return {"chemistry_groups": len(groups), "replicates": repetitions, "differences": result}


def choose_model(x, frame):
    train, val = frame.split.eq("train").values, frame.split.eq("val").values
    grid = [
        ("hgb15", HistGradientBoostingRegressor(max_leaf_nodes=15, max_iter=100,
                                               early_stopping=False, random_state=SEED)),
        ("hgb31", HistGradientBoostingRegressor(max_leaf_nodes=31, max_iter=100,
                                               early_stopping=False, random_state=SEED)),
        ("rf", RandomForestRegressor(n_estimators=300, min_samples_leaf=5,
                                     max_features=1.0, n_jobs=1, random_state=SEED)),
    ]
    selected, best = None, -np.inf
    search = []
    for name, model in grid:
        model.fit(x[train], np.log(frame.loc[train, "target"]))
        prediction = np.exp(model.predict(x[val]))
        rho = spearmanr(frame.loc[val, "target"], prediction).statistic
        finite = bool(np.isfinite(rho))
        search.append({"model": name, "validation_spearman": float(rho) if finite else None})
        if finite and rho > best:
            best, selected = rho, (name, model)
    if selected is None:
        raise ValueError("No model has a defined validation correlation")
    return selected, search


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benchmark_dir", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()
    if args.workers < 1 or args.threads < 1:
        p.error("workers and threads must be positive")
    manifest_path = args.benchmark_dir / "benchmark.json"
    benchmark = json.loads(manifest_path.read_text())
    if benchmark.get("version") != 2:
        raise ValueError("Reprepare with schema 2; the original split has unusable coverage")
    tables = {}
    skipped = {}
    for name in ("edge", "pilot_complex", "pilot_solvent"):
        path = args.benchmark_dir / (name + ".csv")
        if file_hash(path) != benchmark["tables"][path.name]:
            raise ValueError("Benchmark table hash mismatch")
        frame = pd.read_csv(path, keep_default_na=False)
        # Chemistry hashes may include an empty acyclic scaffold, which must remain a string.
        frame["target"] = pd.to_numeric(frame.target)
        reasons = []
        for split in ("train", "val", "test"):
            part = frame[frame.split.eq(split)]
            if len(part) < 100 or part.variable_group.nunique() < 5:
                reasons.append(f"{split}: fewer than 100 rows or 5 variable groups")
        if reasons:
            skipped[name] = reasons
            continue
        tables[name] = frame
    if not tables:
        raise ValueError(f"No tasks have usable split coverage; no scoring performed: {skipped}")
    request = {
        "benchmark_sha256": file_hash(manifest_path),
        "script_sha256": file_hash(Path(__file__)),
        "descriptor_sha256": file_hash(Path(__file__).with_name("descriptor_rbfe.py")),
        "scorer_sha256": file_hash(Path(__file__).with_name("hybrid_topo_rbfe.py")),
        "packages": {n: version(n) for n in ("numpy", "scipy", "pandas", "scikit-learn", "rdkit", "lomap2", "gufe")},
        "lomap_version": LOMAP_SCORING_VERSION, "mcs_timeout": 5,
    }
    if rdBase.rdkitVersion != benchmark["rdkit"]:
        raise ValueError("Prepare and evaluate with the same RDKit version")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    request_path = args.output_dir / "request.json"
    if request_path.exists():
        if json.loads(request_path.read_text()) != request:
            raise ValueError("Existing run request differs; use a fresh output directory")
    elif any(args.output_dir.iterdir()):
        raise ValueError("Nonempty unmanaged output directory")
    else:
        atomic_json(request_path, request)
    if (args.output_dir / "results.json").exists():
        print("Completed run already exists; no reevaluation performed.")
        return
    all_pairs = sorted({tuple(sorted((r.smi_a, r.smi_b)))
                        for frame in tables.values() for r in frame.itertuples()})
    features_path = args.output_dir / "features.npz"
    feature_manifest = args.output_dir / "features.json"
    if features_path.exists():
        if not feature_manifest.exists() or json.loads(feature_manifest.read_text())["sha256"] != file_hash(features_path):
            raise ValueError("Feature cache checksum missing or mismatched")
        cache = np.load(features_path)
        if cache["pairs"].tolist() != [list(pair) for pair in all_pairs]:
            raise ValueError("Feature identities differ")
        x_all = cache["features"]
    else:
        x_all = np.array([pair_features(pair) for pair in all_pairs])
        temporary = args.output_dir / "features.partial.npz"
        np.savez_compressed(temporary, pairs=np.array(all_pairs), features=x_all)
        temporary.replace(features_path)
        atomic_json(feature_manifest, {"sha256": file_hash(features_path)})
    if not np.isfinite(x_all).all():
        raise ValueError("Nonfinite structure features; inspect chemistry before proceeding")
    feature_map = dict(zip(all_pairs, x_all))
    test_pairs = sorted({tuple(sorted((r.smi_a, r.smi_b))) for frame in tables.values()
                         for r in frame[frame.split.eq("test")].itertuples()})
    scores_path = args.output_dir / "lomap.json"
    scores = json.loads(scores_path.read_text()) if scores_path.exists() else {}
    pending = [pair for pair in test_pairs if json.dumps(pair) not in scores]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        values = pool.map(lomap_worker, [(a, b, 5) for a, b in pending])
        for i, (pair, value) in enumerate(zip(pending, values)):
            scores[json.dumps(pair)] = float(value) if np.isfinite(value) else None
            if (i + 1) % 20 == 0:
                atomic_json(scores_path, scores)
    atomic_json(scores_path, scores)
    results = {"setting": benchmark["setting"], "unsupported_tasks": skipped}
    for task, frame in tables.items():
        x = np.array([feature_map[tuple(sorted((r.smi_a, r.smi_b)))] for r in frame.itertuples()])
        test = frame.split.eq("test")
        prediction = frame.loc[test].copy()
        searches = {}
        modes = ["structure"] if task == "edge" else ["structure", "structure_plus_pilot"]
        with threadpool_limits(limits=args.threads):
            for mode in modes:
                inputs = x
                if mode == "structure_plus_pilot":
                    inputs = np.column_stack([x, np.log(frame.pilot_proxy.values.astype(float))])
                (chosen, model), search = choose_model(inputs, frame)
                searches[mode] = {"selected": chosen, "candidates": search}
                prediction[mode] = np.exp(model.predict(inputs[test]))
                model_path = args.output_dir / f"{task}_{mode}.joblib"
                temporary = model_path.with_suffix(".partial")
                joblib.dump(model, temporary)
                temporary.replace(model_path)
        prediction["training_mean"] = frame.loc[frame.split.eq("train"), "target"].mean()
        prediction["lomap"] = [scores[json.dumps(tuple(sorted((r.smi_a, r.smi_b))))]
                               for r in prediction.itertuples()]
        prediction["lomap"] = pd.to_numeric(prediction.lomap, errors="coerce")
        columns = modes + ["training_mean", "lomap"]
        if task != "edge":
            prediction["pilot_direct"] = prediction.pilot_proxy.astype(float)
            columns.append("pilot_direct")
        prediction.to_csv(args.output_dir / (task + "_predictions.csv"), index=False)
        common = prediction[np.isfinite(prediction.lomap)].copy()
        if len(common) < 3:
            raise ValueError("Too few finite LOMAP scores for paired evaluation")
        result = {
            "selection": searches, "test_rows": len(prediction), "lomap_common_rows": len(common),
            "test_reference_groups": prediction.reference_group.nunique(),
            "test_variable_groups": prediction.variable_group.nunique(),
            "bootstrap_scope": ("variable scaffold groups conditional on known references"
                                if benchmark["setting"] == "known_reference" else
                                "both endpoint groups; few held-out reference families limit inference"),
            "common_metrics": {c: metrics(common.target.values, common[c].values) for c in columns},
            "all_test_metrics": {c: metrics(prediction.target.values, prediction[c].values)
                                 for c in columns if c != "lomap"},
            "common_bootstrap": bootstrap(common, columns, "lomap"),
            "per_source_common": {source: {c: metrics(part.target.values, part[c].values)
                                           for c in columns}
                                  for source, part in common.groupby("series_name")},
        }
        if task != "edge":
            result["pilot_comparison_all_test"] = bootstrap(prediction, columns[:-2] + ["pilot_direct"],
                                                            "pilot_direct")
        results[task] = result
    atomic_json(args.output_dir / "results.json", results)
    print(f"Completed: {args.output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
