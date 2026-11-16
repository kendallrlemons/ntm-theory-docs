"""Frozen campaign pilot comparison: direct, learned pilot-only, and descriptors."""
import argparse
from importlib.metadata import version
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rdkit import rdBase
from threadpoolctl import threadpool_limits

from prepare_fah_campaign_pilot import PROTOCOL, verify_parent_subset
from rbfe_artifacts import file_hash, LOMAP_SCORING_VERSION
from run_fah_benchmark import atomic_json, choose_model, metrics, bootstrap


def inputs_for(mode, x, frame):
    pilot = np.log(frame.pilot_proxy.to_numpy(dtype=float))[:, None]
    if mode == "pilot_only":
        return pilot
    if mode == "structure":
        return x
    if mode == "structure_plus_pilot":
        return np.column_stack([x, pilot])
    raise ValueError(mode)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benchmark_dir", required=True, type=Path)
    p.add_argument("--parent_benchmark_dir", required=True, type=Path)
    p.add_argument("--parent_results_dir", required=True, type=Path)
    p.add_argument("--output_dir", required=True, type=Path)
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()
    if args.threads < 1:
        p.error("threads must be positive")
    path = args.benchmark_dir / "benchmark.json"
    benchmark = json.loads(path.read_text())
    if benchmark.get("protocol") != PROTOCOL or benchmark["rdkit"] != rdBase.rdkitVersion:
        raise ValueError("Requires campaign-pilot protocol and matching RDKit")
    parent_request = json.loads((args.parent_results_dir / "request.json").read_text())
    if (parent_request["benchmark_sha256"] != benchmark["parent_benchmark_sha256"]
            or parent_request["lomap_version"] != LOMAP_SCORING_VERSION):
        raise ValueError("Original results belong to a different parent benchmark/scorer")
    descriptor = Path(__file__).with_name("descriptor_rbfe.py")
    if file_hash(descriptor) != parent_request["descriptor_sha256"]:
        raise ValueError("Descriptor code differs from the original feature cache")
    feature_path = args.parent_results_dir / "features.npz"
    if file_hash(feature_path) != json.loads((args.parent_results_dir / "features.json").read_text())["sha256"]:
        raise ValueError("Original feature cache checksum differs")
    with np.load(feature_path, allow_pickle=False) as cache:
        feature_map = {tuple(pair): values for pair, values in zip(cache["pairs"], cache["features"])}
    original = pd.read_csv(args.parent_results_dir / "edge_predictions.csv",
                           keep_default_na=False, float_precision="round_trip")
    if not original.edge_id.is_unique:
        raise ValueError("Duplicate original edge predictions")
    original = original.set_index("edge_id")
    request = {
        "benchmark_sha256": file_hash(path), "protocol": PROTOCOL,
        "parent_benchmark_sha256": benchmark["parent_benchmark_sha256"],
        "parent_request_sha256": file_hash(args.parent_results_dir / "request.json"),
        "parent_features_sha256": file_hash(feature_path),
        "parent_predictions_sha256": file_hash(args.parent_results_dir / "edge_predictions.csv"),
        "lomap_version": LOMAP_SCORING_VERSION,
        "models": "same three candidates as original benchmark; train log(target); select validation Spearman",
        "pilot_only_input": "log(generation-0 proxy), one feature",
        "code": {name: file_hash(Path(__file__).with_name(name)) for name in (
            "run_fah_campaign_pilot.py", "prepare_fah_campaign_pilot.py", "run_fah_benchmark.py", "descriptor_rbfe.py")},
        "packages": {n: version(n) for n in ("numpy", "pandas", "scipy", "scikit-learn", "rdkit")},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    request_path = args.output_dir / "request.json"
    if request_path.exists():
        if json.loads(request_path.read_text()) != request:
            raise ValueError("Run identity changed; use a fresh output directory")
    elif any(args.output_dir.iterdir()):
        raise ValueError("Unmanaged output directory")
    else:
        atomic_json(request_path, request)
    if (args.output_dir / "results.json").exists():
        print("Completed run exists")
        return
    results = {"setting": benchmark["setting"], "protocol": PROTOCOL, "unsupported_tasks": {}}
    for task in ("pilot_complex", "pilot_solvent"):
        status = benchmark["report"]["task_status"][task]
        if not status["runnable"]:
            results["unsupported_tasks"][task] = status["reasons"]
            continue
        table = args.benchmark_dir / (task + ".csv")
        if file_hash(table) != benchmark["tables"][table.name]:
            raise ValueError("Pilot table checksum differs")
        frame = pd.read_csv(table, keep_default_na=False, float_precision="round_trip")
        verify_parent_subset(frame, args.parent_benchmark_dir, benchmark["parent_benchmark_sha256"])
        for name in ("target", "pilot_proxy"):
            if not np.isfinite(frame[name]).all() or not frame[name].gt(0).all():
                raise ValueError(f"Invalid {name}")
        for split in ("train", "val", "test"):
            part = frame[frame.split.eq(split)]
            if len(part) < 100 or part.variable_group.nunique() < 5:
                raise ValueError(f"Insufficient {split} coverage")
        x = np.array([feature_map[tuple(sorted((a, b)))] for a, b in zip(frame.smi_a, frame.smi_b)])
        if x.shape != (len(frame), 25) or not np.isfinite(x).all():
            raise ValueError("Invalid descriptor features")
        test = frame.split.eq("test")
        prediction = frame[test].copy()
        selected = {}
        with threadpool_limits(limits=args.threads):
            for mode in ("structure", "pilot_only", "structure_plus_pilot"):
                inputs = inputs_for(mode, x, frame)
                (name, model), candidates = choose_model(inputs, frame)
                prediction[mode] = np.exp(model.predict(inputs[test]))
                selected[mode] = {"selected": name, "candidates": candidates}
                temporary = args.output_dir / f"{task}_{mode}.partial"
                joblib.dump(model, temporary)
                temporary.replace(args.output_dir / f"{task}_{mode}.joblib")
        baseline = original.loc[prediction.edge_id]
        for column in ("smi_a", "smi_b", "source_sha256", "split"):
            if not np.array_equal(prediction[column].values, baseline[column].values):
                raise ValueError("LOMAP reference identity differs")
        prediction["lomap"] = [float(v) if v != "" else np.nan for v in baseline.lomap]
        prediction["pilot_direct"] = prediction.pilot_proxy
        prediction["training_mean"] = frame.loc[frame.split.eq("train"), "target"].mean()
        columns = ["structure", "pilot_only", "structure_plus_pilot", "pilot_direct", "training_mean", "lomap"]
        if not np.isfinite(prediction[columns[:-1]]).all().all():
            raise ValueError("Nonfinite model predictions")
        common = prediction[np.isfinite(prediction.lomap)]
        def scores(part, cols):
            return {c: metrics(part.target.to_numpy(), part[c].to_numpy()) for c in cols}
        result = {
            "selection": selected, "test_rows": len(prediction), "lomap_common_rows": len(common),
            "common_metrics": scores(common, columns),
            "all_test_metrics": scores(prediction, columns[:-1]),
            "all_test_mse": {c: float(np.mean((prediction.target - prediction[c])**2)) for c in columns[:-1]},
            "paired_bootstrap_all_test": {
                comparator: bootstrap(prediction, ["structure_plus_pilot", comparator], comparator)
                for comparator in ("pilot_only", "pilot_direct", "structure")},
            "pilot_only_vs_direct": bootstrap(prediction, ["pilot_only", "pilot_direct"], "pilot_direct"),
            "per_source_all_test": {name: {"rows": len(part), "metrics": scores(part, columns[:-1])}
                                   for name, part in prediction.groupby("series_name")},
            "test_reference_groups": prediction.reference_group.nunique(),
            "bootstrap_scope": "frozen parent groups; single held-out reference limits inference",
        }
        temporary = args.output_dir / f"{task}_predictions.partial.csv"
        prediction.to_csv(temporary, index=False)
        temporary.replace(args.output_dir / f"{task}_predictions.csv")
        results[task] = result
    atomic_json(args.output_dir / "results.json", results)
    print(f"Completed {args.output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
