"""Verify saved GNN runs and compare architectures; no fitting or model selection."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from rbfe_artifacts import file_hash
from run_fah_benchmark import bootstrap


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results_root", required=True, type=Path)
    p.add_argument("--reference_exports", required=True, type=Path,
                   help="Folder with results_known/strict.json and task_predictions_known/strict.csv")
    p.add_argument("--output_dir", required=True, type=Path)
    args = p.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("Use a fresh review output directory")
    frames, audit, rows = {}, {}, []
    expected = {
        (setting, architecture, task, mode)
        for setting in ("known_reference", "strict_network")
        for architecture in ("hypersphere", "hybrid")
        for task in ("edge", "pilot_complex", "pilot_solvent")
        for mode in ("structure", "structure_plus_pilot")
        if not (task == "edge" and mode != "structure")
        and not (setting == "strict_network" and task != "edge")
    }
    code_requests = []
    for path in sorted(args.results_root.rglob("results.json")):
        setting, architecture, task, mode, _ = path.relative_to(args.results_root).parts
        key = (setting, architecture, task, mode)
        if key not in expected or key in frames:
            raise ValueError(f"Unexpected or repeated run {key}")
        result = json.loads(path.read_text())
        request_path = path.with_name("request.json")
        request = json.loads(request_path.read_text())
        identity = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
        assert identity == result["request_sha256"]
        assert (request["setting"], request["architecture"], request["task"], request["mode"]) == key
        for field, value in zip(("setting", "architecture", "mode"), (setting, architecture, mode)):
            assert result[field] == value
        history = json.loads(path.with_name("history.json").read_text())
        finite = [entry for entry in history if entry["validation_spearman"] is not None]
        best = max(finite, key=lambda x: x["validation_spearman"])
        assert best["epoch"] == result["best_epoch"]
        assert best["validation_spearman"] == result["best_validation_spearman"]
        checkpoint = path.with_name("checkpoint.pt")
        checkpoint_verified = checkpoint.exists()
        if checkpoint_verified:
            assert file_hash(checkpoint) == result["checkpoint_sha256"]
        suffix = "known" if setting == "known_reference" else "strict"
        ref_path = args.reference_exports / f"{task}_predictions_{suffix}.csv"
        ref_results = args.reference_exports / f"results_{suffix}.json"
        assert file_hash(ref_path) == request["reference_predictions_sha256"]
        assert file_hash(ref_results) == request["reference_results_sha256"]
        reference = pd.read_csv(ref_path, keep_default_na=False, float_precision="round_trip")
        prediction_path = path.with_name("predictions.csv")
        frame = pd.read_csv(prediction_path, keep_default_na=False, float_precision="round_trip")
        assert frame.edge_id.is_unique and frame.split.eq("test").all()
        assert set(frame.edge_id) == set(reference.edge_id)
        reference = reference.set_index("edge_id").loc[frame.edge_id]
        for field in ("smi_a", "smi_b", "source_sha256", "resample_group_a", "resample_group_b"):
            assert np.array_equal(frame[field].values, reference[field].values)
        assert np.allclose(frame.target, reference.target, rtol=1e-12, atol=1e-12)
        for field in result["common_metrics"]:
            # Numeric dtypes already preserve round-trip values. Only convert blanks.
            if frame[field].dtype == object:
                frame[field] = frame[field].map(lambda x: float(x) if x != "" else np.nan)
        for saved, original in [("tree_structure", "structure"), ("lomap", "lomap"),
                                ("tree_plus_pilot", "structure_plus_pilot"), ("pilot_direct", "pilot_direct")]:
            if saved not in frame:
                continue
            target = reference[original].map(lambda x: float(x) if x != "" else np.nan)
            assert np.allclose(frame[saved], target, rtol=1e-12, atol=1e-12, equal_nan=True)
        common = frame[np.isfinite(frame[list(result["common_metrics"])]).all(axis=1)]
        assert len(common) == result["common_rows"]
        assert len(frame) == result["test_rows"]
        assert int(frame.graph_valid.sum()) == result["graph_valid_rows"]
        for method, reported in result["common_metrics"].items():
            if method == "training_mean":
                continue  # Old cluster helper could return roundoff correlations for constants.
            computed = {"pearson": pearsonr(common.target, common[method]).statistic,
                        "spearman": spearmanr(common.target, common[method]).statistic}
            for metric, value in computed.items():
                assert abs(value - reported[metric]) < 1e-10, (key, method, metric, value, reported[metric])
            if reported["mse"] is not None:
                assert np.isclose(np.mean((common.target - common[method])**2), reported["mse"], rtol=1e-12)
        frame = frame.set_index("edge_id")
        frames[key] = frame
        method_metrics = result["common_metrics"]["gnn"]
        rows.append(dict(setting=setting, architecture=architecture, task=task, mode=mode,
                         n=result["common_rows"], validation_spearman=result["best_validation_spearman"],
                         best_epoch=result["best_epoch"], **method_metrics))
        audit["/".join(key)] = {
            "request_sha256": file_hash(request_path), "results_sha256": file_hash(path),
            "predictions_sha256": file_hash(prediction_path),
            "checkpoint_checksum_verified": checkpoint_verified,
            "reference_files_verified": True, "metrics_verified": True,
            "config": request["config"], "code": request["code"],
            "paired_bootstrap_vs_existing_baselines": result["paired_bootstrap"],
        }
        code_requests.append(request["code"])
    assert set(frames) == expected
    assert all(code == code_requests[0] for code in code_requests)
    comparisons = {}
    for setting, _, task, mode in sorted(k for k in expected if k[1] == "hypersphere"):
        a = frames[setting, "hypersphere", task, mode]
        b = frames[setting, "hybrid", task, mode].loc[a.index]
        for field in ("target", "resample_group_a", "resample_group_b"):
            assert np.array_equal(a[field], b[field])
        matched = a.copy()
        matched["hypersphere"] = a.gnn
        matched["hybrid"] = b.gnn
        matched = matched[np.isfinite(matched[["hypersphere", "hybrid", "lomap"]]).all(axis=1)]
        comparisons["/".join((setting, task, mode))] = {
            "n": len(matched),
            "hypersphere_minus_hybrid": bootstrap(matched, ["hypersphere", "hybrid"], "hybrid"),
        }
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "review.json").write_text(json.dumps({
        "review_script_sha256": file_hash(Path(__file__)),
        "runs": audit, "architecture_comparisons": comparisons,
    }, indent=2) + "\n")
    pd.DataFrame(rows).to_csv(args.output_dir / "summary.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print(json.dumps(comparisons, indent=2))


if __name__ == "__main__":
    main()
