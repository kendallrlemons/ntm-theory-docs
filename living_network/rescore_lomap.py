"""Correct LOMAP scores on saved descriptor-only test predictions.

No model fitting, inference, split generation, or graph precomputation.
This retains the historical cohort; it does not turn it into a scaffold test.
"""

import argparse
from importlib.metadata import version
import json
import os

import numpy as np
import pandas as pd
from rdkit import rdBase

import descriptor_rbfe as desc
import rbfe_artifacts as artifacts


LOMAP = "LOMAP (real, 1-sim)"
REGRESSION = {"descriptors", "train_mean", "Hybrid-topo GNN"}
METHODS = ["descriptors", "train_mean", "Tanimoto (1-sim)", "Hybrid-topo GNN"]


def compare_saved(frame, corrected):
    """Keep row order and non-LOMAP predictions; recompute common eligibility."""
    artifacts.pair_keys(frame)
    if "descriptors_lomap" in frame:
        raise ValueError("Use descriptor-only predictions: a model trained on old LOMAP features needs separate treatment.")
    required = {"SE_abs", LOMAP, "descriptors", "train_mean", "Tanimoto (1-sim)", "common_eligible"}
    if not required <= set(frame.columns):
        raise ValueError(f"Missing saved prediction columns: {sorted(required-set(frame.columns))}")
    corrected = np.asarray(corrected, dtype=float)
    if corrected.shape != (len(frame),):
        raise ValueError("LOMAP score count does not match saved pairs.")
    y = frame.SE_abs.to_numpy(dtype=float)
    if not np.isfinite(y).all():
        raise ValueError("Saved test labels must be finite.")
    if not frame.common_eligible.isin([True, False]).all():
        raise ValueError("common_eligible must contain booleans.")
    previous = frame.common_eligible.to_numpy(dtype=bool)
    methods = {name: frame[name].to_numpy(dtype=float) for name in METHODS if name in frame}
    methods[LOMAP] = corrected
    valid = np.isfinite(corrected)
    for prediction in methods.values():
        valid &= np.isfinite(prediction)
    if valid.sum() < 2:
        raise ValueError("Fewer than two common eligible test predictions.")
    output = frame.copy()
    output["previous_lomap_difficulty"] = frame[LOMAP]
    output["previous_common_eligible"] = previous
    output[LOMAP] = corrected
    output["common_eligible"] = valid
    def metrics(mask):
        return desc.metric_rows(y[mask], {k: v[mask] for k, v in methods.items()}, REGRESSION)
    # Report new eligibility and also its intersection with the old cohort so
    # changes in score quality and changes in coverage can be distinguished.
    shared = valid & previous
    coverage = {
        "total": len(frame), "previous_common": int(previous.sum()),
        "corrected_lomap_finite": int(np.isfinite(corrected).sum()),
        "corrected_lomap_max_difficulty": int((corrected == 1.).sum()),
        "new_common": int(valid.sum()), "previous_and_new_common": int(shared.sum()),
        "newly_eligible": int((valid & ~previous).sum()),
        "no_longer_eligible": int((previous & ~valid).sum()),
    }
    return output, metrics(valid), metrics(shared) if shared.sum() >= 2 else [], coverage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--mcs_timeout", type=int, default=5)
    args = parser.parse_args()
    if args.workers < 1 or args.mcs_timeout < 1:
        parser.error("workers and mcs_timeout must be positive")
    output_path = os.path.join(args.output_dir, "test_predictions.csv")
    if os.path.realpath(output_path) == os.path.realpath(args.predictions):
        parser.error("Use a separate output directory to preserve original predictions.")
    import gufe
    import lomap
    if not hasattr(lomap, "LomapAtomMapper") or not hasattr(lomap, "default_lomap_score"):
        raise RuntimeError("Installed LOMAP does not expose the expected mapping/scoring API.")
    frame = pd.read_csv(args.predictions)
    # Validate before expensive scoring; this calculation is not saved.
    compare_saved(frame, np.zeros(len(frame)))
    pairs = artifacts.pair_keys(frame)
    request = {
        "source_predictions_sha256": artifacts.file_hash(args.predictions),
        "lomap_scoring_version": artifacts.LOMAP_SCORING_VERSION,
        "mapper_settings": {"time": args.mcs_timeout, "threed": False, "max3d": 0.0, "shift": False},
        "rdkit": rdBase.rdkitVersion, "gufe": version("gufe"),
        "lomap": getattr(lomap, "__version__", None),
        "lomap_module_sha256": artifacts.file_hash(lomap.__file__),
        "benchmark_scope": "Historical saved test cohort; no new split or training",
    }
    os.makedirs(args.output_dir, exist_ok=True)
    manifest = os.path.join(args.output_dir, "rescore_manifest.json")
    if os.path.exists(manifest):
        with open(manifest) as stream:
            if json.load(stream) != request:
                raise ValueError("Rescore inputs/settings differ; use a fresh output directory.")
    else:
        desc.json_dump(request, manifest)
    corrected = desc.score_lomap(pairs, np.arange(len(pairs)), args, request)
    output, full, shared, coverage = compare_saved(frame, corrected)
    output.to_csv(output_path, index=False)
    pd.DataFrame(full).to_csv(os.path.join(args.output_dir, "comparison.csv"), index=False)
    pd.DataFrame(shared).to_csv(os.path.join(args.output_dir, "comparison_previous_cohort.csv"), index=False)
    desc.json_dump(coverage, os.path.join(args.output_dir, "coverage.json"))
    print(json.dumps(coverage, indent=2))
    print(pd.DataFrame(full).to_string(index=False))
    print("Historical cohort only. This does not establish scaffold-held-out performance.")


if __name__ == "__main__":
    main()
