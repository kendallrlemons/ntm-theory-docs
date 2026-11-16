"""Derive generation-0/1 pilot tasks from frozen campaign assignments."""
import argparse
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd
from rdkit import rdBase

from rbfe_artifacts import file_hash

PROTOCOL = "campaign_pilot_gen0_to_gen1_v1"
IDENTITY = ["source_sha256", "smi_a", "smi_b", "split", "group_a", "group_b",
            "reference_group", "variable_group", "resample_group_a", "resample_group_b"]


def read_parent(directory, expected_hash=None):
    path = directory / "benchmark.json"
    if expected_hash is not None and file_hash(path) != expected_hash:
        raise ValueError("Parent benchmark hash differs")
    manifest = json.loads(path.read_text())
    if manifest["version"] != 2 or manifest["rdkit"] != rdBase.rdkitVersion:
        raise ValueError("Requires original schema 2 and matching RDKit")
    if manifest.get("protocol") == PROTOCOL:
        raise ValueError("Parent must be the original campaign benchmark")
    table = directory / "edge.csv"
    if file_hash(table) != manifest["tables"]["edge.csv"]:
        raise ValueError("Parent edge table checksum mismatch")
    frame = pd.read_csv(table, keep_default_na=False, float_precision="round_trip")
    if not frame.edge_id.is_unique or not frame.split.isin(["train", "val", "test"]).all():
        raise ValueError("Invalid parent identities/splits")
    return manifest, frame


def verify_parent_subset(frame, directory, expected_hash):
    _, parent = read_parent(directory, expected_hash)
    if not frame.edge_id.is_unique or not set(frame.edge_id).issubset(set(parent.edge_id)):
        raise ValueError("Pilot task is not a subset of frozen parent edges")
    parent = parent.set_index("edge_id").loc[frame.edge_id]
    for column in IDENTITY:
        if not np.array_equal(frame[column].values, parent[column].values):
            raise ValueError(f"Frozen identity changed: {column}")
    return expected_hash


def make_task(parent, generations, phase, setting):
    g = generations[generations.phase.eq(phase) & generations.edge_id.isin(parent.edge_id)].copy()
    if g.duplicated(["edge_id", "generation"]).any():
        raise ValueError("Repeated generation identifiers")
    # Verify source identity before joining by source-scoped edge ID.
    source = parent.set_index("edge_id").source_sha256
    if not np.array_equal(g.source_sha256.values, g.edge_id.map(source).values):
        raise ValueError("Generation source identity differs from parent")
    fields = ["delta_f_stderr", "num_work_pairs", "raw_work_pair_count"]
    early = g[g.generation.eq(0)].set_index("edge_id")[fields].add_prefix("early_")
    late = g[g.generation.eq(1)].set_index("edge_id")[fields].add_prefix("late_")
    frame = parent.drop(columns="target").set_index("edge_id").join(early, how="inner").join(late, how="inner")
    valid = np.ones(len(frame), dtype=bool)
    for column in ("early_delta_f_stderr", "late_delta_f_stderr",
                   "early_num_work_pairs", "late_num_work_pairs"):
        valid &= np.isfinite(frame[column]) & frame[column].gt(0)
    invalid = int((~valid).sum())
    frame = frame[valid].copy()
    frame["pilot_proxy"] = frame.early_delta_f_stderr * np.sqrt(frame.early_num_work_pairs)
    frame["target"] = frame.late_delta_f_stderr * np.sqrt(frame.late_num_work_pairs)
    if not np.isfinite(frame[["pilot_proxy", "target"]]).all().all():
        raise ValueError("Nonfinite generation proxy")
    removed = 0
    if setting == "known_reference":
        seen = set(frame.loc[frame.split.eq("train"), "reference_smiles"])
        keep = frame.split.eq("train") | frame.reference_smiles.isin(seen)
        removed = int((~keep).sum())
        frame = frame[keep].copy()
    coverage = {}
    for split in ("train", "val", "test"):
        part = frame[frame.split.eq(split)]
        coverage[split] = {"rows": len(part), "variable_groups": part.variable_group.nunique(),
                           "reference_groups": part.reference_group.nunique(),
                           "sources": part.source_sha256.nunique()}
    reasons = [f"{split}: fewer than 100 rows or 5 variable groups"
               for split, stats in coverage.items()
               if stats["rows"] < 100 or stats["variable_groups"] < 5]
    return frame.reset_index(), {
        "coverage": coverage, "invalid_gen01_values": invalid,
        "missing_gen0_or_gen1": len(parent) - len(set(early.index) & set(late.index)),
        "removed_unseen_reference_rows": removed,
        "runnable": not reasons, "reasons": reasons,
        "later_generation_edge_counts": {
            str(gen): int(g[g.generation.eq(gen)].edge_id.nunique())
            for gen in sorted(g.generation.unique()) if gen > 1},
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent_benchmark_dir", required=True, type=Path)
    p.add_argument("--generations", required=True, type=Path)
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--output_dir", required=True, type=Path)
    args = p.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("Use a fresh pilot benchmark directory")
    parent_manifest, parent = read_parent(args.parent_benchmark_dir)
    source = json.loads(args.manifest.read_text())
    if file_hash(args.manifest) != parent_manifest["source_manifest_sha256"]:
        raise ValueError("Source extraction manifest differs from frozen parent")
    if file_hash(args.generations) != source["tables"]["generations.csv"]:
        raise ValueError("Generation table checksum differs")
    generations = pd.read_csv(args.generations, keep_default_na=False, float_precision="round_trip")
    for field in ("generation", "delta_f_stderr", "num_work_pairs", "raw_work_pair_count"):
        generations[field] = pd.to_numeric(generations[field], errors="raise")
    tasks, report = {}, {"tasks": {}, "task_status": {}}
    for task, phase in (("pilot_complex", "complex_phase"), ("pilot_solvent", "solvent_phase")):
        tasks[task], info = make_task(parent, generations, phase, parent_manifest["setting"])
        report["tasks"][task] = info
        report["task_status"][task] = {key: info[key] for key in ("runnable", "reasons")}
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(dir=args.output_dir.parent, prefix=".campaign-pilot-"))
    try:
        for task, frame in tasks.items():
            frame.to_csv(staging / (task + ".csv"), index=False)
        manifest = {
            "version": 2, "protocol": PROTOCOL, "setting": parent_manifest["setting"],
            "rdkit": rdBase.rdkitVersion,
            "parent_benchmark_sha256": file_hash(args.parent_benchmark_dir / "benchmark.json"),
            "source_manifest_sha256": file_hash(args.manifest),
            "generations_sha256": file_hash(args.generations),
            "script_sha256": file_hash(Path(__file__)),
            "pilot_information": "generation 0 only; log(proxy) is the only pilot model input",
            "pilot_target": "generation 1 stderr * sqrt(filtered work pairs), per phase",
            "eligibility": "generations 0 and 1 valid; later exported generations allowed",
            "split": "unchanged retained parent edge assignments; no resplitting",
            "report": report, "tables": {f.name: file_hash(f) for f in staging.glob("*.csv")},
        }
        (staging / "benchmark.json").write_text(json.dumps(manifest, indent=2) + "\n")
        (staging / "coverage.json").write_text(json.dumps(report, indent=2) + "\n")
        staging.rename(args.output_dir)
    except BaseException:
        shutil.rmtree(staging)
        raise
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
