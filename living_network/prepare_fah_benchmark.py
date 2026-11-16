"""Freeze source-preserving structure and generation-0 -> generation-1 tasks.

No fitting. Run on cluster for the full export. Splits use chemistry only.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem.Scaffolds import MurckoScaffold

from rbfe_artifacts import file_hash

VERSION = 2
SEED = 42


def hash_split(group):
    value = int(hashlib.sha256(f"fah-v1:{SEED}:{group}".encode()).hexdigest()[:16], 16) / 2**64
    return "train" if value < .6 else "val" if value < .8 else "test"


def chemistry_split(frame):
    """Unite scaffold families linked by any shared source compound identity."""
    smiles = sorted(set(frame.smi_a) | set(frame.smi_b))
    scaffolds = {}
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smi}")
        scaffolds[smi] = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    parent = {s: s for s in set(scaffolds.values())}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    compound_scaffold = {}
    for endpoint, column in (("initial", "smi_a"), ("final", "smi_b")):
        for cid, smi in zip(frame[endpoint + "_compound_id"], frame[column]):
            scaffold = scaffolds[smi]
            if cid in compound_scaffold:
                a, b = find(scaffold), find(compound_scaffold[cid])
                parent[max(a, b)] = min(a, b)
            compound_scaffold[cid] = scaffold
    group = {smi: find(scaffold) for smi, scaffold in scaffolds.items()}

    frame = frame.copy()
    frame["group_a"] = frame.smi_a.map(group)
    frame["group_b"] = frame.smi_b.map(group)
    a, b = frame.group_a.map(hash_split), frame.group_b.map(hash_split)
    frame["split"] = a.where(a.eq(b), "straddling")
    return frame, {"scaffolds": len(parent), "chemistry_groups": len(set(group.values()))}


def network_split(frame, setting):
    """Identify star references, then assign using chemistry and topology only."""
    frame = frame.copy()
    reference_sides = {}
    for source, part in frame.groupby("source_sha256"):
        initial = part.initial_compound_id.nunique() == 1
        final = part.final_compound_id.nunique() == 1
        if initial == final:
            raise ValueError(f"Source {source} lacks a uniquely identifiable star-reference side")
        reference_sides[source] = "initial" if initial else "final"
    initial = frame.source_sha256.map(reference_sides).eq("initial")
    for name, a, b in [
        ("reference_id", "initial_compound_id", "final_compound_id"),
        ("reference_smiles", "smi_a", "smi_b"),
        ("reference_group", "group_a", "group_b"),
        ("variable_id", "final_compound_id", "initial_compound_id"),
        ("variable_smiles", "smi_b", "smi_a"),
        ("variable_group", "group_b", "group_a"),
    ]:
        frame[name] = frame[a].where(initial, frame[b])
    references = set(frame.reference_group)
    report = {"setting": setting, "reference_groups": len(references),
              "reference_ids": sorted(frame.reference_id.unique()),
              "reference_sides_by_source": reference_sides}
    frame["exclusion_reason"] = ""
    if setting == "known_reference":
        frame["split"] = frame.variable_group.map(hash_split)
        # Variable chemistry may never reuse any reference scaffold family.
        mask = frame.variable_group.isin(references)
        frame.loc[mask, "exclusion_reason"] = "variable_overlaps_reference_group"
        frame["resample_group_a"] = frame.variable_group
        frame["resample_group_b"] = frame.variable_group
    elif setting == "strict_network":
        # One deterministic ordering; no seed search or label-based balancing.
        ordered = sorted(references, key=lambda g: hashlib.sha256(
            f"fah-network-v2:{SEED}:{g}".encode()).hexdigest())
        n_hold = max(1, int(.2 * len(ordered))) if len(ordered) >= 3 else 0
        n_train = len(ordered) - 2 * n_hold
        assignments = {g: ("train" if i < n_train else
                           "val" if i < n_train + n_hold else "test")
                       for i, g in enumerate(ordered)}
        report["reference_group_assignments"] = assignments
        frame["split"] = frame.reference_group.map(assignments)
        # Reference families have fixed assignments even if they also occur as targets.
        other_reference = frame.variable_group.isin(references)
        mismatch = other_reference & frame.variable_group.map(assignments).ne(frame.split)
        frame.loc[mismatch, "exclusion_reason"] = "variable_is_reference_in_other_split"
        eligible = frame[frame.exclusion_reason.eq("") & ~other_reference]
        counts = eligible.groupby("variable_group").split.nunique()
        shared = set(counts[counts.gt(1)].index)
        frame.loc[frame.variable_group.isin(shared), "exclusion_reason"] = "variable_group_spans_network_splits"
        frame["resample_group_a"] = frame.group_a
        frame["resample_group_b"] = frame.group_b
    else:
        raise ValueError(f"Unknown benchmark setting: {setting}")
    frame.loc[frame.exclusion_reason.ne(""), "split"] = "excluded"
    report["exclusions"] = frame.loc[frame.exclusion_reason.ne(""), "exclusion_reason"].value_counts().to_dict()
    return frame, report


def assert_disjoint(frame, columns):
    sets = {s: set(frame.loc[frame.split.eq(s), columns].values.ravel())
            for s in ("train", "val", "test")}
    assert not (sets["train"] & sets["val"] or sets["train"] & sets["test"]
                or sets["val"] & sets["test"])


def prepare(edges, gens, setting="known_reference"):
    assert edges.edge_id.is_unique
    assert gens.edge_id.isin(edges.edge_id).all()
    e = edges.copy()
    for endpoint, name in (("initial", "smi_a"), ("final", "smi_b")):
        def canonical(smi):
            if not isinstance(smi, str):
                return None
            mol = Chem.MolFromSmiles(smi)
            return Chem.MolToSmiles(mol, isomericSmiles=True) if mol else None
        mapping = {s: canonical(s) for s in e[endpoint + "_microstate_smiles"].dropna().unique()}
        e[name] = e[endpoint + "_microstate_smiles"].map(mapping)
    valid = e.initial_resolved.eq(True) & e.final_resolved.eq(True) & e.smi_a.notna() & e.smi_b.notna()
    report = {"input_edges": len(e), "unresolved_or_invalid": int((~valid).sum())}
    e = e[valid].copy()
    report["self_structure_records"] = int(e.smi_a.eq(e.smi_b).sum())
    e = e[e.smi_a.ne(e.smi_b)].copy()
    # Assign before target validity filtering or pilot eligibility checks.
    e, split_info = chemistry_split(e)
    report.update(split_info)
    e, network_info = network_split(e, setting)
    report.update(network_info)
    report["split_counts_before_target_filter"] = e.split.value_counts().to_dict()
    e["pair_key"] = [json.dumps(sorted(pair)) for pair in zip(e.smi_a, e.smi_b)]
    kept = e[e.split.ne("excluded")].copy()
    # Explicit no-overlap checks across all retained records, globally.
    columns = ([["group_a", "group_b"], ["smi_a", "smi_b"],
                ["initial_compound_id", "final_compound_id"]]
               if setting == "strict_network" else
               [["variable_group"], ["variable_smiles"], ["variable_id"]])
    for column_pair in columns:
        assert_disjoint(kept, column_pair)
    cols = ["edge_id", "source_sha256", "series_name", "run_id", "pair_key",
            "smi_a", "smi_b", "group_a", "group_b", "split", "reliable_transformation",
            "reference_id", "reference_smiles", "reference_group", "variable_id",
            "variable_smiles", "variable_group", "resample_group_a", "resample_group_b"]
    tasks = {}
    structure = kept[cols].copy()
    structure["target"] = kept.binding_free_energy_stderr
    structure = structure[np.isfinite(structure.target) & structure.target.gt(0)]
    tasks["edge"] = structure
    for phase, task in (("complex_phase", "pilot_complex"), ("solvent_phase", "pilot_solvent")):
        g = gens[gens.phase.eq(phase)].copy()
        if g.duplicated(["edge_id", "generation"]).any():
            raise ValueError("Repeated generation IDs within an edge/phase")
        # Require exactly two exported generations, labeled 0 and 1.
        eligible = g.groupby("edge_id").generation.agg(lambda x: sorted(x.tolist()) == [0, 1])
        g = g[g.edge_id.isin(eligible[eligible].index)]
        early = g[g.generation.eq(0)].set_index("edge_id")
        late = g[g.generation.eq(1)].set_index("edge_id")
        pilot = kept[cols].set_index("edge_id").join(
            early[["delta_f_stderr", "num_work_pairs", "raw_work_pair_count"]].rename(
                columns=lambda c: "early_" + c), how="inner")
        pilot = pilot.join(late[["delta_f_stderr", "num_work_pairs"]].rename(
            columns=lambda c: "late_" + c), how="inner")
        pilot["pilot_proxy"] = pilot.early_delta_f_stderr * np.sqrt(pilot.early_num_work_pairs)
        pilot["target"] = pilot.late_delta_f_stderr * np.sqrt(pilot.late_num_work_pairs)
        mask = np.isfinite(pilot.target) & pilot.target.gt(0) & np.isfinite(pilot.pilot_proxy) & pilot.pilot_proxy.gt(0)
        mask &= pilot.early_num_work_pairs.gt(0) & pilot.late_num_work_pairs.gt(0)
        tasks[task] = pilot[mask].reset_index()
    report["tasks"] = {}
    report["task_status"] = {}
    for name, frame in tasks.items():
        removed_unknown = 0
        if setting == "known_reference":
            seen = set(frame.loc[frame.split.eq("train"), "reference_smiles"])
            eligible = frame.split.eq("train") | frame.reference_smiles.isin(seen)
            removed_unknown = int((~eligible).sum())
            frame = frame[eligible].copy()
            tasks[name] = frame
        report["tasks"][name] = {
            split: {"rows": len(part), "pairs": part.pair_key.nunique(),
                    "groups": len(set(part.group_a) | set(part.group_b)),
                    "variable_groups": part.variable_group.nunique(),
                    "reference_groups": part.reference_group.nunique(),
                    "sources": part.source_sha256.nunique()}
            for split in ("train", "val", "test")
            for part in [frame[frame.split.eq(split)]]}
        reasons = [
            f"{split}: fewer than 100 rows or 5 variable chemistry groups"
            for split, stats in report["tasks"][name].items()
            if stats["rows"] < 100 or stats["variable_groups"] < 5]
        report["task_status"][name] = {
            "runnable": not reasons, "reasons": reasons,
            "removed_unseen_reference_rows": removed_unknown}
    return tasks, report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--edges", type=Path, required=True)
    p.add_argument("--generations", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--setting", choices=["both", "known_reference", "strict_network"], default="both")
    args = p.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("Use a fresh benchmark output directory")
    source_manifest = json.loads(args.manifest.read_text())
    for path in (args.edges, args.generations):
        if file_hash(path) != source_manifest["tables"][path.name]:
            raise ValueError(f"Source checksum mismatch: {path}")
    edges = pd.read_csv(args.edges, low_memory=False)
    gens = pd.read_csv(args.generations, low_memory=False)
    settings = ["known_reference", "strict_network"] if args.setting == "both" else [args.setting]
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(dir=args.output_dir.parent, prefix=".fah-benchmark-"))
    try:
        reports = {}
        for setting in settings:
            tasks, report = prepare(edges, gens, setting)
            destination = staging / setting
            destination.mkdir()
            for name, frame in tasks.items():
                frame.to_csv(destination / (name + ".csv"), index=False)
            request = {
                "version": VERSION, "setting": setting, "seed": SEED, "rdkit": rdBase.rdkitVersion,
                "source_manifest_sha256": file_hash(args.manifest),
                "script_sha256": file_hash(Path(__file__)),
                "split": ("variable compound-linked scaffold hash, shared known references"
                          if setting == "known_reference" else
                          "fixed reference-family holdout; remove cross-split variable chemistry"),
                "edge_target": "observed pooled binding stderr in source units; not intrinsic difficulty",
                "pilot_target": "gen1 stderr * sqrt(gen1 filtered work pairs), separately by phase",
                "pilot_information": "gen0 only; equal generation index does not imply fixed GPU cost",
                "reliability_filter": False,
                "report": report,
                "tables": {f.name: file_hash(f) for f in destination.glob("*.csv")},
            }
            (destination / "benchmark.json").write_text(json.dumps(request, indent=2) + "\n")
            reports[setting] = report
        (staging / "coverage.json").write_text(json.dumps(reports, indent=2) + "\n")
        staging.rename(args.output_dir)
    except BaseException:
        shutil.rmtree(staging)
        raise
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
