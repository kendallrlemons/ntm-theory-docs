"""Freeze synthetic ligand-arrival schedules; report which routing choices are observable."""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


PROTOCOL = {
    "version": "living-growth-readiness-v1",
    "setting": "known_reference",
    "arrival_seeds": [11, 23, 37],
    "batches": 5,
    "initial_network": "All training edges within one source snapshot",
    "arrival_unit": "Whole held-out variable scaffold group",
    "worlds": "Separate validation and test replays, each starting from training only",
    "candidate_rule": "Each arriving ligand may connect to any previously present ligand, including reference; no within-batch edges",
    "context": "Do not pool source snapshots, receptor variants, phases or repeated analyses",
    "availability": "Only source-observed structural pairs are marked available; no synthetic alternative-edge outcomes",
    "interpretation": "Synthetic reveal order, not chronological replay; structural pair availability is not simulation feasibility or statistical independence",
}
FIELDS = ["edge_id", "source_sha256", "series_name", "split",
          "reference_smiles", "variable_smiles", "variable_group"]


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def schedules(frame):
    # Outcomes never enter scheduling or campaign selection.
    frame = frame[FIELDS].copy()
    if not frame.edge_id.is_unique or not frame.split.isin(["train", "val", "test"]).all():
        raise ValueError("Invalid identities or splits")
    if frame.groupby("variable_group").split.nunique().max() != 1:
        raise ValueError("Scaffold group crosses splits")
    if frame.groupby("variable_smiles").split.nunique().max() != 1:
        raise ValueError("Variable structure crosses splits")
    arrivals, initial, coverage = [], [], []
    for source, part in frame.groupby("source_sha256", sort=True):
        refs = set(part.reference_smiles)
        if len(refs) != 1:
            raise ValueError("This preparation requires one reference per source")
        reference = next(iter(refs))
        observed = {frozenset((r.reference_smiles, r.variable_smiles))
                    for r in part.itertuples()}
        neighbors = {}
        for pair in observed:
            if len(pair) != 2:
                raise ValueError("Self transformation in parent")
            a, b = tuple(pair)
            neighbors.setdefault(a, set()).add(b)
            neighbors.setdefault(b, set()).add(a)
        train = part[part.split.eq("train")]
        for r in train.itertuples():
            initial.append({k: getattr(r, k) for k in FIELDS})
        for split in ("val", "test"):
            held = part[part.split.eq(split)]
            groups = set(held.variable_group)
            if train.empty or len(groups) < PROTOCOL["batches"]:
                coverage.append(dict(source_sha256=source, series_name=part.series_name.iloc[0],
                                     split=split, status="insufficient_initial_network_or_arrival_groups",
                                     initial_edges=len(train), arrival_groups=len(groups)))
                continue
            for seed in PROTOCOL["arrival_seeds"]:
                order = sorted(groups, key=lambda g: hashlib.sha256(
                    f"living-growth-v1|{seed}|{source}|{split}|{g}".encode()).hexdigest())
                stage = {g: 1 + i * PROTOCOL["batches"] // len(order) for i, g in enumerate(order)}
                known = set(train.variable_smiles) | {reference}
                for step in range(1, PROTOCOL["batches"] + 1):
                    batch = held[held.variable_group.map(stage).eq(step)]
                    new = set(batch.variable_smiles)
                    if new & known:
                        raise ValueError("Arrival contains an already present structure")
                    # Pair-level availability; repeated microstate/run records are not new choices.
                    available = sum(len(neighbors.get(v, set()) & known) for v in new)
                    possible = len(new) * len(known)
                    coverage.append(dict(
                        source_sha256=source, series_name=part.series_name.iloc[0], split=split,
                        seed=seed, step=step, status="ready",
                        initial_edges=len(train), arrival_groups=batch.variable_group.nunique(),
                        prior_nodes=len(known), new_nodes=len(new), observed_arrival_records=len(batch),
                        candidate_structural_pairs=possible, observed_candidate_pairs=available,
                        missing_candidate_pairs=possible - available,
                        connection_choice_evaluable=available > len(new)))
                    for r in batch.itertuples():
                        arrivals.append(dict(seed=seed, step=step,
                                             **{k: getattr(r, k) for k in FIELDS}))
                    known.update(new)
    return pd.DataFrame(initial, columns=FIELDS), pd.DataFrame(
        arrivals, columns=["seed", "step"] + FIELDS), pd.DataFrame(coverage)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("Use a fresh output directory")
    path = args.benchmark_dir / "benchmark.json"
    manifest = json.loads(path.read_text())
    table = args.benchmark_dir / "edge.csv"
    if manifest["version"] != 2 or manifest["setting"] != "known_reference":
        raise ValueError("Requires the original schema-v2 known-reference benchmark")
    if digest(table) != manifest["tables"]["edge.csv"]:
        raise ValueError("Parent edge checksum differs")
    frame = pd.read_csv(table, usecols=FIELDS, keep_default_na=False)
    initial, arrivals, coverage = schedules(frame)
    args.output_dir.mkdir(parents=True)
    initial.to_csv(args.output_dir / "initial_edges.csv", index=False)
    arrivals.to_csv(args.output_dir / "arrival_schedule.csv", index=False)
    coverage.to_csv(args.output_dir / "choice_coverage.csv", index=False)
    ready = coverage[coverage.status.eq("ready")]
    choices = int(ready.connection_choice_evaluable.sum()) if len(ready) else 0
    report = dict(protocol=PROTOCOL, benchmark_sha256=digest(path), script_sha256=digest(Path(__file__)),
                  scheduled_edge_records=len(arrivals), ready_batch_contexts=len(ready),
                  batch_contexts_with_any_alternative_observed=choices,
                  decision="prospective_alternative_edges_required" if choices == 0 else "review_edge_coverage_before_policy_evaluation",
                  outputs={p.name: digest(p) for p in args.output_dir.iterdir()})
    (args.output_dir / "growth_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
