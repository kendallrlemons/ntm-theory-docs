"""Descriptive audit of exported Fah tables; no fitting or split selection."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import rdkit
from rdkit import Chem
from scipy.stats import pearsonr, spearmanr


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def correlations(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    valid = np.isfinite(a) & np.isfinite(b)
    a, b = a[valid], b[valid]
    result = {"n": int(len(a)), "pearson": None, "spearman": None}
    if len(a) > 2 and np.ptp(a) > 0 and np.ptp(b) > 0:
        result.update(pearson=float(pearsonr(a, b).statistic),
                      spearman=float(spearmanr(a, b).statistic))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--edges", required=True, type=Path)
    parser.add_argument("--generations", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("Use a fresh audit output directory")
    manifest = json.loads(args.manifest.read_text())
    for path in (args.edges, args.generations):
        if sha256(path) != manifest["tables"][path.name]:
            raise ValueError(f"Manifest checksum mismatch: {path}")
    edges = pd.read_csv(args.edges, low_memory=False)
    gens = pd.read_csv(args.generations, low_memory=False)
    assert edges.reliable_transformation.dropna().isin([True, False]).all()
    edges["reliable_true"] = edges.reliable_transformation.eq(True)
    edges["reliability_missing"] = edges.reliable_transformation.isna()
    assert edges.edge_id.is_unique
    assert len(edges) == manifest["row_counts"]["edges"]
    assert len(gens) == manifest["row_counts"]["generations"]
    assert gens.edge_id.isin(edges.edge_id).all()
    sources = {s["sha256"]: s for s in manifest["sources"]}
    assert edges.source_sha256.isin(sources).all()
    for phase in ("complex_phase", "solvent_phase"):
        edges[phase + "_project"] = edges.source_sha256.map(
            lambda x: sources[x]["metadata"]["fah_projects"].get(phase))
    smiles = set(edges.initial_microstate_smiles.dropna()) | set(
        edges.final_microstate_smiles.dropna())
    canonical = {}
    for smi in sorted(smiles):
        mol = Chem.MolFromSmiles(smi)
        canonical[smi] = Chem.MolToSmiles(mol, isomericSmiles=True) if mol else None
    for endpoint in ("initial", "final"):
        edges[endpoint + "_canonical"] = edges[endpoint + "_microstate_smiles"].map(canonical)
    edges["structure_valid"] = (
        edges.initial_resolved.eq(True) & edges.final_resolved.eq(True)
        & edges.initial_canonical.notna() & edges.final_canonical.notna())
    se = edges.binding_free_energy_stderr
    edges["edge_stderr_valid"] = np.isfinite(se) & se.ge(0)
    edges["usable"] = edges.structure_valid & edges.edge_stderr_valid
    edges["pair_key"] = [
        json.dumps(sorted([a, b])) if valid else None
        for a, b, valid in zip(edges.initial_canonical, edges.final_canonical,
                               edges.structure_valid)]
    report = {
        "audit_version": 1, "rdkit_version": rdkit.__version__,
        "input_manifest_sha256": sha256(args.manifest),
        "script_sha256": sha256(Path(__file__)),
        "records": len(edges), "sources": len(sources),
        "invalid_distinct_smiles": sum(v is None for v in canonical.values()),
        "structure_valid": int(edges.structure_valid.sum()),
        "invalid_edge_stderr": int((~edges.edge_stderr_valid).sum()),
        "zero_edge_stderr": int(se.eq(0).sum()),
        "usable_records": int(edges.usable.sum()),
        "distinct_unordered_structure_pairs": edges.loc[edges.usable, "pair_key"].nunique(),
        "self_structure_records": int((edges.structure_valid &
            edges.initial_canonical.eq(edges.final_canonical)).sum()),
        "reliable_true": int(edges.reliable_true.sum()),
        "reliability_missing": int(edges.reliability_missing.sum()),
    }
    # Project/run identity is a candidate linkage, not proof of shared work samples.
    keys = ["complex_phase_project", "solvent_phase_project", "run_id"]
    run_groups = edges.groupby(keys, dropna=False)
    run_summary = run_groups.agg(
        records=("edge_id", "size"), sources=("source_sha256", "nunique"),
        resolved_structure_pairs=("pair_key", "nunique"))
    report["project_run_groups"] = len(run_summary)
    report["project_run_groups_multiple_sources"] = int(run_summary.sources.gt(1).sum())
    report["project_run_groups_conflicting_resolved_pairs"] = int(
        run_summary.resolved_structure_pairs.gt(1).sum())
    # Exact equality of exported estimates/counts is evidence of duplicate summaries,
    # not confirmation that all underlying work values are identical.
    numeric = [phase + "_" + suffix
               for phase in ("complex_phase", "solvent_phase")
               for suffix in ("delta_f_point", "delta_f_stderr", "num_work_pairs",
                              "bar_overlap")]
    exact_cols = keys + ["pair_key"] + numeric
    usable = edges.loc[edges.usable].copy()
    report["usable_unique_project_run_pair_and_phase_summaries"] = len(
        usable.drop_duplicates(exact_cols))
    report["usable_redundant_exact_summary_records"] = int(
        usable.duplicated(exact_cols).sum())
    report["usable_redundant_pair_and_phase_summaries_ignoring_projects"] = int(
        usable.duplicated(["pair_key"] + numeric).sum())
    pair_sources = usable.groupby("pair_key").source_sha256.nunique()
    report["structure_pairs_in_multiple_sources"] = int(pair_sources.gt(1).sum())
    chemical_overlap = []
    for i, left_id in enumerate(sources):
        left_pairs = set(usable.loc[usable.source_sha256.eq(left_id), "pair_key"])
        for right_id in list(sources)[i + 1:]:
            right_pairs = set(usable.loc[usable.source_sha256.eq(right_id), "pair_key"])
            shared = left_pairs & right_pairs
            if shared:
                chemical_overlap.append({
                    "left": sources[left_id]["metadata"]["name"],
                    "right": sources[right_id]["metadata"]["name"],
                    "shared_unordered_structure_pairs": len(shared),
                })
    # Compare dashboards only on project/run keys unique within each source.
    comparisons = []
    source_ids = list(sources)
    for i, left_id in enumerate(source_ids):
        left = edges[edges.source_sha256.eq(left_id)]
        for right_id in source_ids[i + 1:]:
            right = edges[edges.source_sha256.eq(right_id)]
            if set(map(tuple, left[keys[:2]].values)) != set(map(tuple, right[keys[:2]].values)):
                continue
            l = left.loc[~left.duplicated(keys, keep=False)]
            r = right.loc[~right.duplicated(keys, keep=False)]
            joined = l.merge(r, on=keys, suffixes=("_a", "_b"))
            if joined.empty:
                continue
            same = joined.pair_key_a.notna() & joined.pair_key_a.eq(joined.pair_key_b)
            equal = same.copy()
            for col in numeric:
                equal &= joined[col + "_a"].eq(joined[col + "_b"])
            comparisons.append({
                "left": sources[left_id]["metadata"]["name"],
                "right": sources[right_id]["metadata"]["name"],
                "matched_project_runs": len(joined),
                "same_resolved_pair": int(same.sum()),
                "identical_phase_summaries_and_pair": int(equal.sum()),
                "stderr_spearman_same_pair": correlations(
                    joined.loc[same, "binding_free_energy_stderr_a"],
                    joined.loc[same, "binding_free_energy_stderr_b"])["spearman"],
            })
    report["phase_checks"] = {}
    pilot_rows = []
    count_rows = []
    generation_checks = {}
    for phase in ("complex_phase", "solvent_phase"):
        n = edges[phase + "_num_work_pairs"]
        s = edges[phase + "_delta_f_stderr"]
        proxy = edges[phase + "_stderr_sqrt_work_pairs_proxy"]
        valid = np.isfinite(n) & n.gt(0) & np.isfinite(s) & s.ge(0)
        assert np.allclose(proxy[valid], (s * np.sqrt(n))[valid])
        report["phase_checks"][phase] = {
            "valid_proxy_records": int(valid.sum()),
            "n_quantiles": n.quantile([0, .25, .5, .75, 1]).to_dict(),
        }
        g = gens[gens.phase.eq(phase)].copy()
        duplicate = g.duplicated(["edge_id", "generation"], keep=False)
        generation_checks[phase] = {
            "duplicate_edge_generation_rows": int(duplicate.sum()),
            "generation_count_distribution": {
                str(k): int(v) for k, v in g.groupby("edge_id").size().value_counts().items()},
        }
        counts = g.groupby("edge_id").agg(raw=("raw_work_pair_count", "sum"),
                                        declared=("num_work_pairs", "sum"))
        matched = edges.set_index("edge_id").join(counts)
        generation_checks[phase]["raw_counts_match_exported_phase"] = int(
            matched.raw.eq(matched[phase + "_raw_work_pair_count"]).sum())
        generation_checks[phase]["generation_counts_match_phase_n"] = int(
            matched.declared.eq(matched[phase + "_num_work_pairs"]).sum())
        for source, part in matched.groupby("source_sha256"):
            count_rows.append({
                "series_name": sources[source]["metadata"]["name"], "phase": phase,
                "edges": len(part),
                "summed_generation_n_differs_from_phase_n": int(
                    part.declared.ne(part[phase + "_num_work_pairs"]).sum()),
            })
        # Exactly two distinct generations; compare their separate estimates.
        # Do not compare a pilot with a pooled estimate that contains the pilot.
        sizes = g.groupby("edge_id").size()
        eligible = sizes[sizes.eq(2)].index.difference(g.loc[duplicate, "edge_id"])
        ordered = g[g.edge_id.isin(eligible)].sort_values(["edge_id", "generation"])
        first = ordered.groupby("edge_id").head(1).set_index("edge_id")
        last = ordered.groupby("edge_id").tail(1).set_index("edge_id")
        paired = first[["delta_f_stderr", "stderr_sqrt_work_pairs_proxy",
                        "generation", "num_work_pairs"]].join(
            last[["delta_f_stderr", "stderr_sqrt_work_pairs_proxy",
                  "generation", "num_work_pairs"]], lsuffix="_early", rsuffix="_late")
        paired = paired.join(edges.set_index("edge_id")[["source_sha256", "usable"]])
        paired = paired[paired.usable]
        for source, part in paired.groupby("source_sha256"):
            for field in ("delta_f_stderr", "stderr_sqrt_work_pairs_proxy"):
                pilot_rows.append(dict(
                    source_sha256=source, series_name=sources[source]["metadata"]["name"],
                    phase=phase, quantity=field,
                    **correlations(part[field + "_early"], part[field + "_late"])))
    report["generation_checks"] = generation_checks
    quadrature = np.sqrt(edges.complex_phase_delta_f_stderr ** 2
                         + edges.solvent_phase_delta_f_stderr ** 2)
    report["edge_stderr_matches_phase_quadrature"] = int(
        np.isclose(se, quadrature, rtol=1e-8, atol=1e-10).sum())
    coverage = edges.groupby("series_name").agg(
        records=("edge_id", "size"), usable=("usable", "sum"),
        unique_pairs=("pair_key", "nunique"),
        reliable=("reliable_true", "sum"),
        reliability_missing=("reliability_missing", "sum"))
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    coverage.to_csv(args.output_dir / "source_coverage.csv")
    run_summary.to_csv(args.output_dir / "project_run_groups.csv")
    pd.DataFrame(comparisons).to_csv(args.output_dir / "dashboard_overlap.csv", index=False)
    pd.DataFrame(pilot_rows).to_csv(args.output_dir / "generation_correlations.csv", index=False)
    pd.DataFrame(count_rows).to_csv(args.output_dir / "generation_count_checks.csv", index=False)
    pd.DataFrame(chemical_overlap).to_csv(args.output_dir / "chemical_overlap.csv", index=False)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
