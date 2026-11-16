"""Extract source-preserving Fah-XChem tables; requires ijson.

No filtering, deduplication by SMILES, model fitting, or split assignment.
Uncertainties retain source units. stderr*sqrt(num_work_pairs) is only a proxy.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile

import ijson


VERSION = 2


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def items(path, prefix):
    with open(path, "rb") as handle:
        yield from ijson.items(handle, prefix, use_float=True)


def one(path, prefix):
    values = list(items(path, prefix))
    if len(values) != 1:
        raise ValueError(f"{path}: expected exactly one {prefix}")
    return values[0]


def estimate(value, prefix):
    value = value or {}
    return {prefix + "_point": value.get("point"),
            prefix + "_stderr": value.get("stderr")}


def phase_values(value):
    value = value or {}
    result = estimate(value.get("delta_f"), "delta_f")
    n = value.get("num_work_pairs")
    se = result["delta_f_stderr"]
    valid = (isinstance(n, (int, float)) and not isinstance(n, bool)
             and math.isfinite(n) and n > 0
             and isinstance(se, (int, float)) and not isinstance(se, bool)
             and math.isfinite(se) and se >= 0)
    result.update(num_work_pairs=n, bar_overlap=value.get("bar_overlap"),
                  stderr_sqrt_work_pairs_proxy=se * math.sqrt(n) if valid else None,
                  proxy_valid=valid)
    return result


class Tables:
    def __init__(self, directory):
        self.directory = directory
        self.handles = {}
        self.writers = {}
        self.counts = {}

    def write(self, name, row):
        if name not in self.writers:
            handle = open(self.directory / (name + ".csv"), "w", newline="")
            self.handles[name] = handle
            writer = csv.DictWriter(handle, fieldnames=list(row), extrasaction="raise")
            writer.writeheader()
            self.writers[name] = writer
            self.counts[name] = 0
        self.writers[name].writerow(row)
        self.counts[name] += 1

    def close(self):
        for handle in self.handles.values():
            handle.close()


def extract_source(path, tables):
    source_hash = digest(path)
    metadata = one(path, "series.metadata")
    as_of = one(path, "as_of")
    common = {"source_sha256": source_hash, "series_name": metadata["name"],
              "as_of": as_of}
    lookup = {}
    record_counts = {}
    microstate_count = 0
    compound_ids = set()
    for compound in items(path, "series.compounds.item"):
        cm = compound["metadata"]
        cid = cm["compound_id"]
        if cid in compound_ids:
            raise ValueError(f"Duplicate compound ID in {path}: {cid}")
        compound_ids.add(cid)
        tables.write("compounds", dict(
            common, compound_id=cid, compound_smiles=cm["smiles"],
            metadata_json=json.dumps(cm, sort_keys=True),
            **estimate(compound.get("free_energy"), "free_energy")))
        for microstate_index, entry in enumerate(compound["microstates"]):
            micro = entry["microstate"]
            key = (cid, micro["microstate_id"])
            lookup.setdefault(key, set()).add((cm["smiles"], micro["smiles"]))
            record_counts[key] = record_counts.get(key, 0) + 1
            microstate_count += 1
            tables.write("microstates", dict(
                common, compound_id=cid, microstate_id=key[1],
                source_microstate_index=microstate_index,
                compound_smiles=cm["smiles"], microstate_smiles=micro["smiles"],
                **estimate(entry.get("free_energy"), "free_energy"),
                **estimate(entry.get("first_pass_free_energy"), "first_pass_free_energy"),
                **estimate(micro.get("free_energy_penalty"), "free_energy_penalty")))
    edge_count = reliable_count = unresolved_count = 0
    for index, entry in enumerate(items(path, "series.transformations.item")):
        transform = entry["transformation"]
        edge_id = f"{source_hash}:{index}"
        edge = dict(common, edge_id=edge_id, source_transformation_index=index,
                    run_id=transform["run_id"],
                    xchem_fragment_id=transform.get("xchem_fragment_id"),
                    reliable_transformation=entry.get("reliable_transformation"),
                    **estimate(entry.get("binding_free_energy"), "binding_free_energy"),
                    **estimate(entry.get("exp_ddg"), "experimental_ddg"))
        for endpoint in ("initial", "final"):
            ref = transform[endpoint + "_microstate"]
            key = (ref["compound_id"], ref["microstate_id"])
            candidates = lookup.get(key, set())
            status = ("missing" if not candidates else
                      "unique_structure" if len(candidates) == 1 else
                      "ambiguous_structure")
            smiles = next(iter(candidates)) if len(candidates) == 1 else (None, None)
            # Retain a shared compound structure even when the microstate is ambiguous.
            compound_smiles = {pair[0] for pair in candidates}
            if len(compound_smiles) == 1:
                smiles = (next(iter(compound_smiles)), smiles[1])
            edge.update({
                endpoint + "_compound_id": key[0],
                endpoint + "_microstate_id": key[1],
                endpoint + "_compound_smiles": smiles[0],
                endpoint + "_microstate_smiles": smiles[1],
                endpoint + "_resolved": len(candidates) == 1,
                endpoint + "_resolution_status": status,
                endpoint + "_microstate_record_count": record_counts.get(key, 0),
                endpoint + "_structure_candidate_count": len(candidates),
            })
        resolved = edge["initial_resolved"] and edge["final_resolved"]
        unresolved_count += not resolved
        for phase_name in ("complex_phase", "solvent_phase"):
            phase = entry.get(phase_name) or {}
            for key, value in phase_values(phase.get("free_energy")).items():
                edge[phase_name + "_" + key] = value
            generations = phase.get("gens") or []
            edge[phase_name + "_generation_count"] = len(generations)
            edge[phase_name + "_raw_work_pair_count"] = sum(
                len(gen.get("works") or []) for gen in generations)
            for gen_index, gen in enumerate(generations):
                works = gen.get("works") or []
                tables.write("generations", dict(
                    common, edge_id=edge_id, run_id=transform["run_id"],
                    phase=phase_name, generation_index=gen_index,
                    generation=gen["gen"], raw_work_pair_count=len(works),
                    **phase_values(gen.get("free_energy"))))
        tables.write("edges", edge)
        edge_count += 1
        reliable_count += entry.get("reliable_transformation") is True
    if not compound_ids or not edge_count:
        raise ValueError(f"{path}: empty compounds or transformations")
    if digest(path) != source_hash:
        raise ValueError(f"Source changed during extraction: {path}")
    return dict(path=str(path.resolve()), sha256=source_hash,
                bytes=path.stat().st_size, as_of=as_of, metadata=metadata,
                compounds=len(compound_ids), microstates=microstate_count,
                unique_microstate_keys=len(lookup),
                duplicate_microstate_keys=sum(n > 1 for n in record_counts.values()),
                ambiguous_microstate_keys=sum(len(v) > 1 for v in lookup.values()),
                transformations=edge_count, reliable_transformations=reliable_count,
                unresolved_edges=unresolved_count)


def build(paths, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Use a fresh output directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=output.name + ".partial-", dir=output.parent))
    tables = Tables(staging)
    try:
        sources = [extract_source(Path(path), tables) for path in paths]
        hashes = [source["sha256"] for source in sources]
        if len(hashes) != len(set(hashes)):
            raise ValueError("Identical source content supplied more than once")
        tables.close()
        manifest = {
            "schema_version": VERSION, "sources": sources, "row_counts": tables.counts,
            "loader_sha256": digest(Path(__file__)), "ijson_version": ijson.__version__,
            "units": "Source units, unchanged; verify upstream schema before conversion.",
            "proxy_definition": "phase stderr * sqrt(phase num_work_pairs)",
            "proxy_caveat": "Not a verified intrinsic spread or effective-sample-size correction.",
            "combined_proxy": None,
            "feature_caveat": "Overlap, realized counts, reliability, and uncertainty are post-simulation quantities.",
            "raw_work_values": "Retained in source JSON; not duplicated in CSV.",
            "tables": {p.name: digest(p) for p in sorted(staging.glob("*.csv"))},
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        staging.rename(output)
        return manifest
    except BaseException:
        tables.close()
        shutil.rmtree(staging)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", type=Path,
                        help="Explicit analysis.json files")
    parser.add_argument("--input_dir", type=Path,
                        help="Recursively find analysis.json files")
    parser.add_argument("--output_dir", required=True, type=Path)
    args = parser.parse_args()
    if bool(args.input) == bool(args.input_dir):
        parser.error("Specify exactly one of --input or --input_dir")
    paths = args.input or sorted(args.input_dir.rglob("analysis.json"))
    if not paths:
        parser.error("No analysis.json files found")
    manifest = build(paths, args.output_dir)
    print(json.dumps({"output_dir": str(args.output_dir),
                      "row_counts": manifest["row_counts"]}, indent=2))


if __name__ == "__main__":
    main()
