"""CPU graph preparation for the frozen Fah benchmarks; no labels in caches."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import pickle
import sqlite3

import pandas as pd
from rdkit import rdBase
import torch

from rbfe_artifacts import file_hash


def graph_recipe(architecture):
    base = Path(__file__).parent
    source = "hypersphere_rbfe.py" if architecture == "hypersphere" else "hybrid_topo_rbfe.py"
    return {"version": 1, "architecture": architecture, "rdkit": rdBase.rdkitVersion,
            "torch": str(torch.__version__), "builder_sha256": file_hash(base / source),
            "mcs_timeout": 5, "orientation": "lexicographically sorted canonical endpoint SMILES"}


def graph_key(a, b):
    return json.dumps(sorted((a, b)))


def build_one(task):
    architecture, key = task
    if architecture == "hypersphere":
        from hypersphere_rbfe import smiles_to_graph
        graph = smiles_to_graph(key)
    else:
        from hybrid_topo_rbfe import build_hybrid_graph
        graph = build_hybrid_graph(*json.loads(key), mcs_timeout=5)
    return key, graph


def keys_for(frame, architecture):
    if architecture == "hypersphere":
        return set(frame.smi_a) | set(frame.smi_b)
    return {graph_key(a, b) for a, b in zip(frame.smi_a, frame.smi_b)}


def atomic_json(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benchmark_dirs", nargs="+", type=Path, required=True)
    p.add_argument("--architecture", choices=["hypersphere", "hybrid"], required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()
    if args.workers < 1:
        p.error("workers must be positive")
    keys, identities = set(), []
    for directory in args.benchmark_dirs:
        path = directory / "benchmark.json"
        manifest = json.loads(path.read_text())
        if manifest["version"] != 2 or manifest["rdkit"] != rdBase.rdkitVersion:
            raise ValueError("Requires schema 2 and the preparation RDKit version")
        identities.append(file_hash(path))
        for name, status in manifest["report"]["task_status"].items():
            if not status["runnable"]:
                continue
            table = directory / (name + ".csv")
            if file_hash(table) != manifest["tables"][table.name]:
                raise ValueError(f"Table checksum mismatch: {table}")
            frame = pd.read_csv(table, usecols=["smi_a", "smi_b"], keep_default_na=False)
            keys.update(keys_for(frame, args.architecture))
    if not keys:
        raise ValueError("No runnable tasks")
    request = {"recipe": graph_recipe(args.architecture),
               "benchmark_hashes": sorted(set(identities)),
               "keys_sha256": __import__("hashlib").sha256(
                   json.dumps(sorted(keys)).encode()).hexdigest()}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    database = args.output_dir / "graphs.sqlite"
    manifest_path = args.output_dir / "manifest.json"
    if manifest_path.exists():
        completed = json.loads(manifest_path.read_text())
        if completed["request"] != request or completed["database_sha256"] != file_hash(database):
            raise ValueError("Completed graph cache identity or checksum differs")
        print(f"Verified completed cache: {args.output_dir}")
        return
    if not database.exists() and any(args.output_dir.iterdir()):
        raise ValueError("Unmanaged graph-cache output directory")
    connection = sqlite3.connect(database)
    try:
        connection.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT)")
        connection.execute("CREATE TABLE IF NOT EXISTS graphs (key TEXT PRIMARY KEY, data BLOB NOT NULL)")
        stored = connection.execute("SELECT value FROM metadata WHERE key='request'").fetchone()
        if stored:
            if json.loads(stored[0]) != request:
                raise ValueError("Interrupted cache uses a different recipe or benchmark")
        else:
            if connection.execute("SELECT COUNT(*) FROM graphs").fetchone()[0]:
                raise ValueError("Unverified graph entries")
            connection.execute("INSERT INTO metadata VALUES ('request', ?)", (json.dumps(request),))
            connection.commit()
        done = {row[0] for row in connection.execute("SELECT key FROM graphs")}
        if not done.issubset(keys):
            raise ValueError("Unexpected graph keys")
        pending = sorted(keys - done)
        print(f"Graphs: {len(keys)} total, {len(done)} cached, {len(pending)} pending", flush=True)
        # Spawn avoids inheriting an open SQLite connection into workers.
        import multiprocessing
        with ProcessPoolExecutor(max_workers=args.workers,
                                 mp_context=multiprocessing.get_context("spawn")) as pool:
            results = pool.map(build_one, ((args.architecture, key) for key in pending), chunksize=8)
            for index, (key, graph) in enumerate(results, 1):
                connection.execute("INSERT INTO graphs VALUES (?, ?)",
                                   (key, pickle.dumps(graph, protocol=pickle.HIGHEST_PROTOCOL)))
                if index % 64 == 0:
                    connection.commit()
                    print(f"Completed {len(done) + index}/{len(keys)}", flush=True)
        connection.commit()
        failed, no_core = [], 0
        for key, blob in connection.execute("SELECT key,data FROM graphs"):
            graph = pickle.loads(blob)
            if graph is None:
                failed.append(key)
            elif args.architecture == "hybrid" and graph["n_core"] == 0:
                no_core += 1
    finally:
        connection.close()
    atomic_json(manifest_path, {
        "request": request, "database_sha256": file_hash(database), "entries": len(keys),
        "failed_keys": sorted(failed), "hybrid_graphs_without_core": no_core})
    print(f"Completed graph cache: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
