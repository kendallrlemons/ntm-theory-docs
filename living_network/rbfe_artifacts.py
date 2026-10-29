"""Identity and provenance checks shared by the RBFE pipelines.

Pair keys preserve endpoint order and exact SMILES: the hybrid graph encodes
endpoint roles. No chemical canonicalization or change to the target is made.
Legacy artifacts without provenance are intentionally rejected.
"""

import hashlib
import json
import math
import os


VERSION = 1
GRAPH_VERSION = "hybrid-mcs-v1"
A = "Compound Smiles 1"
B = "Compound Smiles 2"
Y = "Difference in first_pass_free_energy_stderr"


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash_rows(rows):
    digest = hashlib.sha256()
    for row in rows:
        digest.update(json.dumps(row, ensure_ascii=True, allow_nan=False,
                                 separators=(",", ":")).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def pair_keys(df):
    columns = [A, B] if A in df.columns else ["smi_a", "smi_b"]
    keys = list(df[columns].itertuples(index=False, name=None))
    if any(not isinstance(a, str) or not isinstance(b, str) for a, b in keys):
        raise ValueError("Pair identities must contain two SMILES strings.")
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate ordered molecular pairs in artifact.")
    return keys


def benchmark_identity(df):
    """Fingerprint actual ordered pairs, signed labels and split assignments."""
    label = Y if Y in df.columns else "SE"
    keys = pair_keys(df)
    rows = []
    for (a, b), y, magnitude, split in zip(
            keys, df[label], df["SE_abs"], df["split"]):
        y = float(y)
        if not math.isfinite(y) or float(magnitude) != abs(y):
            raise ValueError("Nonfinite or inconsistent SE label.")
        if split not in ("train", "val", "test"):
            raise ValueError("Unknown split assignment.")
        rows.append((a, b, y, split))
    counts = {s: sum(row[3] == s for row in rows)
              for s in ("train", "val", "test")}
    if not all(counts.values()):
        raise ValueError(f"Benchmark requires nonempty train/val/test: {counts}")
    return {"version": VERSION, "rows_sha256": _hash_rows(rows), "counts": counts}


def require_benchmark(checkpoint, expected):
    if expected is None or checkpoint.get("benchmark") != expected:
        raise ValueError(
            "Checkpoint benchmark missing or mismatched. Retrain in a fresh "
            "output directory using the verified split and graph cache.")


def data_request(input_path, **settings):
    return {"version": VERSION, "input_sha256": file_hash(input_path),
            "settings": settings}


def write_cache_manifest(path, request):
    payload = {"request": request, "cache_sha256": file_hash(path)}
    manifest = str(path) + ".manifest.json"
    temporary = manifest + f".{os.getpid()}.tmp"
    with open(temporary, "w") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
    os.replace(temporary, manifest)


def require_cache_manifest(path, request):
    manifest = str(path) + ".manifest.json"
    if not os.path.exists(manifest):
        raise ValueError(f"Unverified legacy cache: {path}. Use a fresh output directory.")
    with open(manifest) as stream:
        payload = json.load(stream)
    if payload.get("request") != request or payload.get("cache_sha256") != file_hash(path):
        raise ValueError(f"Cache input/settings/content mismatch: {path}. Use a fresh output directory.")


def graph_bundle(graphs, keys, mcs_timeout):
    keys = list(keys)
    payload = {"version": VERSION, "graph_version": GRAPH_VERSION,
               "mcs_timeout": mcs_timeout, "pair_keys": keys, "graphs": graphs}
    validate_graph_bundle(payload, mcs_timeout)
    return payload


def validate_graph_bundle(payload, mcs_timeout):
    if not isinstance(payload, dict) or payload.get("version") != VERSION:
        raise ValueError("Unverified legacy graph cache. Recompute in a fresh directory.")
    if (payload.get("graph_version") != GRAPH_VERSION
            or payload.get("mcs_timeout") != mcs_timeout):
        raise ValueError("Graph-cache featurization/MCS settings mismatch.")
    keys = payload["pair_keys"]
    if len(keys) != len(payload["graphs"]) or len(set(keys)) != len(keys):
        raise ValueError("Graph cache has duplicate identities or inconsistent lengths.")
    if any(not isinstance(key, tuple) or len(key) != 2
           or not all(isinstance(s, str) for s in key) for key in keys):
        raise ValueError("Invalid graph pair identity.")


def aligned_graphs(payload, requested_keys, mcs_timeout):
    validate_graph_bundle(payload, mcs_timeout)
    positions = {key: i for i, key in enumerate(payload["pair_keys"])}
    requested_keys = list(requested_keys)
    missing = [key for key in requested_keys if key not in positions]
    if missing:
        raise ValueError(f"Graph cache is missing {len(missing)} requested molecular pairs.")
    # Preserve failed graph entries (None) and the exact requested row order.
    return [payload["graphs"][positions[key]] for key in requested_keys]
