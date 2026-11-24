"""Recompute headline intervals and Route 1 from frozen GNN evaluation folders."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def ranks(x, w):
    mass = np.bincount(x, weights=w)
    return (mass.cumsum() - mass / 2)[x]


def corr(x, y, w):
    x, y = x - np.average(x, weights=w), y - np.average(y, weights=w)
    den = np.sqrt(w @ (x*x) * (w @ (y*y)))
    return (w @ (x*y)) / den if den else np.nan


def selection(y, w, orders):
    total = count = 0
    for idx in orders:
        weights = w[idx]
        quota = (weights.sum() + 3) // 4
        take = np.minimum(weights, np.maximum(0, quota - weights.cumsum() + weights))
        total += take @ y[idx]
        count += quota
    return total / count, int(count)


def analyze(root, setting, task):
    edge = task == "edge"
    known = setting == "known_reference"
    common = known or edge
    folder = (root / "fah_gnn_results_v1" if common else root) / setting
    directory = folder / "hypersphere" / task / "structure"

    def load(d):
        f = pd.read_csv(d / "predictions.csv").set_index("edge_id", verify_integrity=True)
        assert f.split.eq("test").all()
        return f, json.loads((d / "results.json").read_text())

    f, saved = load(directory)
    other, saved_other = load(folder / "hybrid" / task / "structure" if edge
                              else directory.with_name("structure_plus_pilot"))
    other = other.loc[f.index]
    assert f[["target", "source_sha256", "resample_group_a", "resample_group_b"]].equals(
        other[["target", "source_sha256", "resample_group_a", "resample_group_b"]])
    f["other"] = other.gnn
    cols = (["lomap", "tree_structure", "other", "gnn"] if edge else
            ["gnn", "pilot_direct", *([] if known else ["tree_pilot_only"]), "other"])
    f = f.loc[np.isfinite(f[cols + (["lomap"] if common else [])]).all(axis=1)].copy()
    assert len(f) == saved["common_rows" if common else "graph_valid_rows"]
    y = f.target.to_numpy()
    xs = {c: f[c].to_numpy() for c in cols}
    enc = lambda x: np.unique(x, return_inverse=True)[1]
    ry = enc(y)
    rx = {c: enc(x) for c, x in xs.items()}
    groups, idx = np.unique(np.r_[f.resample_group_a, f.resample_group_b], return_inverse=True)
    a, b = np.split(idx, 2)
    rng = np.random.default_rng(42)
    orders = {}
    if edge and known:
        ties = np.array([hashlib.sha256(i.encode()).hexdigest() for i in f.index])
        for c in ["gnn", "lomap"]:
            orders[c] = [idx[np.lexsort((ties[idx], xs[c][idx]))]
                         for idx in f.groupby("source_sha256").indices.values()]
    draws = []
    for i in range(1001):
        counts = (rng.multinomial(len(groups), np.full(len(groups), 1 / len(groups)))
                  if i else np.ones(len(groups), dtype=int))
        w = counts[a] * np.where(a == b, 1, counts[b])
        if np.count_nonzero(w) < 3:
            continue
        yr = ranks(ry, w)
        row = {}
        for c, x in xs.items():
            row[c + ".pearson"] = corr(y, x, w)
            row[c + ".spearman"] = corr(yr, ranks(rx[c], w), w)
            if c != "lomap":
                row[c + ".mse"] = np.average((x-y)**2, weights=w)
        for c, order in orders.items():
            row[c + ".selected"], n = selection(y, w, order)
        if orders:
            row["difference.selected"] = row["gnn.selected"] - row["lomap.selected"]
            if not i:
                selected = n
        draws.append(row)
    df = pd.DataFrame(draws)
    metricset = "common_metrics" if common else "all_graph_valid_metrics"
    result = {}
    for key, point in df.iloc[0].items():
        c, m = key.split(".")
        if m != "selected":
            expected = (saved_other if c == "other" else saved)[metricset]["gnn" if c == "other" else c][m]
            np.testing.assert_allclose(point, expected, atol=1e-10)
        vals = df[key].iloc[1:].dropna()
        assert len(vals) >= 200
        result[key] = [point, *np.quantile(vals, [.025, .975]), len(vals)]
    if orders:
        for m in ("pearson", "spearman"):
            delta = df["gnn."+m].iloc[1:] - df["lomap."+m].iloc[1:]
            np.testing.assert_allclose(np.quantile(delta, [.025, .975]), saved["paired_bootstrap"]
                                       ["lomap"]["differences"]["gnn_minus_lomap_"+m]["ci95"])
    return dict(rows=len(f), sources=f.source_sha256.nunique(), groups=len(groups),
                selected=selected if orders else None, statistics=result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_root", type=Path)
    args = parser.parse_args()
    print(json.dumps({s + "/" + t: analyze(args.artifact_root, s, t)
                      for s in ("known_reference", "strict_network")
                      for t in ("edge", "pilot_complex", "pilot_solvent")}, indent=2, allow_nan=False))
