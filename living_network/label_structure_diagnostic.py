"""Does the signed label behave like a difference of per-compound quantities?

Training split only: no validation/test labels, models, LOMAP, or graphs.
Self-pairs (A == A) are reported separately and excluded from the fits,
because a difference model predicts exactly zero for them.

1. Reversed pairs: is y(B, A) close to -y(A, B)?
2. Edge-held-out fits: difference model y = s_A - s_B versus the more general
   two-slot model y = c + u_A + v_B. Edges are held out, not compounds.
3. Triangle closure: residual y_AB + y_BC - y_AC versus a permutation null.
   A ratio near 0 means the labels add around cycles; near 1 means they don't.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import lsqr
from scipy.stats import pearsonr, spearmanr

import descriptor_rbfe as desc
import hybrid_topo_rbfe as hto
import rbfe_artifacts as artifacts


DIAGNOSTIC_VERSION = 1


def encode(frame):
    codes, names = pd.factorize(pd.concat([frame[artifacts.A], frame[artifacts.B]], ignore_index=True))
    m = len(frame)
    return codes[:m].astype(np.int64), codes[m:].astype(np.int64), len(names)


def lookup(src, dst, n, query_src, query_dst):
    """Edge index for each ordered query pair, or -1 if absent."""
    keys = src * n + dst
    order = np.argsort(keys)
    sorted_keys = keys[order]
    query = query_src * n + query_dst
    pos = np.minimum(np.searchsorted(sorted_keys, query), len(sorted_keys) - 1)
    return np.where(sorted_keys[pos] == query, order[pos], -1)


def correlations(x, y):
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None, None
    return float(pearsonr(x, y).statistic), float(spearmanr(x, y).statistic)


def reversed_pairs(src, dst, y, n):
    reverse = lookup(src, dst, n, dst, src)
    forward = np.flatnonzero((reverse >= 0) & (src < dst))
    yf, yr = y[forward], y[reverse[forward]]
    result = {"pairs": int(len(forward))}
    if len(forward) < 3:
        return result
    signed_r, signed_rho = correlations(yf, -yr)
    magnitude_r, magnitude_rho = correlations(np.abs(yf), np.abs(yr))
    result.update(
        pearson_forward_vs_negated_reverse=signed_r,
        spearman_forward_vs_negated_reverse=signed_rho,
        pearson_magnitudes=magnitude_r, spearman_magnitudes=magnitude_rho,
        fraction_opposite_sign=float(np.mean(np.sign(yf) == -np.sign(yr))),
        rms_antisymmetry_residual=float(np.sqrt(np.mean((yf + yr) ** 2))),
        rms_symmetry_residual=float(np.sqrt(np.mean((yf - yr) ** 2))),
        rms_forward_label=float(np.sqrt(np.mean(yf ** 2))),
    )
    return result


def design(src, dst, n, kind):
    m = len(src)
    rows = np.r_[np.arange(m), np.arange(m)]
    if kind == "difference":
        return sparse.csr_matrix((np.r_[np.ones(m), -np.ones(m)], (rows, np.r_[src, dst])), shape=(m, n))
    return sparse.csr_matrix((np.ones(2 * m), (rows, np.r_[src, dst + n])), shape=(m, 2 * n))


def r_squared(y, prediction):
    return float(1 - np.sum((y - prediction) ** 2) / np.sum((y - y.mean()) ** 2))


def fit_models(src, dst, y, n, seed, holdout=.1):
    held = np.random.default_rng(seed).random(len(y)) < holdout
    if held.sum() < 3 or (~held).sum() < 3:
        raise ValueError("Need at least three fitting and three held-out edges.")
    seen = np.zeros(n, dtype=bool)
    seen[src[~held]] = seen[dst[~held]] = True
    result = {"fit_edges": int((~held).sum()), "heldout_edges": int(held.sum()),
              "heldout_edges_with_unseen_compound": int((~(seen[src[held]] & seen[dst[held]])).sum())}
    for kind in ("difference", "two_slot"):
        # The difference model is antisymmetric, so it has no intercept.
        offset = 0. if kind == "difference" else float(y[~held].mean())
        X_fit, X_held = design(src[~held], dst[~held], n, kind), design(src[held], dst[held], n, kind)
        coefficients = lsqr(X_fit, y[~held] - offset, atol=1e-10, btol=1e-10, iter_lim=10_000)[0]
        fitted, predicted = X_fit @ coefficients + offset, X_held @ coefficients + offset
        magnitude_r, magnitude_rho = correlations(np.abs(y[held]), np.abs(predicted))
        result[kind] = {
            "fit_r2": r_squared(y[~held], fitted), "heldout_r2": r_squared(y[held], predicted),
            "heldout_magnitude_pearson": magnitude_r, "heldout_magnitude_spearman": magnitude_rho,
        }
    return result


def triangle_closure(src, dst, y, n, samples, seed):
    """Random A->B->C paths whose ordered edge A->C also exists."""
    rng = np.random.default_rng(seed)
    order = np.argsort(src, kind="stable")
    start = np.searchsorted(src[order], np.arange(n))
    count = np.searchsorted(src[order], np.arange(n), side="right") - start
    first = rng.integers(len(y), size=samples)
    a, b = src[first], dst[first]
    has_next = count[b] > 0
    first, a, b = first[has_next], a[has_next], b[has_next]
    second = order[start[b] + (rng.random(len(b)) * count[b]).astype(np.int64)]
    c = dst[second]
    third = lookup(src, dst, n, a, c)
    keep = (third >= 0) & (c != a)
    first, second, third = first[keep], second[keep], third[keep]
    result = {"sampled_paths": samples, "closed_triangles": int(keep.sum()),
              "unique_triangles": int(len(np.unique(np.c_[first, second, third], axis=0)))}
    if keep.sum() < 3:
        return result
    residual = y[first] + y[second] - y[third]
    shuffled = rng.permutation(y)
    null = shuffled[first] + shuffled[second] - shuffled[third]
    rms, rms_null = float(np.sqrt(np.mean(residual ** 2))), float(np.sqrt(np.mean(null ** 2)))
    closure_r, closure_rho = correlations(y[first] + y[second], y[third])
    result.update(rms_residual=rms, rms_permutation_null=rms_null, ratio_to_null=rms / rms_null,
                  pearson_path_sum_vs_direct=closure_r, spearman_path_sum_vs_direct=closure_rho)
    return result


def diagnose(frame, seed=42, holdout=.1, triangle_samples=200_000):
    y_all = frame[artifacts.Y].to_numpy(dtype=float)
    if not np.isfinite(y_all).all():
        raise ValueError("Signed labels must be finite.")
    self_pair = (frame[artifacts.A] == frame[artifacts.B]).to_numpy()
    edges = frame.loc[~self_pair]
    src, dst, n = encode(edges)
    y = y_all[~self_pair]
    return {
        "edges": int(len(edges)), "compounds": int(n),
        "signed_label": {"mean": float(y.mean()), "std": float(y.std()),
                         "fraction_positive": float(np.mean(y > 0))},
        "self_pairs": {"count": int(self_pair.sum()),
                       "mean_abs_label": float(np.abs(y_all[self_pair]).mean()) if self_pair.any() else None},
        "reversed_pairs": reversed_pairs(src, dst, y, n),
        "models": fit_models(src, dst, y, n, seed, holdout),
        "triangles": triangle_closure(src, dst, y, n, triangle_samples, seed),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--reference_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--triangle_samples", type=int, default=200_000)
    args = parser.parse_args()
    if args.triangle_samples < 1:
        parser.error("triangle_samples must be positive")
    if os.path.realpath(args.reference_dir) == os.path.realpath(args.output_dir):
        parser.error("Use a separate diagnostic output directory.")
    args.seed, args.sample_size = 42, 0
    args.split_method, args.val_frac, args.test_frac = "scaffold", .1, .1
    with open(os.path.join(args.reference_dir, "benchmark.json")) as stream:
        expected = json.load(stream)
    df = hto.phase1_load(args, args.reference_dir)
    benchmark = artifacts.benchmark_identity(df)
    artifacts.require_benchmark({"benchmark": benchmark}, expected)
    result = {"diagnostic_version": DIAGNOSTIC_VERSION, "benchmark": benchmark, "split": "train",
              "seed": args.seed, "holdout_fraction": .1,
              **diagnose(df[df.split == "train"], args.seed, .1, args.triangle_samples)}
    os.makedirs(args.output_dir, exist_ok=True)
    desc.json_dump(result, os.path.join(args.output_dir, "label_structure.json"))
    print(json.dumps({k: v for k, v in result.items() if k != "benchmark"}, indent=2))


if __name__ == "__main__":
    main()
