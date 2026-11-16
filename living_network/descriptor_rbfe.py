"""Cluster experiment: descriptor boosting versus real LOMAP on a frozen split.

Training pairs may be capped explicitly for tractable LOMAP preprocessing.
Validation and test membership are never sampled or reassigned. Test labels
are used only for final reporting, after model selection.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from importlib.metadata import version
import json
import os
import pickle

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import average_precision_score, mean_absolute_error, mean_squared_error, roc_auc_score
from threadpoolctl import threadpool_limits

import hybrid_topo_rbfe as hto
import rbfe_artifacts as artifacts


FEATURE_VERSION = 1
DESCRIPTORS = [
    ("heavy_atoms", Descriptors.HeavyAtomCount),
    ("mol_weight", Descriptors.MolWt),
    ("logp", Descriptors.MolLogP),
    ("tpsa", Descriptors.TPSA),
    ("hbd", Descriptors.NumHDonors),
    ("hba", Descriptors.NumHAcceptors),
    ("rotatable_bonds", Descriptors.NumRotatableBonds),
    ("rings", Descriptors.RingCount),
    ("aromatic_rings", Descriptors.NumAromaticRings),
    ("fraction_csp3", Descriptors.FractionCSP3),
    ("formal_charge", Chem.GetFormalCharge),
]
FEATURE_NAMES = (
    [f"sum_{name}" for name, _ in DESCRIPTORS]
    + [f"abs_delta_{name}" for name, _ in DESCRIPTORS]
    + ["morgan_tanimoto_difficulty", "morgan_shared_bits", "morgan_changed_bits"]
)


@lru_cache(maxsize=100_000)
def molecule_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    desc = np.array([fn(mol) for _, fn in DESCRIPTORS], dtype=float)
    fp = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048).GetFingerprint(mol)
    return desc, fp


def pair_features(pair):
    a, b = map(molecule_features, pair)
    if a is None or b is None:
        return np.full(len(FEATURE_NAMES), np.nan)
    da, fa = a
    db, fb = b
    sa, sb = set(fa.GetOnBits()), set(fb.GetOnBits())
    return np.r_[da + db, np.abs(da - db),
                 1 - DataStructs.TanimotoSimilarity(fa, fb),
                 len(sa & sb), len(sa ^ sb)]


def lomap_worker(task):
    a, b, timeout = task
    return hto._lomap_like_score(a, b, timeout)


def select_rows(df, train_cap, seed):
    """Return explicit train subset, followed by ALL validation and test rows."""
    train = df[df.split == "train"]
    if train_cap > 0 and len(train) > train_cap:
        train = train.sample(n=train_cap, random_state=seed).sort_index()
    return pd.concat([train, df[df.split == "val"], df[df.split == "test"]]).reset_index(drop=True)


def split_diagnostics(df):
    scaffold = {}
    for smi in pd.unique(pd.concat([df[artifacts.A], df[artifacts.B]])):
        scaffold[smi] = artifacts.scaffold_smiles(smi)
    groups, result = {}, {}
    for split in ("train", "val", "test"):
        sub = df[df.split == split]
        ligands = set(sub[artifacts.A]) | set(sub[artifacts.B])
        scaffolds = {scaffold[s] for s in ligands if scaffold[s] is not None}
        groups[split] = (ligands, scaffolds)
        y = sub.SE_abs.to_numpy()
        result[split] = {
            "pairs": len(sub), "unique_ligands": len(ligands),
            "unique_scaffolds": len(scaffolds),
            "unresolved_scaffold_ligands": sum(scaffold[s] is None for s in ligands),
            "empty_scaffold_present": "" in scaffolds,
            "label_mean": float(y.mean()), "label_std": float(y.std()),
            "label_quantiles": {str(q): float(np.quantile(y, q)) for q in (0, .25, .5, .75, .95, 1)},
        }
    result["overlap"] = {
        f"{a}_{b}": {"ligands": len(groups[a][0] & groups[b][0]),
                      "scaffolds": len(groups[a][1] & groups[b][1])}
        for a, b in (("train", "val"), ("train", "test"), ("val", "test"))
    }
    return result


def json_dump(value, path):
    with open(path, "w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def score_lomap(pairs, indices, args, request):
    """Checkpoint completed scores, including failed mappings, every 100 results.

    Only the parent writes. Metadata and arrays share one atomically replaced
    file so interrupted writes cannot certify a different row order.
    """
    path = os.path.join(args.output_dir, "lomap_progress.pkl")
    scores = np.full(len(pairs), np.nan)
    done = np.zeros(len(pairs), dtype=bool)
    if os.path.exists(path):
        with open(path, "rb") as stream:
            saved = pickle.load(stream)
        if saved["request"] != request or saved["pair_keys"] != pairs:
            raise ValueError("LOMAP progress settings/pairs mismatch; use a fresh output directory.")
        scores, done = saved["scores"], saved["done"]
        if scores.shape != (len(pairs),) or done.shape != (len(pairs),):
            raise ValueError("Invalid LOMAP progress array lengths.")

    def save():
        temporary = path + ".tmp"
        with open(temporary, "wb") as stream:
            pickle.dump({"request": request, "pair_keys": pairs,
                         "scores": scores, "done": done}, stream)
        os.replace(temporary, path)

    pending = [i for i in indices if not done[i]]
    print(f"LOMAP: {len(indices)-len(pending):,} cached; {len(pending):,} remaining", flush=True)
    if not pending:
        return scores
    tasks = ((pairs[i][0], pairs[i][1], args.mcs_timeout) for i in pending)
    try:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for count, (i, score) in enumerate(
                    zip(pending, pool.map(lomap_worker, tasks, chunksize=8)), start=1):
                scores[i] = score if np.isfinite(score) else np.nan
                done[i] = True
                if count % 100 == 0:
                    save()
                    print(f"LOMAP {int(done[indices].sum()):,}/{len(indices):,}", flush=True)
    finally:
        # A hard scheduler kill still retains the last 100-result checkpoint.
        save()
    return scores


def load_features(df, args, benchmark):
    rows = select_rows(df, args.train_cap, args.seed)
    # Import checks happen before costly work: a missing LOMAP install must
    # never masquerade as molecular mapping failures.
    import gufe
    import lomap
    if not hasattr(lomap, "LomapAtomMapper") or not hasattr(lomap, "default_lomap_score"):
        raise RuntimeError("Install the LOMAP implementation exposing LomapAtomMapper/default_lomap_score.")
    request = {
        "benchmark": benchmark, "selected_rows": artifacts.benchmark_identity(rows),
        "feature_version": FEATURE_VERSION, "feature_names": FEATURE_NAMES,
        "train_cap": args.train_cap, "seed": args.seed, "mcs_timeout": args.mcs_timeout,
        "lomap_mode": args.lomap_mode,
        "lomap_scoring_version": artifacts.LOMAP_SCORING_VERSION,
        "mapper_settings": {"threed": False, "max3d": 0.0, "shift": False},
        "rdkit": rdBase.rdkitVersion,
        "gufe": version("gufe"), "lomap": getattr(lomap, "__version__", None),
        "lomap_module_sha256": artifacts.file_hash(lomap.__file__),
    }
    path = os.path.join(args.output_dir, "features.pkl")
    if os.path.exists(path):
        artifacts.require_cache_manifest(path, request)
        with open(path, "rb") as stream:
            payload = pickle.load(stream)
        if payload["pair_keys"] != artifacts.pair_keys(rows):
            raise ValueError("Feature cache pair identities mismatch.")
        return rows, payload["X"], payload["lomap"], request
    pairs = artifacts.pair_keys(rows)
    print(f"Computing descriptors for {len(pairs):,} selected pairs", flush=True)
    X = np.asarray([pair_features(pair) for pair in pairs], dtype=np.float32)
    X[~np.isfinite(X)] = np.nan
    indices = (np.flatnonzero(rows.split.to_numpy() == "test")
               if args.lomap_mode == "test-only" else np.arange(len(rows)))
    print(f"Scoring real LOMAP on {len(indices):,} pairs with {args.workers} workers.", flush=True)
    scores = score_lomap(pairs, indices, args, request)
    with open(path, "wb") as stream:
        pickle.dump({"pair_keys": pairs, "X": X, "lomap": scores}, stream)
    artifacts.write_cache_manifest(path, request)
    return rows, X, scores, request


def select_model(X_train, y_train, X_val, y_val, seed):
    """Fixed small search; only validation Spearman selects a model.

    Internal random validation is disabled. Nonfinite correlations are rejected;
    ties retain the earlier (smaller) candidate. Test data is never passed here.
    """
    history, best, best_rho = [], None, -np.inf
    for leaves in (15, 31):
        model = HistGradientBoostingRegressor(
            max_leaf_nodes=leaves, learning_rate=.05, l2_regularization=1.,
            min_samples_leaf=30, early_stopping=False, warm_start=True,
            random_state=seed,
        )
        for iterations in (50, 100, 200):
            model.set_params(max_iter=iterations)
            model.fit(X_train, y_train)
            pred = model.predict(X_val)
            rho = float(spearmanr(y_val, pred).statistic)
            history.append({"leaves": leaves, "iterations": iterations,
                            "val_spearman": rho if np.isfinite(rho) else None,
                            "val_mse": float(mean_squared_error(y_val, pred))})
            if np.isfinite(rho) and rho > best_rho:
                best_rho = rho
                # Snapshot now: warm-start mutates the candidate afterwards.
                best = pickle.loads(pickle.dumps(model))
    if best is None:
        raise ValueError("No candidate produced a finite validation rank correlation.")
    return best, history


def metric_rows(y, scores, regression_names):
    """All scores must already share one finite eligibility mask.

    Top-quarter labels use the held-out cohort quantile for retrospective
    ranking, matching the existing report. This is not a deployment cutoff.
    """
    hard = y >= np.quantile(y, .75)
    result = []
    for name, score in scores.items():
        nonconstant = np.ptp(score) > 0 and np.ptp(y) > 0
        row = {"method": name, "n": len(y),
               "pearson_r": float(pearsonr(y, score).statistic) if nonconstant else None,
               "spearman_r": float(spearmanr(y, score).statistic) if nonconstant else None,
               "hard_prevalence": float(hard.mean()),
               "auroc_hard_top25": float(roc_auc_score(hard, score)) if len(np.unique(hard)) == 2 else None,
               "auprc_hard_top25": float(average_precision_score(hard, score)) if len(np.unique(hard)) == 2 else None}
        if name in regression_names:
            row.update(mse=float(mean_squared_error(y, score)),
                       mae=float(mean_absolute_error(y, score)))
        result.append(row)
    return result


def gnn_predictions(frame, args, benchmark):
    """Inference only, preserving failed graphs as NaNs in their original rows."""
    import torch
    checkpoint = torch.load(args.hybrid_ckpt, map_location=hto.DEVICE, weights_only=False)
    artifacts.require_benchmark(checkpoint, benchmark)
    settings = checkpoint["args"]
    if settings["mcs_timeout"] != args.mcs_timeout:
        raise ValueError("GNN checkpoint and graph MCS timeout differ.")
    model = hto.HybridGNN(hto.ATOM_DIM, hto.BOND_DIM, settings["hidden_dim"],
                          settings["num_layers"], settings["embed_dim"],
                          settings["dropout"]).to(hto.DEVICE)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    graphs = hto._load_graphs("test", args.graph_cache_dir,
                              pair_df=frame, mcs_timeout=args.mcs_timeout)
    if graphs is None:
        raise ValueError("GNN comparison requires the verified graph cache.")
    pred = np.full(len(frame), np.nan)
    with torch.no_grad():
        for start in range(0, len(graphs), 256):
            indices = [i for i in range(start, min(start+256, len(graphs))) if graphs[i] is not None]
            if not indices:
                continue
            batch, _ = hto._collate([(graphs[i], 0.) for i in indices])
            batch = {k: v.to(hto.DEVICE) for k, v in batch.items()}
            pred[indices] = model(batch).cpu().numpy()
    return pred


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--reference_dir", required=True, help="Existing results_hybrid_topo_v3 directory")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--train_cap", type=int, default=100_000,
                        help="Explicit train-only cap for both variants; 0 uses all training pairs")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--mcs_timeout", type=int, default=5)
    parser.add_argument("--lomap_mode", choices=("both", "test-only"), default="both",
                        help="test-only: train descriptors alone; LOMAP only benchmarks test pairs")
    parser.add_argument("--hybrid_ckpt")
    parser.add_argument("--graph_cache_dir")
    parser.add_argument("--phases", default="features,train,evaluate")
    args = parser.parse_args()
    phases = args.phases.split(",")
    if not set(phases) <= {"features", "train", "evaluate"}:
        parser.error("phases must be features,train,evaluate")
    if args.train_cap < 0 or min(args.workers, args.threads, args.mcs_timeout) < 1:
        parser.error("cap must be nonnegative; workers/threads/timeout must be positive")
    if bool(args.hybrid_ckpt) != bool(args.graph_cache_dir):
        parser.error("Provide both --hybrid_ckpt and --graph_cache_dir, or neither")
    if os.path.realpath(args.output_dir) == os.path.realpath(args.reference_dir):
        parser.error("Use a separate output directory; preserve the GNN reference run.")
    # Freeze to the completed experiment, not a newly chosen split.
    args.seed, args.sample_size = 42, 0
    args.split_method, args.val_frac, args.test_frac = "scaffold", .1, .1
    with open(os.path.join(args.reference_dir, "benchmark.json")) as stream:
        expected = json.load(stream)
    df = hto.phase1_load(args, args.reference_dir)
    benchmark = artifacts.benchmark_identity(df)
    artifacts.require_benchmark({"benchmark": benchmark}, expected)
    os.makedirs(args.output_dir, exist_ok=True)
    json_dump(vars(args), os.path.join(args.output_dir, "config.json"))
    json_dump(benchmark, os.path.join(args.output_dir, "benchmark.json"))
    if "features" in phases:
        json_dump(split_diagnostics(df), os.path.join(args.output_dir, "split_diagnostics.json"))
    elif not os.path.exists(os.path.join(args.output_dir, "features.pkl")):
        raise ValueError("Run the features phase first.")
    rows, X, lomap, request = load_features(df, args, benchmark)
    y = rows.SE_abs.to_numpy()
    test = rows.split.to_numpy() == "test"
    valid = np.isfinite(X).all(axis=1) & np.isfinite(y)
    if args.lomap_mode == "both":
        valid &= np.isfinite(lomap)
    else:
        valid &= ~test | np.isfinite(lomap)
    train = (rows.split.to_numpy() == "train") & valid
    val = (rows.split.to_numpy() == "val") & valid
    coverage = {
        s: {"selected": int((rows.split == s).sum()),
            "eligible": int(((rows.split == s).to_numpy() & valid).sum())}
        for s in ("train", "val", "test")
    }
    json_dump(coverage, os.path.join(args.output_dir, "coverage.json"))
    print(f"Eligibility ({args.lomap_mode}):", coverage, flush=True)
    if min(train.sum(), val.sum(), (test & valid).sum()) < 2:
        raise ValueError("Insufficient common eligible rows; inspect LOMAP installation and coverage.")
    # In both mode the variants share training/validation rows. Test-only mode
    # never conditions training eligibility on uncomputed LOMAP scores.
    variants = {"descriptors": X}
    if args.lomap_mode == "both":
        variants["descriptors_lomap"] = np.c_[X, lomap]
    model_path = os.path.join(args.output_dir, "models.pkl")
    with threadpool_limits(limits=args.threads):
        if "train" in phases:
            models, history = {}, {}
            for name, features in variants.items():
                models[name], history[name] = select_model(
                    features[train], y[train], features[val], y[val], args.seed)
            payload = {"benchmark": benchmark, "request": request, "models": models,
                       "train_mean": float(y[train].mean())}
            with open(model_path, "wb") as stream:
                pickle.dump(payload, stream)
            json_dump(history, os.path.join(args.output_dir, "validation_search.json"))
        if "evaluate" not in phases:
            return
        with open(model_path, "rb") as stream:
            payload = pickle.load(stream)
        artifacts.require_benchmark(payload, benchmark)
        if payload["request"] != request:
            raise ValueError("Model training subset or feature settings mismatch.")
        frame = rows.loc[test].reset_index(drop=True)
        scores = {name: model.predict(variants[name][test])
                  for name, model in payload["models"].items()}
        scores["train_mean"] = np.full(test.sum(), payload["train_mean"])
        scores["LOMAP (real, 1-sim)"] = lomap[test]
        scores["Tanimoto (1-sim)"] = X[test, FEATURE_NAMES.index("morgan_tanimoto_difficulty")]
        if args.hybrid_ckpt:
            scores["Hybrid-topo GNN"] = gnn_predictions(frame, args, benchmark)
        common = valid[test].copy()
        for pred in scores.values():
            common &= np.isfinite(pred)
        if common.sum() < 2:
            raise ValueError("Insufficient common test predictions.")
        output = frame[[artifacts.A, artifacts.B, "SE_abs"]].copy()
        for name, pred in scores.items():
            output[name] = pred
        output["common_eligible"] = common
        output.to_csv(os.path.join(args.output_dir, "test_predictions.csv"), index=False)
        coverage["comparison"] = {"test_total": len(frame), "common_eligible": int(common.sum()),
                                  "finite_predictions": {k: int(np.isfinite(v).sum()) for k, v in scores.items()}}
        json_dump(coverage, os.path.join(args.output_dir, "coverage.json"))
        metrics = metric_rows(y[test][common], {k: v[common] for k, v in scores.items()},
                              {"descriptors", "descriptors_lomap", "train_mean", "Hybrid-topo GNN"})
        pd.DataFrame(metrics).to_csv(os.path.join(args.output_dir, "comparison.csv"), index=False)
        print(pd.DataFrame(metrics).to_string(index=False))
        print("Paired scores only; no significance or independent-pair confidence claim.")


if __name__ == "__main__":
    main()
