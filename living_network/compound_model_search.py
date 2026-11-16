"""Local compound-level model search on the exported v4 compound values.

Inputs are small cluster exports: train_compound_values.csv (two-slot u/v
recovered from training edges) and the compound_model_v1 test_predictions.csv.

Selection uses scaffold-grouped K-fold cross-validation on TRAINING compounds
only. Target s = (u - v) / 2, because u ~= -v. Test pairs are scored once, for
the CV-selected configuration and for the cluster reference (two u/v models,
baseline features, 7 leaves / 50 iterations), which must reproduce the
cluster's test result as a data-consistency check.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import compound_uncertainty_model as cm
import descriptor_rbfe as desc
import rbfe_artifacts as artifacts


SEARCH_VERSION = 1
LOMAP = cm.LOMAP


def feature_sets(smiles):
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    if any(m is None for m in mols):
        raise ValueError("Unparseable SMILES in compound table.")
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    rdkit = np.array([[f(m) for _, f in Descriptors.descList] for m in mols], dtype=float)
    rdkit[~np.isfinite(rdkit)] = np.nan
    # Signed log tames heavy-tailed descriptors (e.g. Ipc) for linear models.
    rdkit = np.sign(rdkit) * np.log1p(np.abs(rdkit))
    counts = np.array([generator.GetCountFingerprintAsNumPy(m) for m in mols], dtype=float)
    return {
        "baseline": cm.compound_features(smiles),
        "rdkit": rdkit,
        "morgan_counts": counts,
        "rdkit+morgan": np.c_[rdkit, counts],
        "morgan_bits": counts > 0,
    }


def candidates(seed):
    def scaled(model):
        return lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), model)
    grid = []
    for features in ("baseline", "rdkit", "morgan_counts", "rdkit+morgan"):
        for alpha in (1, 10, 100, 1000):
            grid.append((features, f"ridge_a{alpha}", scaled(Ridge(alpha=alpha))))
        for iterations in (50, 100):
            grid.append((features, f"hgb_l7_i{iterations}", lambda i=iterations: HistGradientBoostingRegressor(
                max_leaf_nodes=7, max_iter=i, learning_rate=.05, l2_regularization=1.,
                min_samples_leaf=20, early_stopping=False, random_state=seed)))
        grid.append((features, "rf", lambda: make_pipeline(SimpleImputer(strategy="median"), RandomForestRegressor(
            n_estimators=300, min_samples_leaf=5, max_features=.3, n_jobs=-1, random_state=seed))))
    for k in (5, 10, 20):
        grid.append(("morgan_bits", f"knn_jaccard_k{k}",
                     lambda k=k: KNeighborsRegressor(n_neighbors=k, metric="jaccard")))
    return grid


def cross_validate(features, s, groups, grid, folds):
    rows, oof = [], {}
    splitter = GroupKFold(n_splits=folds)
    for feature_name, model_name, factory in grid:
        X = features[feature_name]
        prediction = np.empty(len(s))
        for fit, held in splitter.split(X, s, groups):
            prediction[held] = factory().fit(X[fit], s[fit]).predict(X[held])
        key = f"{feature_name}|{model_name}"
        oof[key] = prediction
        rows.append({"config": key, "cv_spearman": float(spearmanr(s, prediction).statistic),
                     "cv_pearson": float(pearsonr(s, prediction).statistic),
                     "cv_r2": cm.labels.r_squared(s, prediction)})
        print(f"{key:36s} spearman={rows[-1]['cv_spearman']:.4f} pearson={rows[-1]['cv_pearson']:.4f}", flush=True)
    return pd.DataFrame(rows).sort_values("cv_spearman", ascending=False, kind="stable"), oof


def evaluate(test, y_hat, s_by_smiles, test_values, min_edges):
    y = test[artifacts.Y].to_numpy(dtype=float)
    keep = test_values[(test_values.n_first >= min_edges) & (test_values.n_second >= min_edges)]
    recovered = ((keep.u - keep.v) / 2).to_numpy()
    scores = {"model": np.abs(y_hat), LOMAP: test[LOMAP].to_numpy(dtype=float),
              "Tanimoto (1-sim)": test["Tanimoto (1-sim)"].to_numpy(dtype=float)}
    return {
        "signed": cm.signed_metrics(y, y_hat),
        "compound_spearman_vs_test_recovered": (
            float(spearmanr(s_by_smiles.loc[keep.smiles], recovered).statistic) if s_by_smiles is not None else None),
        "compounds": int(len(keep)),
        "magnitude": desc.metric_rows(np.abs(y), scores, {"model"}),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_values", required=True)
    parser.add_argument("--test_predictions", required=True)
    parser.add_argument("--summary", required=True, help="compound_model_v1 summary.json (train intercept)")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--min_edges", type=int, default=10)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    seed = 42
    table = pd.read_csv(args.train_values)
    test = pd.read_csv(args.test_predictions)
    artifacts.pair_keys(test)
    if set(table.smiles) & (set(test[artifacts.A]) | set(test[artifacts.B])):
        raise ValueError("Training and test compounds overlap.")
    train = table[(table.n_first >= args.min_edges) & (table.n_second >= args.min_edges)].reset_index(drop=True)
    s = ((train.u - train.v) / 2).to_numpy()
    # Pair intercept for y = c + s_A - s_B: c + mean(u + v) over training compounds.
    with open(args.summary) as stream:
        c = json.load(stream)["train_intercept"]
    intercept = c + float((train.u + train.v).mean())
    groups = [artifacts.scaffold_smiles(x) for x in train.smiles]
    test_smiles = pd.unique(pd.concat([test[artifacts.A], test[artifacts.B]]))
    all_smiles = np.r_[train.smiles.to_numpy(), test_smiles]
    features = feature_sets(all_smiles)
    n = len(train)
    train_features = {k: v[:n] for k, v in features.items()}
    test_features = {k: pd.DataFrame(v[n:], index=test_smiles) for k, v in features.items()}

    grid = candidates(seed)
    ranking, _ = cross_validate(train_features, s, groups, grid, args.folds)
    best = ranking.iloc[0].config
    feature_name, model_name = best.split("|")
    factory = dict(((f"{f}|{m}"), fac) for f, m, fac in grid)[best]

    # Single final test look: CV-selected model.
    model = factory().fit(train_features[feature_name], s)
    s_test = pd.Series(model.predict(test_features[feature_name].to_numpy()), index=test_smiles)
    y_hat = intercept + s_test.loc[test[artifacts.A]].to_numpy() - s_test.loc[test[artifacts.B]].to_numpy()
    test_values, _, ceiling = cm.fit_two_slot(test)
    selected = evaluate(test, y_hat, s_test, test_values, args.min_edges)

    # Reference reproduction of the cluster model (two u/v models, baseline).
    reference_models = []
    for target, count in (("u", "n_first"), ("v", "n_second")):
        subset = table[table[count] >= args.min_edges]
        reference_models.append(HistGradientBoostingRegressor(
            max_leaf_nodes=7, max_iter=50, learning_rate=.05, l2_regularization=1., min_samples_leaf=20,
            early_stopping=False, random_state=seed).fit(cm.compound_features(subset.smiles.tolist()), subset[target]))
    y_reference = cm.pair_predict(reference_models, c, test_features["baseline"], test)
    reference = evaluate(test, y_reference, None, test_values, args.min_edges)

    result = {"search_version": SEARCH_VERSION, "selection": "scaffold-grouped CV on training compounds",
              "folds": args.folds, "training_compounds": n, "intercept": intercept, "selected": best,
              "test_ceiling_r2": ceiling, "selected_test": selected, "reference_reproduction_test": reference}
    os.makedirs(args.output_dir, exist_ok=True)
    ranking.to_csv(os.path.join(args.output_dir, "cv_ranking.csv"), index=False)
    desc.json_dump(result, os.path.join(args.output_dir, "result.json"))
    output = test[[artifacts.A, artifacts.B, artifacts.Y, "SE_abs", LOMAP, "Tanimoto (1-sim)"]].copy()
    output["model_signed"], output["model"] = y_hat, np.abs(y_hat)
    output["reference_signed"] = y_reference
    output.to_csv(os.path.join(args.output_dir, "test_predictions.csv"), index=False)
    print(json.dumps({k: v for k, v in result.items()}, indent=1, default=str))


if __name__ == "__main__":
    main()
