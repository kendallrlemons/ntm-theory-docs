"""Small cluster-only audit of the historical LOMAP scoring path.

This diagnoses selected cases, not a performance estimate. No training or
split regeneration occurs. Default: up to 8 cases in each disjoint category,
plus two simple controls. Completed cases are resumable.
"""

import argparse
import inspect
import json
import os

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem

import rbfe_artifacts as artifacts


SCORE = "LOMAP (real, 1-sim)"
AUDIT_VERSION = 2


def select_cases(frame, per_group=8, seed=42):
    """Fixed stratified diagnostic sample; self-pairs are their own category."""
    if per_group < 1:
        raise ValueError("per_group must be positive")
    same = frame[artifacts.A] == frame[artifacts.B]
    finite = np.isfinite(frame[SCORE])
    masks = {
        "identical_smiles": same,
        "missing": ~same & ~finite,
        "saturated": ~same & finite & (frame[SCORE] == 1.),
        "other_finite": ~same & finite & (frame[SCORE] != 1.),
    }
    cases, counts = [], {}
    for group, mask in masks.items():
        subset = frame.loc[mask]
        counts[group] = len(subset)
        for index, row in subset.sample(n=min(per_group, len(subset)), random_state=seed).iterrows():
            cases.append({
                "id": f"row-{index}", "group": group,
                "smiles_a": row[artifacts.A], "smiles_b": row[artifacts.B],
                "label": float(row.SE_abs),
                "historical_difficulty": float(row[SCORE]) if np.isfinite(row[SCORE]) else None,
            })
    cases.extend([
        {"id": "control-identical", "group": "control", "smiles_a": "Cc1ccccc1",
         "smiles_b": "Cc1ccccc1", "label": None, "historical_difficulty": None},
        {"id": "control-small-change", "group": "control", "smiles_a": "Cc1ccccc1",
         "smiles_b": "CCc1ccccc1", "label": None, "historical_difficulty": None},
    ])
    return cases, counts


def audit_case(case, lomap, gufe, scorers, timeout, max3d=1.0):
    """Mirror baseline preparation/mapping; expose failure stage and sub-scores."""
    result = dict(case)
    result["mapper_settings"] = {"time": timeout, "threed": False, "max3d": max3d, "shift": False}
    result["preparation"] = {}
    stage = "parse_a"
    try:
        molecules = []
        for endpoint in ("a", "b"):
            stage = f"parse_{endpoint}"
            mol = Chem.MolFromSmiles(case[f"smiles_{endpoint}"])
            if mol is None:
                raise ValueError("RDKit could not parse SMILES")
            mol = Chem.AddHs(mol)
            stage = f"embed_{endpoint}"
            status = AllChem.EmbedMolecule(mol, randomSeed=0xC0FFEE)
            if status != 0:
                raise ValueError(f"EmbedMolecule returned {status}")
            stage = f"mmff_{endpoint}"
            status = AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
            result["preparation"][endpoint] = {
                "mmff_status": status, "heavy_atoms": mol.GetNumHeavyAtoms(),
                "formal_charge": Chem.GetFormalCharge(mol),
            }
            molecules.append(mol)
        stage = "component_construction"
        ca, cb = [gufe.SmallMoleculeComponent(rdkit=mol) for mol in molecules]
        stage = "mapping"
        mapper = lomap.LomapAtomMapper(time=timeout, threed=False, max3d=max3d, shift=False)
        mapping = next(iter(mapper.suggest_mappings(ca, cb)), None)
        if mapping is None:
            result.update(status="no_mapping", stage=stage)
            return result
        stage = "default_score"
        similarity = float(lomap.default_lomap_score(mapping))
        if not np.isfinite(similarity):
            raise ValueError("Nonfinite default LOMAP score")
        result.update(status="scored", similarity=similarity, difficulty=1.-similarity)
        # These are exposed sub-scores, not an assumed reconstruction of the
        # default product. Save the installed default scorer source separately.
        result["rules"] = {}
        for name, function in inspect.getmembers(scorers, inspect.isfunction):
            if name.startswith("_") or not name.endswith("_score") or name == "default_lomap_score":
                continue
            try:
                parameters = inspect.signature(function).parameters
                kwargs = {}
                if "charge_changes_score" in parameters:
                    parameter = inspect.signature(lomap.default_lomap_score).parameters["charge_changes_score"]
                    if parameter.default is inspect.Parameter.empty:
                        raise ValueError("Unknown default charge-change score")
                    kwargs["charge_changes_score"] = parameter.default
                score = float(function(mapping, **kwargs))
                if not np.isfinite(score):
                    raise ValueError("Nonfinite rule score")
                result["rules"][name] = {"value": score}
            except Exception as exc:
                result["rules"][name] = {"error": f"{type(exc).__name__}: {exc}"}
        historical = case.get("historical_difficulty")
        if historical is not None:
            result["difference_from_historical"] = result["difficulty"] - historical
    except Exception as exc:
        result.update(status="error", stage=stage, error=f"{type(exc).__name__}: {exc}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--raw_csv", help="Optional: read header only to inventory available metadata")
    parser.add_argument("--per_group", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mcs_timeout", type=int, default=5)
    parser.add_argument("--max3d", type=float, default=1.0,
                        help="Diagnostic pruning limit: 1 reproduces baseline; 0 disables geometric trimming")
    args = parser.parse_args()
    if args.per_group < 1 or args.mcs_timeout < 1:
        parser.error("per_group and mcs_timeout must be positive")
    if not np.isfinite(args.max3d) or args.max3d < 0:
        parser.error("max3d must be finite and nonnegative")
    import gufe
    import lomap
    from lomap import mcs
    from lomap.gufe_bindings import scorers
    # Capture the code actually installed on the cluster, not an assumed
    # package-version API. An unsupported installation should fail visibly.
    source = inspect.getsource(lomap.default_lomap_score)
    rule_source = inspect.getsource(scorers)
    frame = pd.read_csv(args.predictions)
    artifacts.pair_keys(frame)
    cases, counts = select_cases(frame, args.per_group, args.seed)
    request = {
        "audit_version": AUDIT_VERSION, "predictions_sha256": artifacts.file_hash(args.predictions),
        "seed": args.seed, "per_group": args.per_group, "mcs_timeout": args.mcs_timeout,
        "max3d": args.max3d, "threed": False, "shift": False,
        "rdkit_version": rdBase.rdkitVersion, "default_scorer_source": source,
        "rule_source": rule_source, "mapper_source": inspect.getsource(lomap.LomapAtomMapper),
        "mcs_source": inspect.getsource(mcs.MCS),
        "gufe_version": getattr(gufe, "__version__", None),
        "lomap_version": getattr(lomap, "__version__", None),
        "stratum_counts": counts, "cases": cases,
    }
    if args.raw_csv:
        request["raw_csv_columns"] = pd.read_csv(args.raw_csv, nrows=0).columns.tolist()
    os.makedirs(args.output_dir, exist_ok=True)
    manifest = os.path.join(args.output_dir, "audit_manifest.json")
    output = os.path.join(args.output_dir, "cases.jsonl")
    if os.path.exists(manifest):
        with open(manifest) as stream:
            if json.load(stream) != request:
                raise ValueError("Audit inputs/settings changed; use a fresh output directory.")
    else:
        if os.path.exists(output):
            raise ValueError("Audit results have no manifest; use a fresh output directory.")
        with open(manifest, "w") as stream:
            json.dump(request, stream, indent=2, allow_nan=False)
    records = []
    if os.path.exists(output):
        with open(output) as stream:
            for line in stream:
                records.append(json.loads(line))
    completed = {record["id"] for record in records}
    for case in cases:
        if case["id"] in completed:
            continue
        result = audit_case(case, lomap, gufe, scorers, args.mcs_timeout, args.max3d)
        records.append(result)
        # The audit is bounded and small. Replace the whole JSONL atomically
        # so a killed write cannot corrupt the last completed record.
        temporary = output + ".tmp"
        with open(temporary, "w") as stream:
            for record in records:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
        os.replace(temporary, output)
        print(case["id"], case["group"], result["status"], flush=True)
    print(f"Audit saved in {args.output_dir}. Diagnostic sample only; no performance estimates.")


if __name__ == "__main__":
    main()
