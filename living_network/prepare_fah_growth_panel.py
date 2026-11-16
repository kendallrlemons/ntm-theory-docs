"""Propose a small validation-only measurement panel; does not prepare or run MD."""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import rdFingerprintGenerator

SOURCE = "2fed73582cebfa4c2543722c371ffbf1badf0d37cc9575eed8ab08a15fa94135"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def key(text):
    return hashlib.sha256(text.encode()).hexdigest()


def panel(initial, arrivals):
    initial = initial[initial.source_sha256.eq(SOURCE)].copy()
    arrivals = arrivals[arrivals.source_sha256.eq(SOURCE) & arrivals.split.eq("val")
                        & arrivals.seed.eq(11)].copy()
    if initial.empty or arrivals.empty or not initial.split.eq("train").all():
        raise ValueError("Missing source training or validation records")
    refs = set(initial.reference_smiles) | set(arrivals.reference_smiles)
    if len(refs) != 1:
        raise ValueError("Expected one source-specific reference")
    reference = next(iter(refs))
    smiles = set(initial.variable_smiles) | set(arrivals.variable_smiles) | refs
    if set(initial.variable_smiles) & set(arrivals.variable_smiles):
        raise ValueError("Incoming structure already in initial network")
    mols = {s: Chem.MolFromSmiles(s) for s in smiles}
    if any(m is None for m in mols.values()):
        raise ValueError("Invalid structure")
    charge = {s: Chem.GetFormalCharge(m) for s, m in mols.items()}
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True)
    fp = {s: generator.GetFingerprint(m) for s, m in mols.items()}
    groups = sorted(set(arrivals.variable_group),
                    key=lambda g: key(f"living-growth-v1|11|{SOURCE}|val|{g}"))
    selected = []
    for group in groups:
        part = arrivals[arrivals.variable_group.eq(group)]
        eligible = sorted({s for s in part.variable_smiles if charge[s] == charge[reference]},
                          key=lambda s: key("growth-panel-v1:" + s))
        if not eligible:
            continue
        s = eligible[0]
        selected.append(dict(smiles=s, group=group, step=int(part.step.iloc[0]),
                             historical_reference_edge_ids=sorted(part[part.variable_smiles.eq(s)].edge_id)))
        if len(selected) == 6:
            break
    if len(selected) != 6:
        raise ValueError("Fewer than six charge-compatible validation groups")
    initial_variables = set(initial.variable_smiles) - refs
    rows = []
    for idx, new in enumerate(selected, 1):
        previous = {x["smiles"] for x in selected if x["step"] < new["step"]}
        available = (initial_variables | previous) - {new["smiles"]}
        compatible = [s for s in available if charge[s] == charge[new["smiles"]]]
        ranked = sorted(compatible, key=lambda s: (
            -DataStructs.TanimotoSimilarity(fp[new["smiles"]], fp[s]), key(s)))
        if len(ranked) < 2:
            raise ValueError("Insufficient compatible prior anchors")
        for rank, anchor in enumerate([reference] + ranked[:2]):
            rows.append(dict(
                source_sha256=SOURCE, series_name=initial.series_name.iloc[0],
                split="val", arrival_seed=11, original_step=new["step"], new_ligand=f"new_{idx:02d}",
                variable_group=new["group"], new_smiles=new["smiles"], anchor_smiles=anchor,
                anchor_kind=("reference" if rank == 0 else
                             "initial" if anchor in initial_variables else "earlier_arrival"),
                alternative_rank=rank, formal_charge=charge[anchor],
                morgan_tanimoto=DataStructs.TanimotoSimilarity(fp[new["smiles"]], fp[anchor]),
                historical_reference_edge_ids=";".join(new["historical_reference_edge_ids"]) if rank == 0 else "",
                outcome_status="historical_reference_exists_protocol_match_required" if rank == 0 else "new_simulation_required",
                mapping_and_pose_status="not_checked",
            ))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--growth_dir", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("Use a fresh output directory")
    manifest = json.loads((args.growth_dir / "growth_manifest.json").read_text())
    for name, expected in manifest["outputs"].items():
        if sha(args.growth_dir / name) != expected:
            raise ValueError(f"Changed growth artifact: {name}")
    initial = pd.read_csv(args.growth_dir / "initial_edges.csv", keep_default_na=False)
    arrivals = pd.read_csv(args.growth_dir / "arrival_schedule.csv", keep_default_na=False)
    result = panel(initial, arrivals)
    args.output_dir.mkdir(parents=True)
    result.to_csv(args.output_dir / "candidate_transformations.csv", index=False)
    report = {
        "version": "growth-validation-panel-v1",
        "status": "structure_screen_only_not_simulation_ready",
        "source_sha256": SOURCE,
        "source_choice": "Small retrospective dimer-neutral campaign; no performance-based source selection",
        "selection": "First six charge-compatible validation groups in frozen seed-11 order; one hash-selected structure per group",
        "initial_network": "All source training edges; the panel is a subset of incoming validation ligands",
        "candidate_rule": "Reference plus two most Morgan-similar same-charge prior nodes; no same-batch/future anchors",
        "fingerprint": "Morgan radius 2, 2048 bits, chirality enabled; screening only, not LOMAP",
        "rdkit": rdBase.rdkitVersion, "script_sha256": sha(Path(__file__)),
        "growth_manifest_sha256": sha(args.growth_dir / "growth_manifest.json"),
        "candidate_csv_sha256": sha(args.output_dir / "candidate_transformations.csv"),
        "new_structures": int(result.new_smiles.nunique()), "candidate_pairs": len(result),
        "missing_alternative_pairs": int(result.alternative_rank.ne(0).sum()),
        "required_checks": [
            "Prepared receptor and ligand poses, protonation/microstate identity and mapping feasibility",
            "One consistent simulation protocol across initial and new edges",
            "Equal total incremental budgets, independent repeats/reference measurements and pilot reliability safeguards",
            "If changing historical protocol, remeasure required initial edges; do not combine incompatible free energies",
        ],
        "interpretation": "Validation development panel; not an untouched confirmation or a simulation cost estimate",
    }
    (args.output_dir / "panel_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
