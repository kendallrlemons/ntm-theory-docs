"""Fetch and validate public setup structures for the growth validation panel.

Builds an input bundle (ligand poses, receptor, edge table); does not run MD.
"""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase

BUCKET = "https://fah-public-data-covid19-moonshot-sprints.s3.us-east-2.amazonaws.com"
CAMPAIGN = "sprint-5-retrospective-x11498-dimer-neutral"
SETUP = "sprint-5-retrospective/setup"
ANALYSIS = f"dashboards/sprint-5-retrospective/{CAMPAIGN}/analysis.json"
DOCKED = f"{SETUP}/docked/{CAMPAIGN}.sdf"
SETUP_JSON = f"{SETUP}/json/{CAMPAIGN}.json"
COMPLEX = SETUP + "/13442/RUNS/RUN{run}/hybrid_complex.pdb"
SHA256 = {
    ANALYSIS: "2fed73582cebfa4c2543722c371ffbf1badf0d37cc9575eed8ab08a15fa94135",
    DOCKED: "6c7e2336d67f106a8f75caa43d4476414c183baf02f067e258f66bf638fc2841",
    SETUP_JSON: "67f59867d957c2bd4dffa5438a43e4d58da227ff882c5f67ad1e2578af8ad8f5",
    COMPLEX.format(run=281): "93cf79f58a169b649e55e3cbe46d5e6e8e90e0972da5566404c48637bccc2885",
    COMPLEX.format(run=323): "7f3a1ce24595d5b1d66ea9edeb100c72596d1334e557689604031e03a6a9940b",
}
CLASH = 2.2
CONTACT = 4.0
RELAX_FLAG = 1.5
PROTEIN_CHAINS = ("A", "B")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fetch(key, cache):
    path = cache / key
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(f"{BUCKET}/{key}", timeout=120) as response:
            path.write_bytes(response.read())
    if key not in SHA256 or sha(path) != SHA256[key]:
        raise ValueError(f"Unpinned or changed public file: {key}")
    return path


def canonical(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid structure: {smiles}")
    return Chem.MolToSmiles(mol)


def heavy_positions(mol):
    keep = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
    return mol.GetConformer().GetPositions()[keep]


def ligand_checks(mol, expected):
    """Identity and completeness checks for one explicit-hydrogen 3D pose."""
    heavy = Chem.RemoveHs(mol)
    from_3d = Chem.Mol(mol)
    Chem.AssignStereochemistryFrom3D(from_3d)
    centers = Chem.FindMolChiralCenters(heavy, includeUnassigned=True, useLegacyImplementation=False)
    return dict(
        identity_ok=Chem.MolToSmiles(heavy) == canonical(expected),
        stereo_from_3d_ok=Chem.MolToSmiles(Chem.RemoveHs(from_3d)) == canonical(expected),
        unassigned_stereocenters=sum(label == "?" for _, label in centers),
        explicit_h_ok=Chem.AddHs(mol).GetNumAtoms() == mol.GetNumAtoms(),
        is_3d=bool(mol.GetNumConformers() and mol.GetConformer().Is3D()),
        formal_charge=Chem.GetFormalCharge(mol), atoms=mol.GetNumAtoms(),
        heavy_atoms=heavy.GetNumAtoms(),
    )


def protein_records(pdb_text):
    """Protein ATOM lines (chains A/B) with coordinates, elements and residue keys."""
    lines = [line for line in pdb_text.splitlines()
             if line.startswith("ATOM") and line[21] in PROTEIN_CHAINS]
    xyz = np.array([[float(line[30:38]), float(line[38:46]), float(line[46:54])] for line in lines])
    element = np.array([line[76:78].strip() for line in lines])
    residue = np.array([f"{line[17:20]}{line[21]}{int(line[22:26])}" for line in lines])
    return lines, xyz, element, residue


def residue_atoms(lines, chain, number):
    return {line[12:16].strip() for line in lines if line[21] == chain and int(line[22:26]) == number}


def receptor_checks(lines, residue):
    report = {"atoms": len(lines)}
    for chain in PROTEIN_CHAINS:
        numbers = sorted({int(line[22:26]) for line in lines if line[21] == chain})
        his41, cys145 = residue_atoms(lines, chain, 41), residue_atoms(lines, chain, 145)
        report[f"chain_{chain}"] = dict(
            residues=len(numbers), first=numbers[0], last=numbers[-1],
            contiguous=numbers == list(range(numbers[0], numbers[-1] + 1)),
            his41_neutral=len(his41 & {"HD1", "HE2"}) == 1, cys145_neutral="HG" in cys145,
            his41_protons=sorted(his41 & {"HD1", "HE2"}),
            his163_protons=sorted(residue_atoms(lines, chain, 163) & {"HD1", "HE2"}),
        )
    report["has_hydrogens"] = any(line[76:78].strip() == "H" for line in lines)
    report["dyad_ok"] = all(report[f"chain_{c}"]["his41_neutral"] and report[f"chain_{c}"]["cys145_neutral"]
                            and report[f"chain_{c}"]["contiguous"] for c in PROTEIN_CHAINS)
    return report


def contacts(mol, xyz, element):
    distance = np.linalg.norm(heavy_positions(mol)[:, None] - xyz[element != "H"][None], axis=2)
    return float(distance.min()), int((distance < CLASH).sum())


def write_receptor(lines, path):
    out = []
    for chain in PROTEIN_CHAINS:
        out += [line for line in lines if line[21] == chain] + ["TER"]
    path.write_text("\n".join(out + ["END"]) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel_dir", required=True, type=Path)
    parser.add_argument("--cache_dir", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("Use a fresh output directory")
    panel_manifest = json.loads((args.panel_dir / "panel_manifest.json").read_text())
    panel_csv = args.panel_dir / "candidate_transformations.csv"
    if sha(panel_csv) != panel_manifest["candidate_csv_sha256"]:
        raise ValueError("Changed panel")
    panel = pd.read_csv(panel_csv, keep_default_na=False)
    if set(panel.source_sha256) != {SHA256[ANALYSIS]}:
        raise ValueError("Panel is not from the pinned campaign")

    analysis = json.loads(fetch(ANALYSIS, args.cache_dir).read_text())["series"]
    setup = json.loads(fetch(SETUP_JSON, args.cache_dir).read_text())
    if setup["metadata"]["name"] != CAMPAIGN or analysis["metadata"]["name"] != CAMPAIGN:
        raise ValueError("Campaign mismatch")
    edge = lambda t: (t["run_id"], t["initial_microstate"]["microstate_id"], t["final_microstate"]["microstate_id"])
    transformations = [edge(t["transformation"]) for t in analysis["transformations"]]
    if sorted(transformations) != sorted(edge(t) for t in setup["transformations"]):
        raise ValueError("Setup and analysis transformations differ")
    microstates = {m["microstate"]["microstate_id"]: canonical(m["microstate"]["smiles"])
                   for c in analysis["compounds"] for m in c["microstates"]}
    used = {mid for _, a, b in transformations for mid in (a, b)}
    by_smiles = {}
    for mid in sorted(used):
        by_smiles.setdefault(microstates[mid], []).append(mid)
    wanted = {canonical(s) for s in set(panel.new_smiles) | set(panel.anchor_smiles)}
    initial_ids = {a for _, a, _ in transformations}
    duplicates = {s: ids for s, ids in by_smiles.items() if s in wanted and len(ids) > 1}
    for s, ids in duplicates.items():
        by_smiles[s] = [mid for mid in ids if mid in initial_ids]
    ambiguous = {s: by_smiles.get(s) for s in wanted if len(by_smiles.get(s, [])) != 1}
    if ambiguous:
        raise ValueError(f"Panel structures without one simulated microstate: {ambiguous}")
    ligand_id = {s: by_smiles[s][0] for s in wanted}

    poses = {}
    for mol in Chem.SDMolSupplier(str(fetch(DOCKED, args.cache_dir)), removeHs=False):
        if mol is None:
            raise ValueError("Unreadable docked record")
        if mol.GetProp("_Name") in poses:
            raise ValueError("Duplicate docked microstate")
        poses[mol.GetProp("_Name")] = mol
    if not used <= set(poses):
        raise ValueError("Simulated microstates missing from the campaign docked file")

    reference = ligand_id[canonical(panel.anchor_smiles[panel.anchor_kind.eq("reference")].iloc[0])]
    reference_runs = []
    for ids in panel.historical_reference_edge_ids[panel.alternative_rank.eq(0)]:
        for edge_id in ids.split(";"):
            run, initial, final = transformations[int(edge_id.split(":")[1])]
            new = panel.new_smiles[panel.historical_reference_edge_ids.eq(ids)].iloc[0]
            if initial != reference or final != ligand_id[canonical(new)]:
                raise ValueError("Historical edge does not match panel structures")
            reference_runs.append(run)
    receptor, frame = None, []
    for run in sorted(reference_runs):
        key = COMPLEX.format(run=run)
        lines, xyz, element, residue = protein_records(fetch(key, args.cache_dir).read_text())
        minimum, clashes = contacts(poses[reference], xyz, element)
        in_frame = clashes == 0 and minimum < CONTACT
        frame.append(dict(run_id=run, reference_min_heavy_distance=round(minimum, 3), in_docking_frame=in_frame))
        if in_frame:
            receptor = dict(run_id=run, key=key, lines=lines, xyz=xyz, element=element, residue=residue)
            break
    if receptor is None:
        raise ValueError("No candidate setup complex is in the docking frame")
    receptor_report = receptor_checks(receptor["lines"], receptor["residue"])
    reference_xyz = heavy_positions(poses[reference])
    near = np.linalg.norm(receptor["xyz"][:, None] - reference_xyz[None], axis=2).min(1) < CONTACT
    receptor_report["reference_site_residues"] = sorted(set(receptor["residue"][near & (receptor["element"] != "H")]))

    role = {ligand_id[canonical(r.new_smiles)]: r.new_ligand for r in panel.itertuples()}
    rows = []
    for smiles, mid in sorted(ligand_id.items(), key=lambda item: item[1]):
        mol = poses[mid]
        minimum, clashes = contacts(mol, receptor["xyz"], receptor["element"])
        rows.append(dict(
            ligand_id=mid, panel_role=role.get(mid, "reference" if mid == reference else "initial"),
            smiles=smiles, **ligand_checks(mol, smiles),
            upstream_clash_score=float(mol.GetProp("clash_score")) if mol.HasProp("clash_score") else np.nan,
            min_heavy_distance_to_receptor=round(minimum, 3), heavy_contacts_below_2p2=clashes,
            requires_relaxation_check=minimum < RELAX_FLAG,
        ))
    ligands = pd.DataFrame(rows)
    ok = (ligands.identity_ok & ligands.stereo_from_3d_ok & ligands.explicit_h_ok & ligands.is_3d
          & ligands.unassigned_stereocenters.eq(0) & ligands.formal_charge.eq(0))
    if not ok.all() or not receptor_report["dyad_ok"] or not receptor_report["has_hydrogens"]:
        raise ValueError("Structure validation failed:\n" + ligands[~ok].to_string())

    edges = panel.assign(
        ligand_a=[ligand_id[canonical(s)] for s in panel.anchor_smiles],
        ligand_b=[ligand_id[canonical(s)] for s in panel.new_smiles])
    edges.insert(0, "edge_name", edges.ligand_a + "__to__" + edges.ligand_b)
    if edges.edge_name.duplicated().any():
        raise ValueError("Duplicate edges")
    edges = edges.drop(columns=["mapping_and_pose_status"])

    args.output_dir.mkdir(parents=True)
    writer = Chem.SDWriter(str(args.output_dir / "ligands.sdf"))
    for mid in ligands.ligand_id:
        writer.write(poses[mid])
    writer.close()
    write_receptor(receptor["lines"], args.output_dir / "receptor.pdb")
    ligands.to_csv(args.output_dir / "structure_validation.csv", index=False)
    edges.to_csv(args.output_dir / "panel_edges.csv", index=False)
    outputs = ["ligands.sdf", "receptor.pdb", "structure_validation.csv", "panel_edges.csv"]
    report = {
        "version": "growth-structures-v1",
        "status": "structures_validated_not_simulation_tested",
        "campaign": CAMPAIGN, "bucket": BUCKET,
        "public_inputs": {key: SHA256[key] for key in [ANALYSIS, DOCKED, SETUP_JSON, receptor["key"]]},
        "panel_manifest_sha256": sha(args.panel_dir / "panel_manifest.json"),
        "panel_csv_sha256": sha(panel_csv),
        "ligand_pose_source": "Campaign setup docked microstates; coordinates and hydrogens unchanged",
        "receptor_source": "Protein chains A/B of the relaxed campaign setup complex; ligand and ions removed",
        "receptor_rule": "Lowest run among the panel's historical reference edges whose setup complex is "
                         "in the docking frame (docked reference in contact without heavy-atom clashes)",
        "receptor_run_id": receptor["run_id"], "receptor_frame_checks": frame, "receptor": receptor_report,
        "thresholds_angstrom": {"clash": CLASH, "contact": CONTACT, "relaxation_flag": RELAX_FLAG},
        "ligands": len(ligands), "edges": len(edges),
        "duplicate_structure_ids_resolved_to_campaign_reference": duplicates,
        "ligands_requiring_relaxation_check": ligands.ligand_id[ligands.requires_relaxation_check].tolist(),
        "rdkit": rdBase.rdkitVersion, "script_sha256": sha(Path(__file__)),
        "outputs": {name: sha(args.output_dir / name) for name in outputs},
        "limitations": [
            "The unrelaxed receptor used for docking is not public; the receptor is one relaxed setup complex",
            "Docked poses can clash with this receptor; minimization/equilibration must be checked per complex",
            "Contact distances are diagnostics; they are not used to select panel ligands or edges",
            "Atom mappings, charges, force fields and simulation stability are not validated here",
        ],
    }
    (args.output_dir / "bundle_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "receptor_run_id", "ligands", "edges",
                                              "ligands_requiring_relaxation_check")}, indent=2))


if __name__ == "__main__":
    main()
