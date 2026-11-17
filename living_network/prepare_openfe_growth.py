"""Build OpenFE/OpenMM transformations for the growth validation panel; runs no MD."""
import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import pandas as pd
from rdkit import Chem

TEMPERATURE_KELVIN = 300.0
ION_MOLAR = 0.07
MAPPER = dict(time=20, threed=True, max3d=1.0, element_change=True, shift=False)
SMOKE_PS = dict(equilibration=10, production=50, checkpoint=10)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mapping_stats(mol_a, mol_b, atom_map):
    """Summarize an A-index to B-index atom map for two explicit-hydrogen molecules."""
    heavy = lambda mol, i: mol.GetAtomWithIdx(i).GetAtomicNum() > 1
    pairs = [(a, b) for a, b in atom_map.items() if heavy(mol_a, a) and heavy(mol_b, b)]
    count = lambda mol: sum(atom.GetAtomicNum() > 1 for atom in mol.GetAtoms())
    return dict(
        mapped_atoms=len(atom_map), mapped_heavy=len(pairs),
        unique_heavy_a=count(mol_a) - len(pairs), unique_heavy_b=count(mol_b) - len(pairs),
        element_changes=sum(mol_a.GetAtomWithIdx(a).GetAtomicNum() != mol_b.GetAtomWithIdx(b).GetAtomicNum()
                            for a, b in atom_map.items()),
    )


def smoke_edges(edges, flagged):
    """One unflagged reference edge and the first edge touching a pose flagged for relaxation."""
    clean = edges[edges.anchor_kind.eq("reference") & ~edges.ligand_a.isin(flagged) & ~edges.ligand_b.isin(flagged)]
    risky = edges[edges.ligand_a.isin(flagged[:1]) | edges.ligand_b.isin(flagged[:1])]
    return list(dict.fromkeys(clean.edge_name.head(1).tolist() + risky.edge_name.head(1).tolist()))


def protocol_settings(kind):
    from openfe.protocols.openmm_rfe import RelativeHybridTopologyProtocol
    from openff.units import unit
    settings = RelativeHybridTopologyProtocol.default_settings()
    settings.protocol_repeats = 1
    settings.thermo_settings.temperature = TEMPERATURE_KELVIN * unit.kelvin
    settings.engine_settings.compute_platform = "CUDA"
    if kind == "smoke":
        settings.simulation_settings.equilibration_length = SMOKE_PS["equilibration"] * unit.picosecond
        settings.simulation_settings.production_length = SMOKE_PS["production"] * unit.picosecond
        settings.simulation_settings.real_time_analysis_interval = None
        settings.output_settings.checkpoint_interval = SMOKE_PS["checkpoint"] * unit.picosecond
    return RelativeHybridTopologyProtocol(settings)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle_dir", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--kind", required=True, choices=["smoke", "openfe_default"])
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("Use a fresh output directory")
    bundle = json.loads((args.bundle_dir / "bundle_manifest.json").read_text())
    for name, expected in bundle["outputs"].items():
        if sha(args.bundle_dir / name) != expected:
            raise ValueError(f"Changed bundle artifact: {name}")

    from openfe import (ChemicalSystem, ProteinComponent, SmallMoleculeComponent,
                        SolventComponent, Transformation)
    from openfe.protocols.openmm_utils.charge_generation import bulk_assign_partial_charges
    from openfe.setup import LomapAtomMapper
    from openfe.setup.atom_mapping.lomap_scorers import default_lomap_score
    from openff.units import unit

    edges = pd.read_csv(args.bundle_dir / "panel_edges.csv", keep_default_na=False)
    validation = pd.read_csv(args.bundle_dir / "structure_validation.csv")
    mols = [m for m in Chem.SDMolSupplier(str(args.bundle_dir / "ligands.sdf"), removeHs=False)]
    if any(m is None for m in mols) or [m.GetProp("_Name") for m in mols] != validation.ligand_id.tolist():
        raise ValueError("Ligand file does not match validation table")
    if any(Chem.GetFormalCharge(m) for m in mols):
        raise ValueError("Charged ligands need an explicit net-charge-change treatment")
    components = bulk_assign_partial_charges(
        molecules=[SmallMoleculeComponent(m, name=m.GetProp("_Name")) for m in mols],
        overwrite=False, method="am1bcc", toolkit_backend="ambertools",
        generate_n_conformers=None, nagl_model=None, processors=1)
    ligand = {c.name: c for c in components}
    if set(ligand) != set(validation.ligand_id):
        raise ValueError("Charge assignment changed ligand identities")
    protein = ProteinComponent.from_pdb_file(str(args.bundle_dir / "receptor.pdb"), name="Mpro-x11498-dimer")
    solvent = SolventComponent(ion_concentration=ION_MOLAR * unit.molar)
    mapper = LomapAtomMapper(**MAPPER)
    protocol = protocol_settings(args.kind)

    args.output_dir.mkdir(parents=True)
    (args.output_dir / "transformations").mkdir()
    (args.output_dir / "ligands_charged.sdf").write_text("".join(ligand[i].to_sdf() for i in validation.ligand_id))
    rows, files = [], {}
    for edge in edges.itertuples():
        a, b = ligand[edge.ligand_a], ligand[edge.ligand_b]
        row = dict(edge_name=edge.edge_name, new_ligand=edge.new_ligand, anchor_kind=edge.anchor_kind,
                   ligand_a=edge.ligand_a, ligand_b=edge.ligand_b)
        found = list(mapper.suggest_mappings(a, b))
        if not found:
            rows.append(dict(row, status="no_mapping"))
            continue
        mapping = found[0]
        row.update(status="prepared", lomap_score=float(default_lomap_score(mapping)),
                   **mapping_stats(a.to_rdkit(), b.to_rdkit(), mapping.componentA_to_componentB))
        for leg in ("complex", "solvent"):
            extra = {"protein": protein} if leg == "complex" else {}
            states = [ChemicalSystem({"ligand": x, "solvent": solvent, **extra}) for x in (a, b)]
            if hasattr(protocol, "validate"):
                protocol.validate(stateA=states[0], stateB=states[1], mapping=mapping)
            transformation = Transformation(stateA=states[0], stateB=states[1], mapping=mapping,
                                            protocol=protocol, name=f"{edge.edge_name}__{leg}")
            path = args.output_dir / "transformations" / f"{edge.edge_name}__{leg}.json"
            transformation.to_json(path) if hasattr(transformation, "to_json") else transformation.dump(path)
            files[f"transformations/{path.name}"] = sha(path)
            row[f"{leg}_transformation"] = f"transformations/{path.name}"
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(args.output_dir / "mappings.csv", index=False)
    flagged = validation.sort_values("min_heavy_distance_to_receptor").ligand_id[
        validation.sort_values("min_heavy_distance_to_receptor").requires_relaxation_check].tolist()
    prepared = edges[edges.edge_name.isin(table.edge_name[table.status.eq("prepared")])]
    tasks = [f"transformations/{name}__{leg}.json" for name in smoke_edges(prepared, flagged)
             for leg in ("complex", "solvent")]
    (args.output_dir / "smoke_tasks.txt").write_text("\n".join(tasks) + "\n")
    report = {
        "version": "growth-openfe-setup-v1", "kind": args.kind,
        "status": "transformations_written_not_simulation_tested",
        "bundle_manifest_sha256": sha(args.bundle_dir / "bundle_manifest.json"),
        "packages": {name: version(name) for name in
                     ("openfe", "gufe", "openmm", "lomap2", "kartograf", "openff-toolkit", "openmmforcefields", "rdkit")
                     if _installed(name)},
        "protocol": "openfe RelativeHybridTopologyProtocol defaults except the listed changes",
        "changes": {"protocol_repeats": 1, "temperature_kelvin": TEMPERATURE_KELVIN,
                    "compute_platform": "CUDA", "ion_concentration_molar": ION_MOLAR,
                    "smoke_lengths_ps": SMOKE_PS if args.kind == "smoke" else None},
        "partial_charges": "am1bcc (ambertools) on the supplied poses",
        "mapper": {"LomapAtomMapper": MAPPER, "score": "default_lomap_score"},
        "edges": len(table), "prepared_edges": int(table.status.eq("prepared").sum()),
        "unmapped_edges": table.edge_name[table.status.ne("prepared")].tolist(),
        "smoke_tasks": tasks, "script_sha256": sha(Path(__file__)),
        "outputs": {**files, **{name: sha(args.output_dir / name)
                                for name in ("mappings.csv", "ligands_charged.sdf", "smoke_tasks.txt")}},
        "limitations": [
            "Not protocol-compatible with the historical perses/Folding@home results; do not combine free energies",
            "openfe_default lengths are OpenFE defaults, not a frozen experimental sampling budget",
            "Each JSON is one independent repeat; repeats must be separate runs of the same file",
            "Smoke runs test setup and stability only and must not be analysed as free-energy outcomes",
        ],
    }
    (args.output_dir / "openfe_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(table.drop(columns=[c for c in table if c.endswith("_transformation")]).to_string())


def _installed(name):
    try:
        version(name)
        return True
    except Exception:
        return False


if __name__ == "__main__":
    main()
