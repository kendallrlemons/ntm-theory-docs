"""Compare atom mappers on the growth panel edges; uses structures only, no simulation outcomes."""
import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import pandas as pd
from rdkit import Chem

from prepare_openfe_growth import MAPPER, mapping_stats

LOMAP_NO_3D = dict(time=20, threed=False, max3d=0.0, element_change=True, shift=False)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle_dir", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("Use a fresh output directory")
    bundle = json.loads((args.bundle_dir / "bundle_manifest.json").read_text())
    for name, expected in bundle["outputs"].items():
        if sha(args.bundle_dir / name) != expected:
            raise ValueError(f"Changed bundle artifact: {name}")

    from openfe import SmallMoleculeComponent
    from openfe.setup import KartografAtomMapper, LomapAtomMapper
    from openfe.setup.atom_mapping.lomap_scorers import default_lomap_score

    mappers = {
        "lomap_3d": LomapAtomMapper(**MAPPER),
        "lomap_no_3d": LomapAtomMapper(**LOMAP_NO_3D),
        "kartograf": KartografAtomMapper(),
    }
    ligand = {m.GetProp("_Name"): SmallMoleculeComponent(m, name=m.GetProp("_Name"))
              for m in Chem.SDMolSupplier(str(args.bundle_dir / "ligands.sdf"), removeHs=False)}
    edges = pd.read_csv(args.bundle_dir / "panel_edges.csv", keep_default_na=False)
    rows = []
    for edge in edges.itertuples():
        a, b = ligand[edge.ligand_a], ligand[edge.ligand_b]
        for name, mapper in mappers.items():
            row = dict(edge_name=edge.edge_name, new_ligand=edge.new_ligand,
                       anchor_kind=edge.anchor_kind, mapper=name)
            found = list(mapper.suggest_mappings(a, b))
            if found:
                row.update(status="mapped", lomap_score=float(default_lomap_score(found[0])),
                           **mapping_stats(a.to_rdkit(), b.to_rdkit(), found[0].componentA_to_componentB))
            else:
                row.update(status="no_mapping")
            rows.append(row)
    table = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True)
    table.to_csv(args.output_dir / "mapping_comparison.csv", index=False)
    report = {
        "version": "growth-mapping-comparison-v1",
        "bundle_manifest_sha256": sha(args.bundle_dir / "bundle_manifest.json"),
        "mappers": {"lomap_3d": MAPPER, "lomap_no_3d": LOMAP_NO_3D, "kartograf": "KartografAtomMapper defaults"},
        "score": "default_lomap_score applied to every mapper's first mapping",
        "packages": {name: version(name) for name in ("openfe", "gufe", "lomap2", "kartograf", "rdkit")},
        "script_sha256": sha(Path(__file__)),
        "outputs": {"mapping_comparison.csv": sha(args.output_dir / "mapping_comparison.csv")},
        "interpretation": "Structural mapping diagnostics only; not evidence of simulation difficulty",
    }
    (args.output_dir / "mapping_comparison_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    wide = table.pivot(index="edge_name", columns="mapper", values=["mapped_heavy", "lomap_score"])
    print(wide.loc[edges.edge_name].round(3).to_string())


if __name__ == "__main__":
    main()
