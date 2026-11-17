"""Re-pose a ligand so its mapped heavy atoms sit on an anchor pose (RDKit only)."""
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign
from rdkit.Geometry import Point3D

KEEP_MAX_DEVIATION = 1.0
MAX_MAPPED_RMSD = 0.5
CLASH = 2.2
CONFORMERS = 50
SEED = 20261117
RESTRAINT = dict(max_displacement=0.3, force_constant=100.0)


def heavy_pairs(mol_a, mol_b, atom_map):
    return [(a, b) for a, b in sorted(atom_map.items())
            if mol_a.GetAtomWithIdx(a).GetAtomicNum() > 1 and mol_b.GetAtomWithIdx(b).GetAtomicNum() > 1]


def mapped_deviation(mol_a, mol_b, pairs, conf_id=-1):
    """RMSD and maximum distance between mapped heavy atoms, without alignment."""
    xa = mol_a.GetConformer().GetPositions()[[a for a, _ in pairs]]
    xb = mol_b.GetConformer(conf_id).GetPositions()[[b for _, b in pairs]]
    distance = np.linalg.norm(xa - xb, axis=1)
    return float(np.sqrt((distance ** 2).mean())), float(distance.max())


def stereo_smiles(mol, conf_id=-1):
    copy = Chem.Mol(mol)
    Chem.AssignStereochemistryFrom3D(copy, confId=conf_id, replaceExistingTags=True)
    return Chem.MolToSmiles(Chem.RemoveHs(copy))


def receptor_contacts(mol, receptor_heavy_xyz, conf_id=-1):
    heavy = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1]
    distance = np.linalg.norm(mol.GetConformer(conf_id).GetPositions()[heavy][:, None]
                              - receptor_heavy_xyz[None], axis=2)
    return float(distance.min()), int((distance < CLASH).sum())


def repose(mol_a, mol_b, atom_map, receptor_heavy_xyz):
    """Return (re-posed copy of mol_b or None, report). Atom order and properties are unchanged."""
    pairs = heavy_pairs(mol_a, mol_b, atom_map)
    rmsd, worst = mapped_deviation(mol_a, mol_b, pairs)
    minimum, clashes = receptor_contacts(mol_b, receptor_heavy_xyz)
    report = dict(docked_mapped_rmsd=rmsd, docked_mapped_max_deviation=worst,
                  docked_min_receptor_distance=minimum, docked_receptor_contacts=clashes)
    if worst <= KEEP_MAX_DEVIATION:
        return Chem.Mol(mol_b), dict(report, pose_source="docked", pose_status="ok")
    expected = stereo_smiles(mol_b)
    work = Chem.Mol(mol_b)
    target = mol_a.GetConformer().GetPositions()
    coord_map = {b: Point3D(*target[a]) for a, b in pairs}
    ids = list(AllChem.EmbedMultipleConfs(work, numConfs=CONFORMERS, randomSeed=SEED, coordMap=coord_map,
                                          useRandomCoords=True, clearConfs=True))
    properties = AllChem.MMFFGetMoleculeProperties(work)
    candidates = []
    for conf_id in ids:
        rdMolAlign.AlignMol(work, mol_a, prbCid=conf_id, atomMap=[(b, a) for a, b in pairs])
        if properties is not None:
            field = AllChem.MMFFGetMoleculeForceField(work, properties, confId=conf_id)
            for _, b in pairs:
                field.MMFFAddPositionConstraint(b, RESTRAINT["max_displacement"], RESTRAINT["force_constant"])
            field.Minimize(maxIts=2000)
            rdMolAlign.AlignMol(work, mol_a, prbCid=conf_id, atomMap=[(b, a) for a, b in pairs])
        fit, fit_worst = mapped_deviation(mol_a, work, pairs, conf_id)
        if stereo_smiles(work, conf_id) != expected or fit > MAX_MAPPED_RMSD:
            continue
        minimum, clashes = receptor_contacts(work, receptor_heavy_xyz, conf_id)
        candidates.append((clashes, -minimum, conf_id, fit, fit_worst))
    report.update(pose_source="reposed", conformers_generated=len(ids), conformers_valid=len(candidates),
                  force_field="MMFF94" if properties is not None else "none")
    if not candidates:
        return None, dict(report, pose_status="repose_failed")
    clashes, negative_minimum, conf_id, fit, fit_worst = min(candidates)
    result = Chem.Mol(mol_b)
    moved = work.GetConformer(conf_id).GetPositions()
    for index, xyz in enumerate(moved):
        result.GetConformer().SetAtomPosition(index, Point3D(*xyz))
    heavy = [atom.GetIdx() for atom in mol_b.GetAtoms() if atom.GetAtomicNum() > 1]
    shift = np.linalg.norm(moved[heavy] - mol_b.GetConformer().GetPositions()[heavy], axis=1)
    return result, dict(report, pose_status="ok", reposed_mapped_rmsd=fit, reposed_mapped_max_deviation=fit_worst,
                        reposed_min_receptor_distance=-negative_minimum, reposed_receptor_contacts=clashes,
                        heavy_rmsd_from_docked=float(np.sqrt((shift ** 2).mean())))
