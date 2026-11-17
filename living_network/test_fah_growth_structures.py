import unittest

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

from fetch_fah_growth_structures import (canonical, contacts, ligand_checks, protein_records,
                                         receptor_checks)
from prepare_openfe_growth import mapping_stats, smoke_edges


def pose(smiles):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=7)
    return mol


def atom(serial, name, residue, chain, number, xyz, element):
    return (f"ATOM  {serial:5d} {name:<4s} {residue} {chain}{number:4d}    "
            f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00  0.00          {element:>2s}")


class StructureTests(unittest.TestCase):
    def test_ligand_identity_and_stereo(self):
        mol = pose("C[C@H](N)C(=O)O")
        good = ligand_checks(mol, "N[C@@H](C)C(=O)O")
        self.assertTrue(good["identity_ok"] and good["stereo_from_3d_ok"] and good["explicit_h_ok"] and good["is_3d"])
        self.assertEqual((good["unassigned_stereocenters"], good["formal_charge"], good["heavy_atoms"]), (0, 0, 6))
        self.assertFalse(ligand_checks(mol, "C[C@@H](N)C(=O)O")["identity_ok"])

    def test_missing_hydrogens_and_unassigned_stereo_are_reported(self):
        self.assertFalse(ligand_checks(Chem.RemoveHs(pose("CCO")), "CCO")["explicit_h_ok"])
        self.assertEqual(ligand_checks(pose("CC(N)C(=O)O"), "CC(N)C(=O)O")["unassigned_stereocenters"], 1)
        with self.assertRaises(ValueError):
            canonical("not a structure")

    def test_receptor_records_drop_ligand_and_report_dyad(self):
        lines = []
        for chain in "AB":
            lines += [atom(1, "CA", "GLY", chain, 40, (0, 0, 0), "C"),
                      atom(2, "HE2", "HIS", chain, 41, (9, 0, 0), "H"),
                      atom(3, "CA", "HIS", chain, 41, (9, 1, 0), "C")]
        lines += [atom(4, "C1", "MOL", "C", 1, (1, 0, 0), "C"), "HETATM noise"]
        kept, xyz, element, residue = protein_records("\n".join(lines))
        self.assertEqual((len(kept), xyz.shape, residue[0]), (6, (6, 3), "GLYA40"))
        report = receptor_checks(kept, residue)
        self.assertTrue(report["chain_A"]["his41_neutral"] and report["chain_A"]["contiguous"])
        self.assertFalse(report["chain_A"]["cys145_neutral"] or report["dyad_ok"])
        mol = pose("C")
        mol.GetConformer().SetAtomPosition(0, (1.0, 0.0, 0.0))
        minimum, clashes = contacts(mol, xyz, element)
        self.assertAlmostEqual(minimum, 1.0)
        self.assertEqual(clashes, 2)

    def test_mapping_stats_and_smoke_selection(self):
        a, b = pose("CCO"), pose("CCN")
        stats = mapping_stats(a, b, {0: 0, 1: 1, 2: 2, 3: 3})
        self.assertEqual((stats["mapped_heavy"], stats["unique_heavy_a"], stats["element_changes"]), (3, 0, 1))
        edges = pd.DataFrame(dict(edge_name=["r_x", "r_y", "z_y"], anchor_kind=["reference", "reference", "initial"],
                                  ligand_a=["r", "r", "z"], ligand_b=["x", "y", "y"]))
        self.assertEqual(smoke_edges(edges, ["z", "x"]), ["r_y", "z_y"])
        self.assertEqual(smoke_edges(edges, np.array([]).tolist()), ["r_x"])


if __name__ == "__main__":
    unittest.main()
