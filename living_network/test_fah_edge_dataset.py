import csv
import json
from pathlib import Path
import tempfile
import unittest

from fah_edge_dataset import build, phase_values


class FahExtractionTests(unittest.TestCase):
    def test_proxy_requires_valid_count_and_stderr(self):
        self.assertEqual(phase_values({
            "delta_f": {"stderr": 0.5}, "num_work_pairs": 16,
        })["stderr_sqrt_work_pairs_proxy"], 2)
        for n, se in [(0, 1), (None, 1), (4, None), (4, -1), (4, float("nan"))]:
            value = phase_values({"delta_f": {"stderr": se}, "num_work_pairs": n})
            self.assertIsNone(value["stderr_sqrt_work_pairs_proxy"])
            self.assertFalse(value["proxy_valid"])

    def test_identity_missing_endpoint_and_source_preservation(self):
        # Same SMILES with distinct IDs must remain separate observations.
        compounds = [{
            "metadata": {"compound_id": cid, "smiles": "CC"},
            "microstates": [{
                "microstate": {"microstate_id": cid + "-m", "smiles": "CC"},
                "first_pass_free_energy": {"point": 1, "stderr": se},
            }],
        } for cid, se in [("A", 0.2), ("B", 0.8)]]
        transformation = {
            "transformation": {
                "run_id": 5,
                "initial_microstate": {"compound_id": "A", "microstate_id": "A-m"},
                "final_microstate": {"compound_id": "B", "microstate_id": "missing"},
            },
            "reliable_transformation": False,
            "binding_free_energy": {"point": 3, "stderr": 0.7},
            "complex_phase": {
                "free_energy": {"delta_f": {"stderr": 0.5}, "num_work_pairs": 16},
                "gens": [{"gen": 2, "works": [{"clone": 9, "forward": 1, "reverse": 2}]}],
            },
        }
        data = {"as_of": "date", "series": {
            "metadata": {"name": "fixture"},
            "compounds": compounds, "transformations": [transformation],
        }}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "analysis.json"
            source.write_text(json.dumps(data))
            output = root / "tables"
            manifest = build([source], output)
            self.assertEqual(manifest["row_counts"], {
                "compounds": 2, "microstates": 2, "edges": 1, "generations": 1,
            })
            self.assertEqual(manifest["sources"][0]["unresolved_edges"], 1)
            with open(output / "edges.csv") as handle:
                edge = next(csv.DictReader(handle))
            self.assertEqual(edge["final_resolved"], "False")
            self.assertEqual(edge["final_microstate_smiles"], "")
            self.assertEqual(edge["binding_free_energy_stderr"], "0.7")
            self.assertEqual(edge["complex_phase_stderr_sqrt_work_pairs_proxy"], "2.0")
            self.assertEqual(edge["solvent_phase_stderr_sqrt_work_pairs_proxy"], "")
            self.assertEqual(edge["complex_phase_raw_work_pair_count"], "1")
            with self.assertRaises(FileExistsError):
                build([source], output)
            # Repeated IDs with different protonation states are not interchangeable.
            import copy
            duplicate = copy.deepcopy(compounds[0]["microstates"][0])
            duplicate["microstate"]["smiles"] = "C[NH3+]"
            compounds[0]["microstates"].append(duplicate)
            source.write_text(json.dumps(data))
            ambiguous = root / "ambiguous"
            report = build([source], ambiguous)
            self.assertEqual(report["sources"][0]["microstates"], 3)
            self.assertEqual(report["sources"][0]["duplicate_microstate_keys"], 1)
            self.assertEqual(report["sources"][0]["ambiguous_microstate_keys"], 1)
            with open(ambiguous / "microstates.csv") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([r["source_microstate_index"] for r in rows], ["0", "1", "0"])
            with open(ambiguous / "edges.csv") as handle:
                edge = next(csv.DictReader(handle))
            self.assertEqual(edge["initial_resolved"], "False")
            self.assertEqual(edge["initial_resolution_status"], "ambiguous_structure")
            self.assertEqual(edge["initial_microstate_smiles"], "")
            self.assertEqual(edge["initial_compound_smiles"], "CC")
            self.assertEqual(edge["initial_structure_candidate_count"], "2")
            # Identical structures can resolve without discarding either source record.
            duplicate["microstate"]["smiles"] = "CC"
            source.write_text(json.dumps(data))
            repeated = root / "repeated"
            report = build([source], repeated)
            self.assertEqual(report["sources"][0]["ambiguous_microstate_keys"], 0)
            with open(repeated / "edges.csv") as handle:
                edge = next(csv.DictReader(handle))
            self.assertEqual(edge["initial_resolved"], "True")
            self.assertEqual(edge["initial_microstate_record_count"], "2")
            with self.assertRaisesRegex(ValueError, "Identical source"):
                build([source, source], root / "duplicate")
            self.assertFalse((root / "duplicate").exists())
            data["series"]["compounds"].append(compounds[0])
            source.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "Duplicate compound"):
                build([source], root / "invalid")
            self.assertFalse((root / "invalid").exists())


if __name__ == "__main__":
    unittest.main()
