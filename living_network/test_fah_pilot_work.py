import copy
import csv
import json
import math
from pathlib import Path
import tempfile
import unittest

from audit_fah_pilot_work import audit, diagnostic_bar_root, summarize_generation
from fah_edge_dataset import digest


def generation(gen):
    return {
        "gen": gen, "works": [{"clone": i, "forward": 5., "reverse": -5.} for i in range(3)],
        "free_energy": {"delta_f": {"point": 5., "stderr": .1},
                        "num_work_pairs": 3, "bar_overlap": .5},
    }


class PilotWorkAuditTests(unittest.TestCase):
    def test_known_equal_count_solution(self):
        result = diagnostic_bar_root([5.] * 3, [-5.] * 3)
        self.assertAlmostEqual(result["diagnostic_delta_f"], 5., places=8)
        self.assertAlmostEqual(result["diagnostic_forward_log_mean_acceptance"], -math.log(2), places=8)
        self.assertAlmostEqual(result["diagnostic_log_balance_at_root"], 0., places=8)

    def test_extreme_dissipation_avoids_underflow_plateau(self):
        result = diagnostic_bar_root([1007.] * 3, [993.] * 3)
        self.assertAlmostEqual(result["diagnostic_delta_f"], 7., places=8)
        self.assertAlmostEqual(result["diagnostic_forward_log_mean_acceptance"], -1000., places=8)

    def test_inverted_work_gap_avoids_near_one_plateau(self):
        result = diagnostic_bar_root([-993.] * 3, [-1007.] * 3)
        self.assertAlmostEqual(result["diagnostic_delta_f"], 7., places=8)
        self.assertAlmostEqual(result["diagnostic_forward_log_mean_complement"], -1000., places=8)
        self.assertAlmostEqual(result["diagnostic_reverse_log_mean_complement"], -1000., places=8)

    def test_forward_reverse_swap_and_energy_offset(self):
        forward, reverse = [1., 2., 9.], [-3., -4., -5.]
        root = diagnostic_bar_root(forward, reverse)["diagnostic_delta_f"]
        swapped = diagnostic_bar_root(reverse, forward)["diagnostic_delta_f"]
        shifted = diagnostic_bar_root([x + 100 for x in forward], [x - 100 for x in reverse])
        self.assertAlmostEqual(swapped, -root, places=8)
        self.assertAlmostEqual(shifted["diagnostic_delta_f"], root + 100, places=8)

    def test_counts_clones_and_source_values_are_preserved(self):
        gen = generation(0)
        gen["works"][1]["clone"] = 0
        gen["free_energy"]["num_work_pairs"] = 2
        original = copy.deepcopy(gen)
        row = summarize_generation(gen)
        self.assertEqual(gen, original)
        self.assertEqual(row["duplicate_clone_records"], 1)
        self.assertFalse(row["exported_count_matches_reported"])
        self.assertEqual(row["reported_stderr"], .1)
        self.assertAlmostEqual(row["diagnostic_delta_f"], 5., places=8)

    def test_nonfinite_is_not_silently_filtered(self):
        gen = generation(0)
        gen["works"][0]["forward"] = float("nan")
        row = summarize_generation(gen)
        self.assertEqual(row["forward_nonfinite"], 1)
        self.assertIsNone(row["diagnostic_delta_f"])
        with self.assertRaises(ValueError):
            diagnostic_bar_root([1., float("inf")], [1., 2.])

    def test_source_audit_preserves_generation_labels_and_exports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = {"transformation": {"run_id": 7},
                     "complex_phase": {"gens": [generation(1), generation(0)]},
                     "solvent_phase": {"gens": [generation(0), generation(1)]}}
            source = root / "analysis.json"
            source.write_text(json.dumps({"series": {"transformations": [entry, entry]}}))
            spec = {"source_sha256": digest(source), "source_bytes": source.stat().st_size,
                    "expected_transformations": 2, "case_indices": [0], "control_indices": [1],
                    "phases": ["complex_phase", "solvent_phase"]}
            output = root / "audit"
            report = audit(source, spec, output)
            self.assertEqual(report["selected_transformations"], 2)
            self.assertEqual(report["generation_records"], 8)
            self.assertEqual(report["exported_work_pairs"], 24)
            with open(output / "generation_diagnostics.csv") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["generation"] for row in rows[:2]], ["1", "0"])
            with open(output / "clone_reuse.csv") as handle:
                self.assertTrue(all(row["shared_clone_ids"] == "3" for row in csv.DictReader(handle)))
            for name, expected in report["outputs"].items():
                self.assertEqual(digest(output / name), expected)
            with self.assertRaises(FileExistsError):
                audit(source, spec, output)
            spec["source_sha256"] = "wrong"
            with self.assertRaisesRegex(ValueError, "hash differs"):
                audit(source, spec, root / "bad")


if __name__ == "__main__":
    unittest.main()
