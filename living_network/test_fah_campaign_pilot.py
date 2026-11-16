import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from rdkit import rdBase

from prepare_fah_campaign_pilot import make_task, verify_parent_subset
from rbfe_artifacts import file_hash
from run_fah_campaign_pilot import inputs_for
import prepare_fah_campaign_pilot as preparation
import run_fah_campaign_pilot as evaluation
from sklearn.linear_model import LinearRegression


class CampaignPilotTests(unittest.TestCase):
    def setUp(self):
        self.parent = pd.DataFrame({
            "edge_id": [f"e{i}" for i in range(6)], "target": np.ones(6),
            "source_sha256": ["source"] * 6, "smi_a": ["CCO"] * 6, "smi_b": ["CCN"] * 6,
            "split": ["train", "train", "val", "val", "test", "test"],
            "group_a": ["ref"] * 6, "group_b": list("abcdef"),
            "reference_group": ["ref"] * 6, "variable_group": list("abcdef"),
            "reference_smiles": ["CCO"] * 6,
            "resample_group_a": list("abcdef"), "resample_group_b": list("abcdef"),
        })
        self.generations = pd.DataFrame([
            {"edge_id": edge, "source_sha256": "source", "phase": "complex_phase",
             "generation": gen, "delta_f_stderr": gen + 1., "num_work_pairs": 9.,
             "raw_work_pair_count": 10.}
            for edge in self.parent.edge_id for gen in range(6)])

    def test_six_generations_are_eligible_and_only_zero_one_define_values(self):
        task, report = make_task(self.parent, self.generations, "complex_phase", "strict_network")
        self.assertEqual(len(task), 6)
        np.testing.assert_array_equal(task.pilot_proxy, np.full(6, 3.))
        np.testing.assert_array_equal(task.target, np.full(6, 6.))
        self.assertEqual(task.split.tolist(), self.parent.split.tolist())
        changed = self.generations.copy()
        changed.loc[changed.generation.gt(1), "delta_f_stderr"] = 99999.
        second, _ = make_task(self.parent, changed, "complex_phase", "strict_network")
        pd.testing.assert_frame_equal(task, second)
        self.assertFalse(report["runnable"])  # Synthetic cohorts remain below the fixed gate.

    def test_late_labels_do_not_change_pilot_inputs(self):
        first, _ = make_task(self.parent, self.generations, "complex_phase", "strict_network")
        changed = self.generations.copy()
        changed.loc[changed.generation.eq(1), "delta_f_stderr"] *= 10
        second, _ = make_task(self.parent, changed, "complex_phase", "strict_network")
        x = np.arange(150).reshape(6, 25)
        for mode in ("structure", "pilot_only", "structure_plus_pilot"):
            np.testing.assert_array_equal(inputs_for(mode, x, first), inputs_for(mode, x, second))
        self.assertEqual(inputs_for("pilot_only", x, first).shape, (6, 1))
        self.assertEqual(inputs_for("structure_plus_pilot", x, first).shape, (6, 26))
        self.assertTrue((second.target == first.target * 10).all())

    def test_missing_generation_and_invalid_counts_excluded(self):
        changed = self.generations[
            ~(self.generations.edge_id.eq("e0") & self.generations.generation.eq(1))].copy()
        changed.loc[changed.edge_id.eq("e1") & changed.generation.eq(0), "num_work_pairs"] = 0.
        task, info = make_task(self.parent, changed, "complex_phase", "strict_network")
        self.assertEqual(set(task.edge_id), {"e2", "e3", "e4", "e5"})
        self.assertEqual(info["missing_gen0_or_gen1"], 1)
        self.assertEqual(info["invalid_gen01_values"], 1)

    def test_duplicate_and_wrong_source_rejected(self):
        with self.assertRaisesRegex(ValueError, "Repeated generation"):
            make_task(self.parent, pd.concat([self.generations, self.generations.iloc[:1]]),
                      "complex_phase", "strict_network")
        changed = self.generations.copy()
        changed.loc[0, "source_sha256"] = "wrong"
        with self.assertRaisesRegex(ValueError, "source identity"):
            make_task(self.parent, changed, "complex_phase", "strict_network")

    def test_parent_identity_required_for_graph_reuse(self):
        task, _ = make_task(self.parent, self.generations, "complex_phase", "strict_network")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.parent.to_csv(root / "edge.csv", index=False)
            manifest = {"version": 2, "rdkit": rdBase.rdkitVersion,
                        "tables": {"edge.csv": file_hash(root / "edge.csv")}}
            path = root / "benchmark.json"
            path.write_text(json.dumps(manifest))
            digest = file_hash(path)
            self.assertEqual(verify_parent_subset(task, root, digest), digest)
            for column in ("split", "smi_b", "resample_group_a"):
                changed = task.copy()
                changed.loc[0, column] = "wrong"
                with self.assertRaisesRegex(ValueError, "Frozen identity"):
                    verify_parent_subset(changed, root, digest)
            with self.assertRaisesRegex(ValueError, "hash differs"):
                verify_parent_subset(task, root, "wrong")

    def test_preparation_and_baseline_cli_integration(self):
        parent = pd.concat([self.parent] * 60, ignore_index=True)
        parent["edge_id"] = [f"edge{i}" for i in range(len(parent))]
        parent["variable_group"] = [f"g{i % 20}" for i in range(len(parent))]
        parent["group_b"] = parent.variable_group
        parent["series_name"] = "synthetic"
        parent["smi_b"] = ["C" * (i % 20 + 1) for i in range(len(parent))]
        rows = [
            dict(edge_id=row.edge_id, source_sha256="source", phase=phase, generation=gen,
                 delta_f_stderr=(index % 20 + 1) * (gen + 1) / 10,
                 num_work_pairs=9, raw_work_pair_count=10)
            for index, row in enumerate(parent.itertuples())
            for phase in ("complex_phase", "solvent_phase") for gen in range(6)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base, old, new, results = [root / name for name in ("parent", "old", "new", "results")]
            base.mkdir()
            old.mkdir()
            gens = root / "generations.csv"
            pd.DataFrame(rows).to_csv(gens, index=False)
            source = root / "manifest.json"
            source.write_text(json.dumps({"tables": {"generations.csv": file_hash(gens)}}))
            parent.to_csv(base / "edge.csv", index=False)
            manifest = {"version": 2, "setting": "strict_network", "rdkit": rdBase.rdkitVersion,
                        "tables": {"edge.csv": file_hash(base / "edge.csv")},
                        "source_manifest_sha256": file_hash(source)}
            (base / "benchmark.json").write_text(json.dumps(manifest))
            with patch("sys.argv", ["prepare", "--parent_benchmark_dir", str(base),
                                   "--generations", str(gens), "--manifest", str(source),
                                   "--output_dir", str(new)]):
                preparation.main()
            report = json.loads((new / "coverage.json").read_text())
            self.assertTrue(report["task_status"]["pilot_complex"]["runnable"])
            pairs = sorted({tuple(sorted((a, b))) for a, b in zip(parent.smi_a, parent.smi_b)})
            np.savez(old / "features.npz", pairs=np.array(pairs),
                     features=np.array([[len(a) + len(b)] * 25 for a, b in pairs]))
            (old / "features.json").write_text(json.dumps({"sha256": file_hash(old / "features.npz")}))
            (old / "request.json").write_text(json.dumps({
                "benchmark_sha256": file_hash(base / "benchmark.json"),
                "lomap_version": evaluation.LOMAP_SCORING_VERSION,
                "descriptor_sha256": file_hash(Path(evaluation.__file__).with_name("descriptor_rbfe.py"))}))
            test = parent[parent.split.eq("test")].copy()
            test["lomap"] = np.linspace(0, 1, len(test))
            test.to_csv(old / "edge_predictions.csv", index=False)

            def tiny_train(x, frame):
                train = frame.split.eq("train").values
                model = LinearRegression().fit(x[train], np.log(frame.target.values[train]))
                return ("synthetic_linear", model), []

            real_bootstrap = evaluation.bootstrap
            with patch.object(evaluation, "choose_model", side_effect=tiny_train), patch.object(
                evaluation, "bootstrap",
                side_effect=lambda f, c, b: real_bootstrap(f, c, b, repetitions=25)
            ), patch("sys.argv", [
                "evaluate", "--benchmark_dir", str(new), "--parent_benchmark_dir", str(base),
                "--parent_results_dir", str(old), "--output_dir", str(results), "--threads", "1"
            ]):
                evaluation.main()
            summary = json.loads((results / "results.json").read_text())
            self.assertEqual(summary["pilot_complex"]["test_rows"], 120)
            saved = pd.read_csv(results / "pilot_complex_predictions.csv")
            self.assertEqual(set(saved.edge_id), set(test.edge_id))
            self.assertIn("pilot_only", saved)
            self.assertTrue(np.isfinite(saved[["pilot_only", "structure_plus_pilot"]]).all().all())


if __name__ == "__main__":
    unittest.main()
