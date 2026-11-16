import copy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from rdkit import rdBase
import torch

from fah_gnn_models import FahGNN, collate_graphs
import prepare_fah_gnn as preparation
from rbfe_artifacts import file_hash
import run_fah_gnn as runner


class FahGNNTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        smiles = ["CCO", "CCN", "CCCN", "CCCCN", "C1CCCCC1", "c1ccccc1", "CO", "CNC", "CC(=O)O"]
        cls.frame = pd.DataFrame({
            "edge_id": [f"e{i}" for i in range(8)], "smi_a": ["CCO"] * 8,
            "smi_b": smiles[1:], "target": np.linspace(.2, 1.6, 8),
            "pilot_proxy": np.linspace(.3, 1.7, 8),
            "split": ["test"] * 8, "source_sha256": ["source"] * 8,
            "resample_group_a": list("abcdefgh"), "resample_group_b": list("abcdefgh"),
            "reference_group": ["reference"] * 8, "variable_group": list("abcdefgh"),
            "series_name": ["synthetic"] * 8,
        })
        cls.graphs = {}
        for architecture in ("hypersphere", "hybrid"):
            cls.graphs[architecture] = dict(preparation.build_one((architecture, key))
                                          for key in preparation.keys_for(cls.frame, architecture))

    def test_models_support_structure_and_pilot_forward_and_gradients(self):
        for architecture in ("hypersphere", "hybrid"):
            for pilot in (False, True):
                model = FahGNN(architecture, pilot, hidden_dim=8, embed_dim=8, num_layers=1, dropout=0.)
                records = runner.graph_records(self.frame.iloc[:3], self.graphs[architecture], architecture)
                graphs = collate_graphs(records, architecture, torch.device("cpu"))
                prediction, cosine = model(graphs, torch.zeros(3) if pilot else None)
                self.assertEqual(tuple(prediction.shape), (3,))
                self.assertTrue(torch.isfinite(prediction).all())
                prediction.square().mean().backward()
                self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                                    for p in model.encoder.parameters()))
                if architecture == "hypersphere":
                    norms = model.encoder(graphs[0]).norm(dim=-1)
                    torch.testing.assert_close(norms, torch.ones_like(norms))
                    self.assertEqual(tuple(cosine.shape), (3,))
                if not pilot:
                    with self.assertRaisesRegex(ValueError, "cannot accept pilot"):
                        model(graphs, torch.zeros(3))

    def test_keyed_graph_failure_and_prediction_order(self):
        for architecture in ("hypersphere", "hybrid"):
            graphs = self.graphs[architecture].copy()
            key = ("CCN" if architecture == "hypersphere" else preparation.graph_key("CCO", "CCN"))
            graphs[key] = None
            shuffled = self.frame.iloc[[3, 0, 2]]
            np.testing.assert_array_equal(runner.graph_validity(shuffled, graphs, architecture),
                                          [True, False, True])
            config = dict(runner.CONFIG, hidden_dim=8, embed_dim=8, num_layers=1, dropout=0.)
            model = runner.make_model(architecture, False, config, torch.device("cpu"))
            stats = runner.training_statistics(self.frame, False)
            a = runner.predict_log(model, self.frame, self.graphs[architecture], architecture,
                                   False, stats, torch.device("cpu"), 3)
            b = runner.predict_log(model, self.frame.iloc[::-1], self.graphs[architecture],
                                   architecture, False, stats, torch.device("cpu"), 3)
            np.testing.assert_allclose(a, b[::-1], rtol=1e-6, atol=1e-6)

    def test_reference_alignment_and_identity_rejection(self):
        reference = self.frame.copy()
        reference["pilot_direct"] = reference.pilot_proxy
        actual = runner.align_reference(self.frame, reference.iloc[::-1])
        self.assertEqual(actual.edge_id.tolist(), self.frame.edge_id.tolist())
        for column, value in [("target", 99.), ("smi_a", "CCC"), ("resample_group_a", "wrong")]:
            bad = reference.copy()
            bad.loc[0, column] = value
            with self.assertRaises(ValueError):
                runner.align_reference(self.frame, bad)
        with self.assertRaisesRegex(ValueError, "Repeated edge"):
            runner.align_reference(self.frame, pd.concat([reference, reference.iloc[:1]]))

    def test_exact_cpu_resume_and_train_only_statistics(self):
        for architecture in ("hypersphere", "hybrid"):
            config = dict(runner.CONFIG, hidden_dim=8, embed_dim=8, num_layers=1,
                          dropout=.1, epochs=3, patience=3, batch_size=4)
            train = self.frame.copy()
            val = self.frame.copy()
            stats = runner.training_statistics(train, True)
            expected = float(np.log(train.target).mean())
            self.assertEqual(stats["log_target_mean"], expected)
            with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
                root = Path(directory)
                full, resumed = root / "full", root / "resume"
                full.mkdir()
                resumed.mkdir()
                state = runner.train_model(train, val, self.graphs[architecture], architecture,
                                           True, full, "identity", torch.device("cpu"), config)
                save = runner.atomic_checkpoint

                def interrupt(path, payload):
                    save(path, payload)
                    raise InterruptedError("synthetic interruption after epoch save")

                with patch.object(runner, "atomic_checkpoint", side_effect=interrupt):
                    with self.assertRaises(InterruptedError):
                        runner.train_model(train, val, self.graphs[architecture], architecture,
                                           True, resumed, "identity", torch.device("cpu"), config)
                restored = runner.train_model(train, val, self.graphs[architecture], architecture,
                                              True, resumed, "identity", torch.device("cpu"), config)
                self.assertEqual(state["history"], restored["history"])
                self.assertEqual(state["best_epoch"], restored["best_epoch"])
                for key in state["best_model"]:
                    torch.testing.assert_close(state["best_model"][key], restored["best_model"][key],
                                               rtol=0, atol=0)
                with self.assertRaisesRegex(ValueError, "Checkpoint identity"):
                    runner.train_model(train, val, self.graphs[architecture], architecture,
                                       True, resumed, "different", torch.device("cpu"), config)

    def test_graph_cache_completion_resume_and_checksum_guard(self):
        class SerialPool:
            def __init__(self, **kwargs):
                pass
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return False
            def map(self, fn, tasks, **kwargs):
                return map(fn, tasks)

        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            benchmark = root / "benchmark"
            benchmark.mkdir()
            table = benchmark / "edge.csv"
            self.frame.iloc[:3].to_csv(table, index=False)
            manifest = {"version": 2, "rdkit": rdBase.rdkitVersion,
                        "report": {"task_status": {"edge": {"runnable": True}}},
                        "tables": {"edge.csv": file_hash(table)}}
            path = benchmark / "benchmark.json"
            path.write_text(json.dumps(manifest))
            cache = root / "cache"
            argv = ["prepare", "--benchmark_dirs", str(benchmark), "--architecture", "hybrid",
                    "--output_dir", str(cache), "--workers", "1"]
            with patch("sys.argv", argv), patch.object(preparation, "ProcessPoolExecutor", SerialPool):
                preparation.main()
                # Simulate an interrupted database with one graph left to compute.
                (cache / "manifest.json").unlink()
                conn = sqlite3.connect(cache / "graphs.sqlite")
                conn.execute("DELETE FROM graphs WHERE key=(SELECT key FROM graphs LIMIT 1)")
                conn.commit()
                conn.close()
                preparation.main()
            graphs, _ = runner.load_graph_cache(cache, self.frame.iloc[:3], "hybrid", file_hash(path))
            self.assertEqual(set(graphs), preparation.keys_for(self.frame.iloc[:3], "hybrid"))
            with open(cache / "graphs.sqlite", "ab") as handle:
                handle.write(b"corrupt")
            with self.assertRaisesRegex(ValueError, "checksum"):
                runner.load_graph_cache(cache, self.frame.iloc[:3], "hybrid", file_hash(path))

    def test_evaluation_preserves_failed_rows_and_matches_comparator(self):
        reference = self.frame.copy()
        reference["structure"] = reference.target * .9
        reference["structure_plus_pilot"] = reference.target * 1.1
        reference["pilot_only"] = reference.target * .95
        reference["pilot_direct"] = reference.pilot_proxy
        reference["training_mean"] = .5
        reference["lomap"] = np.arange(8) / 8
        valid = np.array([False] + [True] * 7)
        # Limit synthetic bootstrap work while preserving the actual implementation.
        real_bootstrap = runner.bootstrap
        with patch.object(runner, "bootstrap",
                          side_effect=lambda frame, cols, comp: real_bootstrap(frame, cols, comp, 250)):
            output, result = runner.evaluate(
                self.frame, valid, self.frame.target.values[valid],
                reference.iloc[::-1], "hypersphere", True)
        self.assertEqual(len(output), 8)
        self.assertTrue(np.isnan(output.gnn.iloc[0]))
        self.assertEqual(result["common_rows"], 7)
        self.assertIn("tree_pilot_only", result["paired_bootstrap_all_graph_valid"])
        self.assertAlmostEqual(result["common_metrics"]["gnn"]["pearson"], 1.)
        self.assertEqual(output.edge_id.tolist(), self.frame.edge_id.tolist())


if __name__ == "__main__":
    unittest.main()
