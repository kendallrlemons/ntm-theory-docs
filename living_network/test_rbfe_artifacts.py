import os
import pickle
import sys
import tempfile
import unittest
from collections import Counter
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

import descriptor_rbfe as desc
import descriptor_learning_curve as curve
import audit_lomap as audit
import hybrid_topo_rbfe as hto
import mcs_precompute
import rbfe_artifacts as artifacts
import rescore_lomap as rescore
import label_structure_diagnostic as labels
import compound_uncertainty_model as compound


class GraphArtifactTests(unittest.TestCase):
    def setUp(self):
        self.keys = [("A", "B"), ("C", "D"), ("E", "F")]
        self.graphs = ["g0", None, "g2"]
        self.payload = artifacts.graph_bundle(self.graphs, self.keys, 5)

    def test_alignment_preserves_requested_order_and_none(self):
        requested = pd.DataFrame(
            [self.keys[2], self.keys[0]], columns=["smi_a", "smi_b"],
        ).reset_index(drop=True)
        self.assertEqual(
            artifacts.aligned_graphs(
                self.payload, artifacts.pair_keys(requested), 5,
            ),
            ["g2", "g0"],
        )
        self.assertEqual(
            artifacts.aligned_graphs(self.payload, [self.keys[1]], 5),
            [None],
        )

    def test_reversed_pair_is_not_a_match(self):
        with self.assertRaises(ValueError):
            artifacts.aligned_graphs(self.payload, [("B", "A")], 5)

    def test_missing_key_is_rejected(self):
        with self.assertRaises(ValueError):
            artifacts.aligned_graphs(self.payload, [("X", "Y")], 5)

    def test_duplicate_keys_are_rejected(self):
        with self.assertRaises(ValueError):
            artifacts.graph_bundle(["g0", "g1"], [self.keys[0], self.keys[0]], 5)

    def test_length_mismatch_is_rejected(self):
        payload = dict(self.payload, graphs=self.graphs[:-1])
        with self.assertRaises(ValueError):
            artifacts.validate_graph_bundle(payload, 5)

    def test_legacy_list_is_rejected(self):
        with self.assertRaises(ValueError):
            artifacts.validate_graph_bundle(self.graphs, 5)

    def test_wrong_mcs_timeout_is_rejected(self):
        with self.assertRaises(ValueError):
            artifacts.aligned_graphs(self.payload, self.keys, 2)

    def test_pipeline_loader_aligns_saved_bundle_to_reordered_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            torch.save(self.payload, os.path.join(directory, "graphs_all.pt"))
            requested = pd.DataFrame(
                [self.keys[2], self.keys[0]], columns=[artifacts.A, artifacts.B],
            ).reset_index(drop=True)
            self.assertEqual(
                hto._load_graphs(
                    "test", directory, pair_df=requested, mcs_timeout=5,
                ),
                ["g2", "g0"],
            )


class ProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({
            artifacts.A: ["A", "C", "E"],
            artifacts.B: ["B", "D", "F"],
            artifacts.Y: [1.0, -2.0, 3.0],
            "SE_abs": [1.0, 2.0, 3.0],
            "split": ["train", "val", "test"],
        })

    def test_raw_and_hypersphere_column_identities_match(self):
        renamed = self.frame.rename(columns={
            artifacts.A: "smi_a",
            artifacts.B: "smi_b",
            artifacts.Y: "SE",
        })
        self.assertEqual(
            artifacts.benchmark_identity(self.frame),
            artifacts.benchmark_identity(renamed),
        )

    def test_changed_label_and_magnitude_change_identity(self):
        changed = self.frame.copy()
        changed.loc[0, artifacts.Y] = -1.5
        changed.loc[0, "SE_abs"] = 1.5
        self.assertNotEqual(
            artifacts.benchmark_identity(self.frame),
            artifacts.benchmark_identity(changed),
        )

    def test_changed_split_changes_identity(self):
        changed = self.frame.copy()
        changed.loc[0, "split"] = "val"
        changed.loc[1, "split"] = "train"
        self.assertNotEqual(
            artifacts.benchmark_identity(self.frame),
            artifacts.benchmark_identity(changed),
        )

    def test_missing_checkpoint_benchmark_is_rejected(self):
        expected = artifacts.benchmark_identity(self.frame)
        with self.assertRaises(ValueError):
            artifacts.require_benchmark({}, expected)

    def test_mismatched_checkpoint_benchmark_is_rejected(self):
        expected = artifacts.benchmark_identity(self.frame)
        changed = self.frame.copy()
        changed.loc[0, artifacts.Y] = 1.5
        changed.loc[0, "SE_abs"] = 1.5
        with self.assertRaises(ValueError):
            artifacts.require_benchmark(
                {"benchmark": artifacts.benchmark_identity(changed)}, expected,
            )
        artifacts.require_benchmark({"benchmark": expected}, expected)

    def test_data_cache_request_content_and_manifest_are_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            source = os.path.join(directory, "input.csv")
            cache = os.path.join(directory, "cleaned.pkl")
            with open(source, "wb") as stream:
                stream.write(b"input")
            self.frame.to_pickle(cache)
            request = artifacts.data_request(source, seed=42)
            artifacts.write_cache_manifest(cache, request)
            artifacts.require_cache_manifest(cache, request)

            changed_request = artifacts.data_request(source, seed=43)
            with self.assertRaises(ValueError):
                artifacts.require_cache_manifest(cache, changed_request)

            with open(cache, "ab") as stream:
                stream.write(b"tamper")
            with self.assertRaises(ValueError):
                artifacts.require_cache_manifest(cache, request)

            os.remove(cache + ".manifest.json")
            with self.assertRaises(ValueError):
                artifacts.require_cache_manifest(cache, request)

    def test_phase1_scaffold_cache_reuse_and_request_validation(self):
        pairs = [
            ("C1CC1C", "C1CC1O"),
            ("C1CCC1C", "C1CCC1O"),
            ("C1CCCC1C", "C1CCCC1O"),
            ("C1CCCCC1C", "C1CCCCC1O"),
            ("Cc1ccccc1", "Oc1ccccc1"),
            ("Cc1ccccn1", "Oc1ccccn1"),
        ]
        raw = pd.DataFrame({
            artifacts.A: [pair[0] for pair in pairs],
            artifacts.B: [pair[1] for pair in pairs],
            artifacts.Y: [float(i + 1) for i in range(len(pairs))],
        })
        with tempfile.TemporaryDirectory() as directory:
            source = os.path.join(directory, "input.csv")
            raw.to_csv(source, index=False)
            args = SimpleNamespace(
                input=source, sample_size=0, seed=42,
                split_method="scaffold", val_frac=0.1, test_frac=0.1,
            )
            first = hto.phase1_load(args, directory)
            reused = hto.phase1_load(args, directory)
            pd.testing.assert_frame_equal(first, reused)

            changed_settings = SimpleNamespace(**vars(args))
            changed_settings.seed = 43
            with self.assertRaises(ValueError):
                hto.phase1_load(changed_settings, directory)

            raw.loc[0, artifacts.Y] = 9.0
            raw.to_csv(source, index=False)
            with self.assertRaises(ValueError):
                hto.phase1_load(args, directory)


class ChunkMergeTests(unittest.TestCase):
    def _write_chunks(self, directory, swapped=False):
        keys = [("A", "B"), ("C", "D"), ("E", "F")]
        frame = pd.DataFrame({
            artifacts.A: [key[0] for key in keys],
            artifacts.B: [key[1] for key in keys],
        })
        frame.to_pickle(os.path.join(directory, "cleaned.pkl"))
        first_keys = keys[:2]
        first_graphs = ["g0", "g1"]
        if swapped:
            first_keys = list(reversed(first_keys))
            first_graphs = list(reversed(first_graphs))
        torch.save(
            artifacts.graph_bundle(first_graphs, first_keys, 5),
            os.path.join(directory, "graphs_all_chunk0_of_2.pt"),
        )
        torch.save(
            artifacts.graph_bundle(["g2"], keys[2:], 5),
            os.path.join(directory, "graphs_all_chunk1_of_2.pt"),
        )
        return keys

    def test_merge_chunks_aligns_all_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            keys = self._write_chunks(directory)
            mcs_precompute.merge_chunks(directory, 2)
            merged = torch.load(
                os.path.join(directory, "graphs_all.pt"), weights_only=False,
            )
            self.assertEqual(
                artifacts.aligned_graphs(merged, keys, 5),
                ["g0", "g1", "g2"],
            )

    def test_merge_rejects_swapped_chunk_identities(self):
        with tempfile.TemporaryDirectory() as directory:
            self._write_chunks(directory, swapped=True)
            with self.assertRaises(ValueError):
                mcs_precompute.merge_chunks(directory, 2)

    def test_merge_rejects_legacy_merged_graph(self):
        with tempfile.TemporaryDirectory() as directory:
            self._write_chunks(directory)
            torch.save(
                ["g0", "g1", "g2"],
                os.path.join(directory, "graphs_all.pt"),
            )
            with self.assertRaisesRegex(ValueError, "legacy"):
                mcs_precompute.merge_chunks(directory, 2)


class FakeEstimator:
    fit_records = []
    predict_records = []

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def set_params(self, **kwargs):
        self.__dict__.update(kwargs)
        return self

    def fit(self, X, y):
        type(self).fit_records.append((
            np.array(X, copy=True), np.array(y, copy=True),
        ))
        return self

    def predict(self, X):
        X = np.array(X, copy=True)
        type(self).predict_records.append(X)
        prediction = np.arange(len(X))
        if self.max_iter in (100, 200):
            prediction = prediction[::-1]
        return prediction


class FakeCurveEstimator:
    fit_records = []
    constructor_settings = []

    def __init__(self, **settings):
        self.settings = settings.copy()
        type(self).constructor_settings.append(settings.copy())

    def fit(self, X, y):
        type(self).fit_records.append({
            "X": np.array(X, copy=True),
            "y": np.array(y, copy=True),
        })
        return self

    def predict(self, X):
        return np.arange(len(X), dtype=float) + 1.


class SynchronousPool:
    instances = 0

    def __init__(self, max_workers):
        type(self).instances += 1

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def map(self, function, tasks, chunksize=1):
        return map(function, tasks)


class DescriptorExperimentTests(unittest.TestCase):
    def test_pair_features_are_symmetric_and_invalid_smiles_are_nan(self):
        forward = desc.pair_features(("CCO", "CCN"))
        reverse = desc.pair_features(("CCN", "CCO"))
        self.assertEqual(len(forward), len(desc.FEATURE_NAMES))
        np.testing.assert_allclose(forward, reverse)

        invalid = desc.pair_features(("C1CC", "CCO"))
        self.assertTrue(np.isnan(invalid).all())

    def test_select_rows_caps_only_train_and_preserves_val_test_order(self):
        splits = ["train"] * 8 + ["val"] * 3 + ["test"] * 4
        frame = pd.DataFrame({
            artifacts.A: [f"C{i}" for i in range(len(splits))],
            artifacts.B: [f"N{i}" for i in range(len(splits))],
            "SE_abs": np.arange(1, len(splits) + 1, dtype=float),
            "split": splits,
        })

        selected = desc.select_rows(frame, train_cap=3, seed=42)
        repeated = desc.select_rows(frame, train_cap=3, seed=42)
        self.assertEqual(int((selected.split == "train").sum()), 3)
        self.assertEqual(
            artifacts.pair_keys(selected[selected.split == "val"]),
            artifacts.pair_keys(frame[frame.split == "val"]),
        )
        self.assertEqual(
            artifacts.pair_keys(selected[selected.split == "test"]),
            artifacts.pair_keys(frame[frame.split == "test"]),
        )
        pd.testing.assert_frame_equal(selected, repeated)

        uncapped = desc.select_rows(frame, train_cap=0, seed=42)
        self.assertEqual(len(uncapped), len(frame))
        self.assertEqual(
            artifacts.pair_keys(uncapped[uncapped.split == "train"]),
            artifacts.pair_keys(frame[frame.split == "train"]),
        )

    def test_select_model_uses_only_train_and_validation_and_snapshots_first_tie(self):
        FakeEstimator.fit_records = []
        FakeEstimator.predict_records = []
        X_train = np.array([[10.], [11.], [12.]])
        y_train = np.array([1., 2., 3.])
        X_val = np.array([[20.], [21.], [22.]])
        y_val = np.array([1., 2., 3.])

        with patch.object(desc, "HistGradientBoostingRegressor", FakeEstimator):
            model, history = desc.select_model(
                X_train, y_train, X_val, y_val, seed=42,
            )

        self.assertEqual(model.max_iter, 50)
        self.assertEqual(model.max_leaf_nodes, 15)
        self.assertEqual(len(history), 6)
        self.assertEqual(len(FakeEstimator.fit_records), 6)
        self.assertEqual(len(FakeEstimator.predict_records), 6)
        for X, y in FakeEstimator.fit_records:
            np.testing.assert_array_equal(X, X_train)
            np.testing.assert_array_equal(y, y_train)
        for X in FakeEstimator.predict_records:
            np.testing.assert_array_equal(X, X_val)

    def test_metric_rows_handles_constant_scores_and_tied_targets(self):
        y = np.array([1., 2., 3., 4.])
        rows = desc.metric_rows(
            y,
            {"perfect": y.copy(), "constant": np.full(4, 2.5)},
            {"perfect", "constant"},
        )
        by_name = {row["method"]: row for row in rows}
        self.assertEqual(by_name["perfect"]["spearman_r"], 1.0)
        self.assertEqual(by_name["perfect"]["auroc_hard_top25"], 1.0)
        self.assertEqual(by_name["perfect"]["mse"], 0.0)
        self.assertIsNone(by_name["constant"]["spearman_r"])
        self.assertEqual(by_name["constant"]["mse"], 1.25)
        self.assertEqual(by_name["perfect"]["hard_prevalence"], 0.25)

        tied = desc.metric_rows(
            np.ones(4), {"score": np.arange(4.)}, {"score"},
        )[0]
        self.assertIsNone(tied["auroc_hard_top25"])
        self.assertIsNone(tied["auprc_hard_top25"])

    def test_small_hist_gradient_boosting_search_smoke(self):
        X_train = np.arange(80, dtype=float).reshape(-1, 1)
        y_train = X_train[:, 0].copy()
        X_val = np.arange(10, 70, 5, dtype=float).reshape(-1, 1)
        y_val = X_val[:, 0].copy()
        with threadpool_limits(limits=1):
            model, history = desc.select_model(
                X_train, y_train, X_val, y_val, seed=42,
            )
            prediction = model.predict(X_val)
        self.assertEqual(len(history), 6)
        self.assertEqual(len(prediction), 12)
        self.assertTrue(np.isfinite(prediction).all())

    def test_gnn_predictions_preserve_failed_graph_rows_and_check_benchmark(self):
        import torch

        frame = pd.DataFrame({
            artifacts.A: ["CC", "CO", "CN"],
            artifacts.B: ["CO", "CN", "CC"],
            artifacts.Y: [1., -2., 3.],
            "SE_abs": [1., 2., 3.],
            "split": ["train", "val", "test"],
        })
        benchmark = artifacts.benchmark_identity(frame)
        args = SimpleNamespace(
            hybrid_ckpt="checkpoint.pt", mcs_timeout=5,
            graph_cache_dir="graphs",
        )
        checkpoint = {
            "benchmark": benchmark,
            "args": {
                "mcs_timeout": 5, "hidden_dim": 1, "num_layers": 1,
                "embed_dim": 1, "dropout": 0.,
            },
            "model": {},
        }

        class FakeHybridGNN:
            def __init__(self, *args):
                pass

            def to(self, device):
                return self

            def load_state_dict(self, state):
                pass

            def eval(self):
                return self

            def __call__(self, batch):
                return torch.tensor([7., 9.])

        with (
            patch("torch.load", return_value=checkpoint),
            patch.object(desc.hto, "HybridGNN", FakeHybridGNN),
            patch.object(desc.hto, "_load_graphs",
                         return_value=[object(), None, object()]),
            patch.object(desc.hto, "_collate",
                         return_value=({"x": torch.zeros(2, 1)}, None)),
        ):
            predictions = desc.gnn_predictions(frame, args, benchmark)
        self.assertEqual(predictions[0], 7.)
        self.assertTrue(np.isnan(predictions[1]))
        self.assertEqual(predictions[2], 9.)

        with (
            patch("torch.load", return_value={"benchmark": {}, "args": {}}),
            patch.object(desc.hto, "HybridGNN") as model_constructor,
        ):
            with self.assertRaises(ValueError):
                desc.gnn_predictions(frame, args, benchmark)
        model_constructor.assert_not_called()

    def test_test_only_lomap_progress_caches_finite_and_failed_scores(self):
        pairs = [("CC", "CO"), ("CN", "CC"), ("CO", "CN")]
        indices = np.array([1, 2])
        request = {"mode": "test-only"}
        SynchronousPool.instances = 0

        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(
                output_dir=directory, workers=1, mcs_timeout=5,
            )
            worker = Mock(side_effect=[0.2, np.nan])
            with (
                patch.object(desc, "ProcessPoolExecutor", SynchronousPool),
                patch.object(desc, "lomap_worker", worker),
            ):
                scores = desc.score_lomap(pairs, indices, args, request)
                cached_scores = desc.score_lomap(pairs, indices, args, request)
                with self.assertRaises(ValueError):
                    desc.score_lomap(
                        pairs, indices, args, {"mode": "both"},
                    )

            self.assertTrue(np.isnan(scores[0]))
            self.assertEqual(scores[1], 0.2)
            self.assertTrue(np.isnan(scores[2]))
            np.testing.assert_equal(cached_scores, scores)
            self.assertEqual(worker.call_count, 2)
            self.assertEqual(SynchronousPool.instances, 1)
            with open(os.path.join(directory, "lomap_progress.pkl"), "rb") as stream:
                progress = pickle.load(stream)
            np.testing.assert_array_equal(progress["done"], [False, True, True])

    def test_lomap_progress_resumes_after_worker_exception(self):
        pairs = [("CC", "CO"), ("CN", "CC")]
        indices = np.array([0, 1])
        request = {"mode": "both"}
        SynchronousPool.instances = 0

        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(
                output_dir=directory, workers=1, mcs_timeout=5,
            )
            interrupted_worker = Mock(
                side_effect=[0.2, RuntimeError("synthetic interruption")],
            )
            with (
                patch.object(desc, "ProcessPoolExecutor", SynchronousPool),
                patch.object(desc, "lomap_worker", interrupted_worker),
            ):
                with self.assertRaisesRegex(RuntimeError, "synthetic interruption"):
                    desc.score_lomap(pairs, indices, args, request)

            with open(os.path.join(directory, "lomap_progress.pkl"), "rb") as stream:
                progress = pickle.load(stream)
            np.testing.assert_array_equal(progress["done"], [True, False])
            self.assertEqual(progress["scores"][0], 0.2)

            resumed_worker = Mock(return_value=0.4)
            with (
                patch.object(desc, "ProcessPoolExecutor", SynchronousPool),
                patch.object(desc, "lomap_worker", resumed_worker),
            ):
                scores = desc.score_lomap(pairs, indices, args, request)
            np.testing.assert_allclose(scores, [0.2, 0.4])
            self.assertEqual(resumed_worker.call_count, 1)

    def test_test_only_main_trains_without_uncomputed_lomap_rows(self):
        import json

        splits = ["train"] * 3 + ["val"] * 3 + ["test"] * 3
        frame = pd.DataFrame({
            artifacts.A: [f"CC{i}" for i in range(9)],
            artifacts.B: [f"CO{i}" for i in range(9)],
            artifacts.Y: [1., 2., 3.] * 3,
            "SE_abs": [1., 2., 3.] * 3,
            "split": splits,
        })
        benchmark = artifacts.benchmark_identity(frame)
        X = np.zeros((len(frame), len(desc.FEATURE_NAMES)))
        X[:, desc.FEATURE_NAMES.index("morgan_tanimoto_difficulty")] = np.tile(
            [0.1, 0.2, 0.3], 3,
        )
        lomap = np.array([np.nan] * 6 + [0.1, 0.2, 0.3])
        request = {"mode": "test-only"}
        FakeEstimator.fit_records = []
        FakeEstimator.predict_records = []

        with tempfile.TemporaryDirectory() as directory:
            reference = os.path.join(directory, "reference")
            output = os.path.join(directory, "output")
            os.makedirs(reference)
            os.makedirs(output)
            with open(os.path.join(reference, "benchmark.json"), "w") as stream:
                json.dump(benchmark, stream)
            with open(os.path.join(output, "features.pkl"), "wb"):
                pass

            argv = [
                "descriptor_rbfe.py", "--input", "unused.csv",
                "--reference_dir", reference, "--output_dir", output,
                "--threads", "1", "--lomap_mode", "test-only",
                "--phases", "train,evaluate",
            ]
            with (
                patch.object(sys, "argv", argv),
                patch.object(desc.hto, "phase1_load", return_value=frame),
                patch.object(desc, "load_features",
                             return_value=(frame, X, lomap, request)),
                patch.object(desc, "HistGradientBoostingRegressor", FakeEstimator),
            ):
                desc.main()

            with open(os.path.join(output, "models.pkl"), "rb") as stream:
                models = pickle.load(stream)
            self.assertEqual(set(models["models"]), {"descriptors"})
            self.assertEqual(models["train_mean"], 2.0)
            with open(os.path.join(output, "coverage.json")) as stream:
                coverage = json.load(stream)
            self.assertEqual(coverage["train"]["eligible"], 3)
            self.assertEqual(coverage["val"]["eligible"], 3)
            comparison = pd.read_csv(os.path.join(output, "comparison.csv"))
            self.assertIn("descriptors", comparison.method.to_list())
            self.assertIn("LOMAP (real, 1-sim)", comparison.method.to_list())
            self.assertIn("train_mean", comparison.method.to_list())
            self.assertNotIn("descriptors_lomap", comparison.method.to_list())


class StrictScaffoldTests(unittest.TestCase):
    def test_scaffold_extraction_uses_valid_molecules_strictly(self):
        tolyl = artifacts.scaffold_smiles("Cc1ccccc1")
        phenol = artifacts.scaffold_smiles("Oc1ccccc1")
        cyclohexyl = artifacts.scaffold_smiles("CC1CCCCC1")
        self.assertEqual(tolyl, phenol)
        self.assertEqual(tolyl, "c1ccccc1")
        self.assertNotEqual(tolyl, cyclohexyl)
        self.assertEqual(artifacts.scaffold_smiles("CCO"), "")
        with self.assertRaises(ValueError):
            artifacts.scaffold_smiles("C1CC")

    def test_phase1_scaffold_groups_are_disjoint_across_splits(self):
        from rdkit.Chem.Scaffolds import MurckoScaffold

        families = [
            ("C1CC1C", "C1CC1O"),
            ("C1CCC1C", "C1CCC1O"),
            ("C1CCCC1C", "C1CCCC1O"),
            ("C1CCCCC1C", "C1CCCCC1O"),
            ("Cc1ccccc1", "Oc1ccccc1"),
            ("Cc1ccccn1", "Oc1ccccn1"),
        ]
        pairs = []
        for a, b in families:
            pairs.extend([(a, a), (b, a)])
        for i, (a, _) in enumerate(families):
            next_a = families[(i + 1) % len(families)][0]
            pairs.append((a, next_a))
        raw = pd.DataFrame({
            artifacts.A: [a for a, _ in pairs],
            artifacts.B: [b for _, b in pairs],
            artifacts.Y: np.arange(1, len(pairs) + 1, dtype=float),
        })

        with tempfile.TemporaryDirectory() as directory:
            source = os.path.join(directory, "pairs.csv")
            raw.to_csv(source, index=False)
            args = SimpleNamespace(
                input=source, sample_size=0, seed=42,
                split_method="scaffold", val_frac=0.1, test_frac=0.1,
            )
            frame = hto.phase1_load(args, directory)

        self.assertEqual(set(frame.split), {"train", "val", "test"})
        self.assertTrue((frame.split.value_counts() > 0).all())
        scaffolds_by_split = {}
        for split in ("train", "val", "test"):
            subset = frame[frame.split == split]
            scaffolds = set()
            for a, b in subset[[artifacts.A, artifacts.B]].itertuples(
                    index=False, name=None):
                scaffolds.add(MurckoScaffold.MurckoScaffoldSmiles(smiles=a))
                scaffolds.add(MurckoScaffold.MurckoScaffoldSmiles(smiles=b))
            scaffolds_by_split[split] = scaffolds
        self.assertFalse(scaffolds_by_split["train"] & scaffolds_by_split["val"])
        self.assertFalse(scaffolds_by_split["train"] & scaffolds_by_split["test"])
        self.assertFalse(scaffolds_by_split["val"] & scaffolds_by_split["test"])

        diagnostics = desc.split_diagnostics(frame)
        for split in ("train", "val", "test"):
            self.assertEqual(
                diagnostics[split]["unresolved_scaffold_ligands"], 0,
            )
            self.assertGreater(diagnostics[split]["unique_scaffolds"], 0)
        self.assertEqual(
            diagnostics["overlap"],
            {
                "train_val": {"ligands": 0, "scaffolds": 0},
                "train_test": {"ligands": 0, "scaffolds": 0},
                "val_test": {"ligands": 0, "scaffolds": 0},
            },
        )

    def test_legacy_split_manifest_is_rejected_without_rewriting_pickle(self):
        with tempfile.TemporaryDirectory() as directory:
            source = os.path.join(directory, "pairs.csv")
            pd.DataFrame({
                artifacts.A: ["CCO"],
                artifacts.B: ["CCN"],
                artifacts.Y: [1.0],
            }).to_csv(source, index=False)
            phase_dir = os.path.join(directory, "phase1_hybrid")
            os.makedirs(phase_dir)
            cache = os.path.join(phase_dir, "cleaned_split_scaffold.pkl")
            pd.DataFrame({"legacy": [1]}).to_pickle(cache)
            legacy_request = artifacts.data_request(
                source, sample_size=0, seed=42, split_method="scaffold",
                val_frac=0.1, test_frac=0.1,
            )
            artifacts.write_cache_manifest(cache, legacy_request)
            with open(cache, "rb") as stream:
                original_pickle = stream.read()

            args = SimpleNamespace(
                input=source, sample_size=0, seed=42,
                split_method="scaffold", val_frac=0.1, test_frac=0.1,
            )
            with self.assertRaises(ValueError):
                hto.phase1_load(args, directory)
            with open(cache, "rb") as stream:
                self.assertEqual(stream.read(), original_pickle)


class LomapAuditTests(unittest.TestCase):
    def test_case_sampling_is_disjoint_deterministic_and_bounded(self):
        frame = pd.DataFrame({
            artifacts.A: ["CCO", "CCO", "CCN", "CCC"],
            artifacts.B: ["CCO", "CCN", "CCC", "CCN"],
            "SE_abs": [1.0, 2.0, 3.0, 4.0],
            audit.SCORE: [0.1, np.nan, 1.0, 0.5],
        })
        cases, counts = audit.select_cases(frame, per_group=1, seed=42)
        repeated, repeated_counts = audit.select_cases(frame, per_group=1, seed=42)
        self.assertEqual(cases, repeated)
        self.assertEqual(counts, repeated_counts)
        self.assertEqual(len(cases), 4 + 2)
        self.assertEqual(
            Counter(case["group"] for case in cases),
            Counter({
                "identical_smiles": 1, "missing": 1, "saturated": 1,
                "other_finite": 1, "control": 2,
            }),
        )
        self.assertEqual(
            counts,
            {
                "identical_smiles": 1, "missing": 1,
                "saturated": 1, "other_finite": 1,
            },
        )
        with self.assertRaises(ValueError):
            audit.select_cases(frame, per_group=0, seed=42)

    def test_audit_case_reports_score_and_rule_failures_separately(self):
        mapping = object()
        mapper = Mock()
        mapper.suggest_mappings.return_value = [mapping]
        components = Mock(side_effect=lambda **kwargs: object())

        def default_lomap_score(mapping, charge_changes_score=0.1):
            return 0.75

        def ecr_score(mapping, charge_changes_score):
            return charge_changes_score

        def mcsr_score(mapping, beta=0.1):
            return 0.8

        def broken_score(mapping):
            raise RuntimeError("rule failed")

        lomap = SimpleNamespace(
            LomapAtomMapper=Mock(return_value=mapper),
            default_lomap_score=default_lomap_score,
        )
        gufe = SimpleNamespace(SmallMoleculeComponent=components)
        scorers = SimpleNamespace(
            ecr_score=ecr_score, mcsr_score=mcsr_score,
            broken_score=broken_score,
        )
        case = {
            "id": "scored", "group": "identical_smiles",
            "smiles_a": "Cc1ccccc1", "smiles_b": "Cc1ccccc1",
            "label": 1.0, "historical_difficulty": 0.2,
        }
        with (
            patch.object(audit.AllChem, "EmbedMolecule", return_value=0) as embed,
            patch.object(audit.AllChem, "MMFFOptimizeMolecule", return_value=0) as mmff,
        ):
            result = audit.audit_case(case, lomap, gufe, scorers, timeout=5)
        self.assertEqual(result["status"], "scored")
        self.assertEqual(result["difficulty"], 0.25)
        self.assertEqual(result["rules"]["ecr_score"]["value"], 0.1)
        self.assertEqual(result["rules"]["mcsr_score"]["value"], 0.8)
        self.assertIn("RuntimeError", result["rules"]["broken_score"]["error"])
        lomap.LomapAtomMapper.assert_called_once_with(
            time=5, threed=False, max3d=1.0, shift=False,
        )
        self.assertEqual(components.call_count, 2)
        self.assertEqual(len(embed.call_args_list), 2)
        self.assertTrue(all(
            call.kwargs["randomSeed"] == 0xC0FFEE
            for call in embed.call_args_list
        ))
        self.assertTrue(all(
            call.kwargs["maxIters"] == 200
            for call in mmff.call_args_list
        ))

    def test_audit_case_forwards_max3d_zero_to_mapper(self):
        case = {
            "id": "no3d", "group": "other_finite",
            "smiles_a": "Cc1ccccc1", "smiles_b": "Oc1ccccc1",
            "label": 1.0, "historical_difficulty": 0.5,
        }
        mapper = Mock()
        mapper.suggest_mappings.return_value = iter(())
        lomap = SimpleNamespace(
            LomapAtomMapper=Mock(return_value=mapper),
            default_lomap_score=Mock(return_value=0.5),
        )
        gufe = SimpleNamespace(SmallMoleculeComponent=Mock(return_value=object()))
        with (
            patch.object(audit.AllChem, "EmbedMolecule", return_value=0),
            patch.object(audit.AllChem, "MMFFOptimizeMolecule", return_value=0),
        ):
            result = audit.audit_case(
                case, lomap, gufe, SimpleNamespace(), timeout=5, max3d=0,
            )
        self.assertEqual(result["status"], "no_mapping")
        self.assertEqual(result["mapper_settings"]["max3d"], 0)
        lomap.LomapAtomMapper.assert_called_once_with(
            time=5, threed=False, max3d=0, shift=False,
        )

    def test_audit_case_reports_no_mapping_embedding_and_default_score_failures(self):
        case = {
            "id": "case", "group": "missing",
            "smiles_a": "Cc1ccccc1", "smiles_b": "Oc1ccccc1",
            "label": 1.0, "historical_difficulty": None,
        }
        gufe = SimpleNamespace(SmallMoleculeComponent=Mock(return_value=object()))
        scorers = SimpleNamespace()

        with (
            patch.object(audit.AllChem, "EmbedMolecule", return_value=0),
            patch.object(audit.AllChem, "MMFFOptimizeMolecule", return_value=0),
        ):
            mapper = Mock()
            mapper.suggest_mappings.return_value = iter(())
            lomap = SimpleNamespace(
                LomapAtomMapper=Mock(return_value=mapper),
                default_lomap_score=Mock(return_value=0.5),
            )
            no_mapping = audit.audit_case(case, lomap, gufe, scorers, timeout=5)
        self.assertEqual(no_mapping["status"], "no_mapping")
        self.assertEqual(no_mapping["stage"], "mapping")

        with patch.object(audit.AllChem, "EmbedMolecule", return_value=-1):
            embed_failure = audit.audit_case(
                case, SimpleNamespace(), gufe, scorers, timeout=5,
            )
        self.assertEqual(embed_failure["status"], "error")
        self.assertEqual(embed_failure["stage"], "embed_a")

        mapper = Mock()
        mapper.suggest_mappings.return_value = [object()]

        def broken_default_score(mapping, charge_changes_score=0.1):
            raise RuntimeError("default score failed")

        lomap = SimpleNamespace(
            LomapAtomMapper=Mock(return_value=mapper),
            default_lomap_score=broken_default_score,
        )
        with (
            patch.object(audit.AllChem, "EmbedMolecule", return_value=0),
            patch.object(audit.AllChem, "MMFFOptimizeMolecule", return_value=0),
        ):
            score_failure = audit.audit_case(
                case, lomap, gufe, scorers, timeout=5,
            )
        self.assertEqual(score_failure["status"], "error")
        self.assertEqual(score_failure["stage"], "default_score")
        self.assertIn("RuntimeError", score_failure["error"])


class RescoreLomapTests(unittest.TestCase):
    @staticmethod
    def saved_predictions():
        return pd.DataFrame({
            artifacts.A: ["CCO", "CCN", "CCC", "CCF"],
            artifacts.B: ["CCN", "CCC", "CCF", "CCO"],
            "SE_abs": [1., 2., 3., 4.],
            "descriptors": [1., 2., 3., 4.],
            "train_mean": [2.5] * 4,
            "Tanimoto (1-sim)": [0.1, 0.2, 0.3, 0.4],
            "Hybrid-topo GNN": [1.1, 2.1, 3.1, 4.1],
            rescore.LOMAP: [0.9, np.nan, 0.7, np.nan],
            "common_eligible": [True, False, True, False],
        })

    def test_production_mapper_disables_geometry_trimming(self):
        mapper = Mock()
        mapper.suggest_mappings.return_value = [object()]
        fake_lomap = SimpleNamespace(
            LomapAtomMapper=Mock(return_value=mapper),
            default_lomap_score=Mock(return_value=0.75),
        )
        fake_gufe = SimpleNamespace(
            SmallMoleculeComponent=Mock(return_value=object()),
        )
        with (
            patch.dict(sys.modules, {"lomap": fake_lomap, "gufe": fake_gufe}),
            patch.object(hto.AllChem, "EmbedMolecule", return_value=0),
            patch.object(hto.AllChem, "MMFFOptimizeMolecule", return_value=0),
        ):
            difficulty = hto._lomap_like_score("CCO", "CCN", mcs_timeout=5)
        self.assertEqual(difficulty, 0.25)
        fake_lomap.LomapAtomMapper.assert_called_once_with(
            time=5, threed=False, max3d=0.0, shift=False,
        )

    def test_rescore_preserves_predictions_and_recomputes_common_coverage(self):
        frame = self.saved_predictions()
        original = frame.copy(deep=True)
        corrected = np.array([0.1, 0.2, np.nan, 0.4])

        output, full, shared, coverage = rescore.compare_saved(frame, corrected)

        pd.testing.assert_frame_equal(frame, original)
        for column in (
                artifacts.A, artifacts.B, "SE_abs", "descriptors", "train_mean",
                "Tanimoto (1-sim)", "Hybrid-topo GNN"):
            pd.testing.assert_series_equal(output[column], original[column])
        np.testing.assert_allclose(
            output["previous_lomap_difficulty"], original[rescore.LOMAP],
            equal_nan=True,
        )
        np.testing.assert_allclose(
            output[rescore.LOMAP], corrected, equal_nan=True,
        )
        np.testing.assert_array_equal(
            output["common_eligible"], [True, True, False, True],
        )
        self.assertEqual(
            coverage,
            {
                "total": 4, "previous_common": 2,
                "corrected_lomap_finite": 3,
                "corrected_lomap_max_difficulty": 0,
                "new_common": 3, "previous_and_new_common": 1,
                "newly_eligible": 2, "no_longer_eligible": 1,
            },
        )
        self.assertEqual(shared, [])
        metrics = {row["method"]: row for row in full}
        self.assertEqual(metrics["descriptors"]["mse"], 0.0)
        self.assertEqual(metrics["descriptors"]["spearman_r"], 1.0)

    def test_rescore_rejects_mismatched_or_incompatible_saved_predictions(self):
        frame = self.saved_predictions()
        with self.assertRaises(ValueError):
            rescore.compare_saved(frame, np.array([0.1, 0.2]))

        lomap_trained = frame.assign(descriptors_lomap=[1., 2., 3., 4.])
        with self.assertRaises(ValueError):
            rescore.compare_saved(lomap_trained, np.array([0.1] * 4))

        missing_column = frame.drop(columns=["train_mean"])
        with self.assertRaises(ValueError):
            rescore.compare_saved(missing_column, np.array([0.1] * 4))


class LearningCurveTests(unittest.TestCase):
    @staticmethod
    def small_frame():
        splits = ["train"] * 12 + ["val"] * 4 + ["test"] * 3
        labels = list(np.arange(1., 13.)) + list(np.arange(1., 5.)) + [1000., 1001., 1002.]
        return pd.DataFrame({
            artifacts.A: [f"CC{i}O" for i in range(len(splits))],
            artifacts.B: [f"CN{i}O" for i in range(len(splits))],
            artifacts.Y: labels,
            "SE_abs": labels,
            "split": splits,
        })

    def test_nested_rows_are_nested_train_only_and_validate_sizes(self):
        frame = self.small_frame()
        train = frame[frame.split == "train"]
        selected, val = curve.nested_rows(frame, [4, 9], seed=42)
        sampled4 = train.sample(n=4, random_state=42)
        sampled9 = train.sample(n=9, random_state=42)
        self.assertEqual(
            artifacts.pair_keys(selected.iloc[:4]),
            artifacts.pair_keys(sampled4),
        )
        self.assertEqual(
            artifacts.pair_keys(selected),
            artifacts.pair_keys(sampled9),
        )
        self.assertTrue(
            set(artifacts.pair_keys(selected.iloc[:4]))
            <= set(artifacts.pair_keys(selected)),
        )
        self.assertEqual(
            artifacts.pair_keys(val),
            artifacts.pair_keys(frame[frame.split == "val"]),
        )
        self.assertFalse(
            set(artifacts.pair_keys(selected))
            & set(artifacts.pair_keys(frame[frame.split == "test"])),
        )
        self.assertFalse(
            set(artifacts.pair_keys(val))
            & set(artifacts.pair_keys(frame[frame.split == "test"])),
        )
        for sizes in ([], [9, 4], [4, 4], [0, 4], [20]):
            with self.subTest(sizes=sizes), self.assertRaises(ValueError):
                curve.nested_rows(frame, sizes, seed=42)

    def test_main_fits_only_train_and_validation_and_resumes_models(self):
        import json

        frame = self.small_frame()
        benchmark = artifacts.benchmark_identity(frame)
        FakeCurveEstimator.fit_records = []
        FakeCurveEstimator.constructor_settings = []
        feature_frames = []

        def fake_feature_matrix(subset, path, request):
            feature_frames.append(subset.copy())
            return np.column_stack((
                subset.SE_abs.to_numpy(dtype=float),
                np.ones(len(subset), dtype=float),
            ))

        with tempfile.TemporaryDirectory() as directory:
            reference = os.path.join(directory, "reference")
            output = os.path.join(directory, "output")
            os.makedirs(reference)
            os.makedirs(output)
            with open(os.path.join(reference, "benchmark.json"), "w") as stream:
                json.dump(benchmark, stream)
            argv = [
                "descriptor_learning_curve.py", "--input", "unused.csv",
                "--reference_dir", reference, "--output_dir", output,
                "--train_sizes", "4,9", "--threads", "1",
            ]
            with (
                patch.object(sys, "argv", argv),
                patch.object(curve.hto, "phase1_load", return_value=frame),
                patch.object(curve, "feature_matrix", side_effect=fake_feature_matrix),
                patch.object(curve, "HistGradientBoostingRegressor", FakeCurveEstimator),
            ):
                curve.main()
                curve.main()

            selected, _ = curve.nested_rows(frame, [4, 9], seed=42)
            expected_y4 = selected.iloc[:4].sort_index().SE_abs.to_numpy()
            expected_y9 = selected.iloc[:9].sort_index().SE_abs.to_numpy()
            self.assertEqual(len(FakeCurveEstimator.fit_records), 2)
            np.testing.assert_array_equal(
                FakeCurveEstimator.fit_records[0]["y"], expected_y4,
            )
            np.testing.assert_array_equal(
                FakeCurveEstimator.fit_records[0]["X"][:, 0], expected_y4,
            )
            np.testing.assert_array_equal(
                FakeCurveEstimator.fit_records[1]["y"], expected_y9,
            )
            np.testing.assert_array_equal(
                FakeCurveEstimator.fit_records[1]["X"][:, 0], expected_y9,
            )
            self.assertTrue(all(
                record["y"].max() < 1000
                for record in FakeCurveEstimator.fit_records
            ))
            self.assertEqual(
                FakeCurveEstimator.constructor_settings,
                [curve.MODEL_SETTINGS.copy(), curve.MODEL_SETTINGS.copy()],
            )
            self.assertEqual(len(feature_frames), 4)
            self.assertTrue(all(
                set(subset.split) <= {"train", "val"}
                for subset in feature_frames
            ))
            table = pd.read_csv(
                os.path.join(output, "validation_learning_curve.csv"),
            )
            self.assertEqual(set(table.train_selected), {4, 9})
            self.assertFalse(os.path.exists(
                os.path.join(output, "test_predictions.csv"),
            ))

    def test_fit_point_fixed_model_and_nan_eligibility(self):
        X_train = np.arange(80, dtype=float).reshape(-1, 1)
        y_train = X_train[:, 0].copy()
        X_val = np.array([[10.], [20.], [30.], [40.]])
        y_val = X_val[:, 0].copy()
        with threadpool_limits(limits=1):
            model, record, prediction, valid = curve.fit_point(
                X_train, y_train, X_val, y_val, seed=42,
            )
            self.assertEqual(record["train_mean"], 39.5)
            self.assertEqual(record["train_eligible"], 80)
            self.assertEqual(record["val_eligible"], 4)
            self.assertTrue(valid.all())
            self.assertTrue(np.isfinite(prediction).all())
            for name, value in curve.MODEL_SETTINGS.items():
                self.assertEqual(model.get_params()[name], value)

            X_train_nan = X_train.copy()
            X_train_nan[0, 0] = np.nan
            X_val_nan = X_val.copy()
            X_val_nan[1, 0] = np.nan
            _, nan_record, nan_prediction, nan_valid = curve.fit_point(
                X_train_nan, y_train, X_val_nan, y_val, seed=42,
            )
        self.assertEqual(nan_record["train_eligible"], 79)
        self.assertEqual(nan_record["val_eligible"], 3)
        self.assertEqual(nan_record["train_mean"], 40.0)
        np.testing.assert_array_equal(nan_valid, [True, False, True, True])
        self.assertTrue(np.isnan(nan_prediction[1]))

    def test_feature_cache_reuses_and_rejects_request_or_row_mismatch(self):
        frame = pd.DataFrame({
            artifacts.A: ["CCO", "CC"],
            artifacts.B: ["CCN", "CO"],
        })
        request = {"synthetic": True}
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "train_features.pkl")
            features = curve.feature_matrix(frame, path, request)
            with patch.object(
                    desc, "pair_features",
                    side_effect=AssertionError("feature cache should be reused")):
                cached = curve.feature_matrix(frame, path, request)
            np.testing.assert_array_equal(cached, features)

            with self.assertRaises(ValueError):
                curve.feature_matrix(frame, path, {"synthetic": False})
            reordered = frame.iloc[::-1].reset_index(drop=True)
            with self.assertRaises(ValueError):
                curve.feature_matrix(reordered, path, request)


class LabelStructureTests(unittest.TestCase):
    @staticmethod
    def frame(values_for):
        rng = np.random.default_rng(0)
        pairs = [(i, j) for i in range(30) for j in range(30)
                 if i != j and rng.random() < .6] + [(0, 0), (1, 1)]
        i, j = np.array(pairs).T
        return pd.DataFrame({
            artifacts.A: [f"C{k}" for k in i],
            artifacts.B: [f"C{k}" for k in j],
            artifacts.Y: values_for(i, j, rng),
        })

    def test_additive_labels_are_recovered_and_self_pairs_excluded(self):
        s = np.random.default_rng(1).normal(size=30)
        result = labels.diagnose(
            self.frame(lambda i, j, rng: s[i] - s[j] + 5. * (i == j)),
            triangle_samples=5000,
        )
        self.assertEqual(result["self_pairs"]["count"], 2)
        self.assertEqual(result["self_pairs"]["mean_abs_label"], 5.)
        self.assertGreater(result["models"]["difference"]["heldout_r2"], .999)
        self.assertGreater(result["models"]["two_slot"]["heldout_r2"], .999)
        self.assertAlmostEqual(
            result["reversed_pairs"]["pearson_forward_vs_negated_reverse"], 1.)
        self.assertGreater(result["triangles"]["closed_triangles"], 100)
        self.assertLess(result["triangles"]["ratio_to_null"], 1e-6)

    def test_random_labels_do_not_close_triangles(self):
        result = labels.diagnose(
            self.frame(lambda i, j, rng: rng.normal(size=len(i))),
            triangle_samples=5000,
        )
        self.assertLess(result["models"]["difference"]["heldout_r2"], .1)
        self.assertGreater(result["triangles"]["ratio_to_null"], .8)

    def test_lookup_reports_missing_ordered_edges(self):
        src, dst = np.array([0, 1, 2]), np.array([1, 2, 0])
        np.testing.assert_array_equal(
            labels.lookup(src, dst, 3, np.array([1, 0, 2]), np.array([2, 2, 0])),
            [1, -1, 2],
        )


class CompoundUncertaintyModelTests(unittest.TestCase):
    @staticmethod
    def benchmark_frame():
        rng = np.random.default_rng(0)
        smiles = [p + "C" * k for k in range(1, 31) for p in ("", "O", "N", "F")]
        heavy = {s: len(s) for s in smiles}
        order = rng.permutation(len(smiles))
        groups = {"train": [smiles[i] for i in order[:80]],
                  "val": [smiles[i] for i in order[80:100]],
                  "test": [smiles[i] for i in order[100:]]}
        rows = []
        for split, members in groups.items():
            for a in members:
                for b in members:
                    if a != b and rng.random() < .5:
                        y = .02 * heavy[a] - .01 * heavy[b] + .001 * rng.normal()
                        rows.append((a, b, y, abs(y), split))
        return pd.DataFrame(rows, columns=[
            artifacts.A, artifacts.B, artifacts.Y, "SE_abs", "split"])

    def write_reference(self, directory, frame, scoring_version):
        import json

        test = frame[frame.split == "test"]
        os.makedirs(directory)
        with open(os.path.join(directory, "benchmark.json"), "w") as stream:
            json.dump(artifacts.benchmark_identity(frame), stream)
        with open(os.path.join(directory, "features.pkl.manifest.json"), "w") as stream:
            json.dump({"request": {"lomap_scoring_version": scoring_version}}, stream)
        reference = test[[artifacts.A, artifacts.B, "SE_abs"]].iloc[::-1].copy()
        rng = np.random.default_rng(1)
        for name in compound.REFERENCE_METHODS:
            reference[name] = rng.random(len(reference))
        reference.to_csv(os.path.join(directory, "test_predictions.csv"), index=False)

    def run_main(self, directory, frame, scoring_version):
        import json

        reference = os.path.join(directory, "v4")
        lomap_reference = os.path.join(directory, "descriptors_v4")
        output = os.path.join(directory, "out")
        os.makedirs(reference)
        with open(os.path.join(reference, "benchmark.json"), "w") as stream:
            json.dump(artifacts.benchmark_identity(frame), stream)
        self.write_reference(lomap_reference, frame, scoring_version)
        argv = ["compound_uncertainty_model.py", "--input", "unused.csv",
                "--reference_dir", reference, "--lomap_reference_dir", lomap_reference,
                "--output_dir", output, "--min_edges", "2", "--threads", "1"]
        with (patch.object(sys, "argv", argv),
              patch.object(compound.hto, "phase1_load", return_value=frame)):
            compound.main()
        return output

    def test_two_slot_fit_recovers_exact_labels(self):
        frame = self.benchmark_frame()
        train = frame[frame.split == "train"]
        exact = train.assign(**{artifacts.Y: train[artifacts.Y].round(2)})
        table, _, r2 = compound.fit_two_slot(exact)
        self.assertGreater(r2, .99)
        self.assertEqual(int(table.n_first.sum()), len(exact))

    def test_end_to_end_recovers_size_signal_on_held_out_compounds(self):
        import json

        frame = self.benchmark_frame()
        with tempfile.TemporaryDirectory() as directory:
            output = self.run_main(directory, frame, artifacts.LOMAP_SCORING_VERSION)
            with open(os.path.join(output, "summary.json")) as stream:
                summary = json.load(stream)
            comparison = pd.read_csv(os.path.join(output, "comparison.csv"))
            predictions = pd.read_csv(os.path.join(output, "test_predictions.csv"))
        test = frame[frame.split == "test"]
        self.assertGreater(summary["test_signed"]["spearman"], .9)
        self.assertGreater(summary["test_signed_ceiling_r2_from_test_only_fit"], .99)
        self.assertEqual(
            set(comparison.method), {"compound_model", *compound.REFERENCE_METHODS})
        self.assertEqual(artifacts.pair_keys(predictions), artifacts.pair_keys(test))
        self.assertEqual(summary["test_common_pairs"], len(test))

    def test_uncorrected_lomap_reference_is_rejected(self):
        frame = self.benchmark_frame()
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "corrected scorer"):
                self.run_main(directory, frame, None)

    def test_reference_with_different_pairs_is_rejected(self):
        frame = self.benchmark_frame()
        test = frame[frame.split == "test"].reset_index(drop=True)
        with tempfile.TemporaryDirectory() as directory:
            reference = os.path.join(directory, "ref")
            self.write_reference(reference, frame, artifacts.LOMAP_SCORING_VERSION)
            path = os.path.join(reference, "test_predictions.csv")
            pd.read_csv(path).iloc[1:].to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "exactly the test pairs"):
                compound.load_reference(reference, test, artifacts.benchmark_identity(frame))


class CompoundModelSearchTests(unittest.TestCase):
    def test_feature_sets_and_grouped_cv_select_size_signal(self):
        import compound_model_search as search

        smiles = [p + "C" * k for k in range(1, 26) for p in ("", "O", "N", "F")]
        features = search.feature_sets(smiles)
        self.assertEqual(features["baseline"].shape, (100, len(compound.FEATURE_NAMES)))
        self.assertEqual(features["morgan_counts"].shape, (100, 2048))
        self.assertEqual(features["rdkit+morgan"].shape[1],
                         features["rdkit"].shape[1] + 2048)
        s = np.array([len(x) for x in smiles], dtype=float)
        grid = [g for g in search.candidates(42) if g[1] in ("ridge_a1", "rf")]
        ranking, oof = search.cross_validate(
            features, s, [x[0] for x in smiles], grid, folds=4)
        self.assertEqual(len(ranking), len(grid))
        self.assertGreater(ranking.cv_spearman.max(), .8)
        self.assertTrue(all(len(p) == len(s) for p in oof.values()))


if __name__ == "__main__":
    unittest.main()
