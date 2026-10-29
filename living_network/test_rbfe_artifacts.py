import os
import tempfile
import unittest
from types import SimpleNamespace

import pandas as pd
import torch

import hybrid_topo_rbfe as hto
import mcs_precompute
import rbfe_artifacts as artifacts


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


if __name__ == "__main__":
    unittest.main()
