import unittest

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from prepare_fah_benchmark import chemistry_split, prepare, network_split, assert_disjoint
from run_fah_benchmark import metrics, bootstrap, choose_model


class FahBenchmarkTests(unittest.TestCase):
    def test_compound_links_unite_scaffolds_and_order_does_not_change_assignment(self):
        f = pd.DataFrame({
            "smi_a": ["c1ccccc1", "C1CCCCC1", "c1ccncc1"],
            "smi_b": ["Cc1ccccc1", "CC1CCCCC1", "Cc1ccncc1"],
            "initial_compound_id": ["shared", "shared", "pyridine"],
            "final_compound_id": ["benzene2", "cyclo2", "pyridine2"],
        })
        split, _ = chemistry_split(f)
        self.assertEqual(split.group_a.iloc[0], split.group_a.iloc[1])
        shuffled, _ = chemistry_split(f.iloc[::-1])
        pd.testing.assert_frame_equal(split, shuffled.loc[split.index])
        self.assertTrue(split.group_a.eq(split.group_b).all())

    def test_pilot_uses_generation_labels_and_ignores_reliability(self):
        records, generations = [], []
        for i, reliability in enumerate([True, False, None]):
            records.append({
                "edge_id": str(i), "source_sha256": "source", "series_name": "series",
                "run_id": i, "initial_compound_id": "A", "final_compound_id": f"B{i}",
                "initial_microstate_smiles": "c1ccccc1",
                "final_microstate_smiles": ["CCO", "CCN", "CCC"][i],
                "initial_resolved": True, "final_resolved": True,
                "binding_free_energy_stderr": .5, "reliable_transformation": reliability,
            })
            for phase in ("complex_phase", "solvent_phase"):
                for gen, se, n in [(1, 3., 9), (0, 2., 4)]:
                    generations.append({
                        "edge_id": str(i), "phase": phase, "generation": gen,
                        "delta_f_stderr": se, "num_work_pairs": n,
                        "raw_work_pair_count": n,
                    })
        tasks, _ = prepare(pd.DataFrame(records), pd.DataFrame(generations))
        self.assertEqual(len(tasks["edge"]), 3)
        for task in ("pilot_complex", "pilot_solvent"):
            self.assertEqual(len(tasks[task]), 3)
            self.assertTrue(tasks[task].pilot_proxy.eq(4).all())
            self.assertTrue(tasks[task].target.eq(9).all())
        bad = pd.DataFrame(generations)
        bad.loc[0, "generation"] = 0
        with self.assertRaisesRegex(ValueError, "Repeated generation"):
            prepare(pd.DataFrame(records), bad)

    def test_both_network_settings_preserve_their_distinct_leakage_contracts(self):
        rows = []
        for anchor in range(6):
            for variable in range(120):
                # Shared targets deliberately link different networks.
                group = f"variable-{variable}" if variable < 20 else f"variable-{anchor}-{variable}"
                rows.append({
                    "source_sha256": f"source-{anchor}",
                    "initial_compound_id": f"anchor-{anchor}",
                    "final_compound_id": group,
                    "smi_a": f"anchor-smiles-{anchor}", "smi_b": group,
                    "group_a": f"anchor-group-{anchor}", "group_b": group,
                })
        # A target belongs to another reference family: primary must exclude it.
        rows.append(dict(rows[0], final_compound_id="anchor-1",
                         smi_b="anchor-smiles-1", group_b="anchor-group-1"))
        frame = pd.DataFrame(rows)
        primary, _ = network_split(frame, "known_reference")
        self.assertEqual(primary.iloc[-1].split, "excluded")
        kept = primary[primary.split.ne("excluded")]
        self.assertEqual(set(kept.split), {"train", "val", "test"})
        assert_disjoint(kept, ["variable_group"])
        self.assertTrue(kept.resample_group_a.eq(kept.variable_group).all())
        strict, _ = network_split(frame, "strict_network")
        kept = strict[strict.split.ne("excluded")]
        self.assertEqual(set(kept.split), {"train", "val", "test"})
        assert_disjoint(kept, ["group_a", "group_b"])
        self.assertTrue(strict.loc[strict.variable_group.eq("variable-0"), "split"].eq("excluded").all())
        reversed_input = frame.rename(columns={
            "initial_compound_id": "final_compound_id", "final_compound_id": "initial_compound_id",
            "smi_a": "smi_b", "smi_b": "smi_a", "group_a": "group_b", "group_b": "group_a",
        })
        reversed_split, _ = network_split(reversed_input, "strict_network")
        self.assertEqual(strict.split.tolist(), reversed_split.split.tolist())

    def test_network_assignment_ignores_row_order_and_labels(self):
        frame = pd.DataFrame([{
            "source_sha256": f"s{i}", "initial_compound_id": f"a{i}",
            "final_compound_id": f"b{i}-{j}", "smi_a": f"a{i}",
            "smi_b": f"b{i}-{j}", "group_a": f"a{i}", "group_b": f"b{i}-{j}",
            "target": j,
        } for i in range(4) for j in range(10)])
        first, _ = network_split(frame, "strict_network")
        changed = frame.iloc[::-1].copy()
        changed.target = 1e9
        second, _ = network_split(changed, "strict_network")
        self.assertEqual(first.split.tolist(), second.loc[first.index, "split"].tolist())

    def test_weighted_metrics_match_literal_replication_with_ties(self):
        y = np.array([1., 2., 2., 4.])
        pred = np.array([3., 1., 2., 2.])
        weights = np.array([2, 0, 3, 1])
        result = metrics(y, pred, weights)
        self.assertAlmostEqual(
            result["spearman"], spearmanr(np.repeat(y, weights), np.repeat(pred, weights)).statistic)
        self.assertAlmostEqual(
            result["pearson"], np.corrcoef(np.repeat(y, weights), np.repeat(pred, weights))[0, 1])

    def test_constant_predictions_have_undefined_correlations(self):
        result = metrics(np.arange(7.), np.full(7, .3141592653589793),
                         np.array([1, 3, 5, 2, 1, 4, 1]))
        self.assertIsNone(result["pearson"])
        self.assertIsNone(result["spearman"])

    def test_identical_predictors_have_zero_paired_bootstrap_difference(self):
        frame = pd.DataFrame({
            "group_a": list("abcdef"), "group_b": list("abcdef"),
            "target": np.arange(6.), "first": np.arange(6.), "second": np.arange(6.),
        })
        result = bootstrap(frame, ["first", "second"], "second", repetitions=250)
        for value in result["differences"].values():
            self.assertEqual(value["ci95"], [0., 0.])

    def test_model_selection_is_unaffected_by_test_targets(self):
        from threadpoolctl import threadpool_limits
        rng = np.random.default_rng(42)
        x = rng.normal(size=(120, 3))
        frame = pd.DataFrame({
            "split": ["train"] * 80 + ["val"] * 20 + ["test"] * 20,
            "target": np.exp(x[:, 0] / 3),
        })
        with threadpool_limits(limits=1):
            (name, model), scores = choose_model(x, frame)
            changed = frame.copy()
            changed.loc[changed.split.eq("test"), "target"] = 1e9
            (other_name, other_model), other_scores = choose_model(x, changed)
        self.assertEqual(name, other_name)
        self.assertEqual(scores, other_scores)
        np.testing.assert_array_equal(model.predict(x[-20:]), other_model.predict(x[-20:]))
        self.assertGreater(max(s["validation_spearman"] for s in scores), .8)


if __name__ == "__main__":
    unittest.main()
