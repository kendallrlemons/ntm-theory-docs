import unittest

import pandas as pd

from prepare_fah_growth import schedules


def fixture():
    rows = []
    for split, count in (("train", 2), ("val", 5), ("test", 5)):
        for i in range(count):
            rows.append(dict(edge_id=f"{split}{i}", source_sha256="source", series_name="campaign",
                             split=split, reference_smiles="R", variable_smiles=f"{split}{i}",
                             variable_group=f"{split}{i}", target=i))
    return pd.DataFrame(rows)


class GrowthTests(unittest.TestCase):
    def test_star_reports_missing_choices_and_separate_worlds(self):
        initial, arrivals, coverage = schedules(fixture())
        self.assertEqual(len(initial), 2)
        self.assertEqual(len(arrivals), 30)
        self.assertEqual(len(coverage), 30)
        self.assertFalse(coverage.connection_choice_evaluable.any())
        self.assertTrue(coverage.observed_candidate_pairs.eq(1).all())
        self.assertTrue(coverage[coverage.step.eq(1)].prior_nodes.eq(3).all())
        self.assertTrue(coverage[coverage.step.eq(5)].prior_nodes.eq(7).all())
        self.assertTrue((coverage.missing_candidate_pairs == coverage.prior_nodes - 1).all())

    def test_label_and_row_order_invariance(self):
        f = fixture()
        expected = schedules(f)[1].sort_values(["seed", "edge_id"]).reset_index(drop=True)
        f.target = -999999
        actual = schedules(f.sample(frac=1, random_state=3))[1].sort_values(["seed", "edge_id"]).reset_index(drop=True)
        pd.testing.assert_frame_equal(expected, actual)
        self.assertNotIn("target", actual)

    def test_whole_groups_and_repeated_records_not_new_choices(self):
        f = fixture()
        duplicate = f[f.edge_id.eq("test0")].copy()
        duplicate.edge_id = "repeat"
        f = pd.concat([f, duplicate], ignore_index=True)
        _, arrivals, coverage = schedules(f)
        for _, part in arrivals.groupby(["seed", "split", "variable_group"]):
            self.assertEqual(part.step.nunique(), 1)
        self.assertTrue(coverage.observed_candidate_pairs.eq(1).all())

    def test_overlap_rejected(self):
        f = fixture()
        f.loc[f.edge_id.eq("test0"), "variable_group"] = "train0"
        with self.assertRaises(ValueError):
            schedules(f)

    def test_insufficient_coverage_reported(self):
        f = fixture()
        _, arrivals, coverage = schedules(f[~f.edge_id.eq("test0")])
        self.assertTrue(arrivals.split.eq("val").all())
        self.assertEqual(len(coverage[coverage.status.ne("ready")]), 1)


if __name__ == "__main__":
    unittest.main()
