import json
import os
import tempfile
import unittest

import torch

from .action_generators import (
    candidate_geometry_from_heatmap,
    candidate_indices_from_heatmap,
    get_action_abstraction_spec,
)
from .diagnostics import DecisionLogger
from .graph_oracle_controls import controlled_graph_action
from .oracle_selector import select_oracle_level


class ActionAbstractionSmokeTest(unittest.TestCase):
    def test_fixed_specs_are_ordered(self):
        coarse = get_action_abstraction_spec("coarse")
        default = get_action_abstraction_spec("default")
        fine = get_action_abstraction_spec("fine")
        self.assertLess(coarse.max_predictions, default.max_predictions)
        self.assertLess(default.max_predictions, fine.max_predictions)

    def test_unknown_level_fails(self):
        with self.assertRaises(ValueError):
            get_action_abstraction_spec("unknown")

    def test_heatmap_candidates_have_expected_geometry(self):
        heatmap = torch.zeros(120, 12)
        heatmap[10, 2] = 1.0
        heatmap[80, 5] = 0.9
        indices = candidate_indices_from_heatmap(heatmap, "default")
        self.assertGreaterEqual(len(indices), 1)
        geometry = candidate_geometry_from_heatmap(heatmap, "default")
        self.assertEqual(len(geometry["angles"]), len(geometry["distances"]))
        self.assertEqual(len(geometry["angles"]), len(geometry["scores"]))
        self.assertTrue(all(0.25 <= d <= 3.0 for d in geometry["distances"]))

    def test_dense_native_union_contains_default_candidates(self):
        torch.manual_seed(7)
        heatmap = torch.rand(120, 12)
        native = {(a, d) for a, d, _ in candidate_indices_from_heatmap(heatmap, "default")}
        dense = {(a, d) for a, d, _ in candidate_indices_from_heatmap(heatmap, "dense_native_union")}
        self.assertTrue(native.issubset(dense))
        self.assertGreaterEqual(len(dense), len(native))

    def test_oracle_selector_is_analysis_only_and_stable(self):
        level, value = select_oracle_level({
            "coarse": [{"progress": 0.2}],
            "fine": [{"progress": 0.6}],
        })
        self.assertEqual(level, "fine")
        self.assertEqual(value, 0.6)

    def test_graph_oracle_control_separates_stop_and_ranking(self):
        self.assertEqual(controlled_graph_action("rank_only", 0, 4, 3, 5.0), 0)
        self.assertEqual(controlled_graph_action("rank_only", 4, 4, 3, 5.0), 3)
        self.assertEqual(controlled_graph_action("stop_only", 4, 4, 3, 1.0), 0)
        self.assertEqual(controlled_graph_action("stop_only", 4, 4, 3, 5.0), 4)
        self.assertEqual(controlled_graph_action("rank_and_stop", 4, 4, 3, 5.0), 3)

    def test_diagnostic_jsonl_is_append_only(self):
        fd, path = tempfile.mkstemp(prefix="aaa_diag_", suffix=".jsonl")
        os.close(fd)
        try:
            logger = DecisionLogger(path)
            logger.write({"episode_id": "e0", "score": 1.0})
            logger.write({"episode_id": "e1", "score": 2.0})
            with open(path) as stream:
                rows = [json.loads(line) for line in stream]
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[1]["diagnostic_index"], 1)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
