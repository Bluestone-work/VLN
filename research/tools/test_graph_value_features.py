import unittest

import numpy as np

from graph_value_feasibility import action_features


def packed(values):
    return {"__array__": list(values), "dtype": "<f4"}


def state():
    return {
        "privileged_labels_in_features": False,
        "graph_position": packed([0.0, 0.0, 0.0]),
        "options": [
            {"index": 0, "admissible": True, "logit": 0.0,
             "embedding": packed([0.0, 1.0]),
             "action": {"act": 0, "back_path": []}},
            {"index": 1, "admissible": True, "logit": 1.0,
             "embedding": packed([1.0, 3.0]),
             "action": {"act": 4, "back_path": [],
                        "front_pos": packed([0.0, 0.0, 0.0]),
                        "ghost_pos": packed([0.0, 0.0, 2.0])}},
        ],
    }


class GraphValueFeatureTests(unittest.TestCase):
    def test_stop_probability_is_state_level_and_uses_admissible_order(self):
        row = state()
        stop = action_features(row, row["options"][0])
        move = action_features(row, row["options"][1])
        expected = 1.0 / (1.0 + np.exp(1.0))
        self.assertAlmostEqual(stop[3], expected)
        self.assertAlmostEqual(move[3], expected)

    def test_back_path_cost_starts_at_current_graph_position(self):
        row = state()
        option = row["options"][1]
        option["action"]["back_path"] = [
            {"__tuple__": ["old", packed([-2.0, 0.0, 0.0])]},
            {"__tuple__": ["old2", packed([-2.0, 0.0, 3.0])]},
        ]
        values = action_features(row, option)
        self.assertAlmostEqual(values[8], 2.0 + 3.0)
        self.assertEqual(values[9], 2.0)

    def test_privileged_or_nonfinite_inputs_are_rejected(self):
        row = state()
        row["privileged_labels_in_features"] = True
        with self.assertRaises(ValueError):
            action_features(row, row["options"][1])
        row = state()
        row["options"][1]["logit"] = float("nan")
        with self.assertRaises(ValueError):
            action_features(row, row["options"][0])


if __name__ == "__main__":
    unittest.main()
