import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
from audit_ranker_validity import assert_geometry, cluster_ci, graph_pool, indexed_jsonl
from audit_selected_target_geometry import proposal_position


class ValidityTest(unittest.TestCase):
    def test_pose_and_bearing_use_baseline_coordinate_convention(self):
        pose = {'position': [1, 2, 3], 'rotation': [0, 0, 0, 1]}
        np.testing.assert_allclose(proposal_position(pose, {'angle': 0, 'distance': 1}), [1, 2, 2])
        pose['rotation'] = [0, math.sin(math.pi / 4), 0, math.cos(math.pi / 4)]
        np.testing.assert_allclose(proposal_position(pose, {'angle': 0, 'distance': 1}), [0, 2, 3], atol=1e-12)

    def test_geometry_audit_rejects_unmodeled_pitch(self):
        with self.assertRaises(ValueError):
            proposal_position({'position': [0, 0, 0], 'rotation': [.1, 0, 0, .995]}, {'angle': 0, 'distance': 1})

    def test_geometry_wrap_and_reordered_candidate_rejection(self):
        assert_geometry({'angle': 0, 'distance': 1}, {'angle': 2 * math.pi, 'distance': 1}, 1e-5)
        with self.assertRaises(ValueError):
            assert_geometry({'angle': 0, 'distance': 1}, {'angle': 0.5, 'distance': 1}, 1e-5)

    def test_one_scene_has_no_generalization_interval(self):
        result = cluster_ci([1, -1, 0], ['one_scene'] * 3, 100, 100)
        self.assertIsNone(result['ci95_m'])
        self.assertEqual(result['clusters'], 1)

    def test_resampling_whole_episodes_keeps_repeated_states_together(self):
        result = cluster_ci([1] * 100 + [-1] * 100, ['A'] * 100 + ['B'] * 100, 100, 1000)
        self.assertEqual(result['ci95_m'], [-1.0, 1.0])

    def test_graph_pool_is_invariant_to_alias_row_order(self):
        y = np.array([1.0, -1.0, 0.2]); scores = np.array([3.0, 3.0, 2.0])
        first = graph_pool(y, scores, ['g0', 'g0', 'g1'], 'mean')
        second = graph_pool(y[[1, 0, 2]], scores, ['g0', 'g0', 'g1'], 'mean')
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[0], [0.0, 0.2])

    def test_duplicate_decisions_are_not_silently_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'records.jsonl'
            row = json.dumps({'episode_id': '1', 'high_level_step': 0})
            path.write_text(row + '\n' + row + '\n')
            with self.assertRaises(ValueError):
                indexed_jsonl(path)


if __name__ == '__main__':
    unittest.main()
