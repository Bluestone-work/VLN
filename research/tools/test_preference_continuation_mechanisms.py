"""Tests for information boundaries and diagnostic identities."""
import copy
import unittest

from audit_preference_continuation_mechanisms import execution_symptoms, pre_action_features, reselection
from decompose_continuation_costs import breakdown


def record(step=0, act=4, ghost='g0', start=0., end=1., target=1., count=1):
    return {'high_level_step': step, 'action': {'act': act, 'ghost_vp': ghost, 'ghost_pos': [target, 0., 0.]},
            'pre_pose': {'position': [start, 0., 0.]}, 'post_pose': {'position': [end, 0., 0.]},
            'primitives': [{'pose': {'position': [start + (end-start)*(i+1)/count, 0., 0.]},
                            'collided': False} for i in range(count)]}


class DiagnosticTests(unittest.TestCase):
    def test_reselection_requires_stable_target_not_just_ghost_id(self):
        first = record()
        changed_target = record(step=1, start=1., end=2., target=2.)
        result = reselection([first, changed_target], 0, first['action'], .25)
        self.assertTrue(result['found'])
        self.assertTrue(result['immediate_next_decision'])
        self.assertFalse(result['stable_target'])
        changed_target['action']['ghost_pos'] = [1.1, 0., 0.]
        self.assertTrue(reselection([first, changed_target], 0, first['action'], .25)['stable_target'])

    def test_stop_return_movement_is_not_later_navigation(self):
        rows = [record(), record(step=1, start=1., end=2., count=2),
                record(step=2, act=0, start=2., end=0., count=3)]
        result = breakdown(rows, 0)
        self.assertEqual(result['later_navigation']['primitives'], 2)
        self.assertEqual(result['later_stop']['primitives'], 3)
        self.assertAlmostEqual(result['later_stop']['movement_m'], 2.)

    def test_stall_does_not_count_pure_rotation_as_motion(self):
        config = {'endpoint_deviation_m': .5, 'stall_primitives_min': 3, 'stall_path_m_max': .1}
        result = execution_symptoms(record(start=0., end=0., count=3), config)
        self.assertTrue(result['stall'])
        self.assertFalse(result['collision'])
        self.assertTrue(result['large_endpoint_deviation'])

    def test_future_labels_cannot_change_pre_action_features(self):
        native = {'act': 4, 'front_vp': '0', 'front_pos': [0., 0., 0.], 'ghost_pos': [1., 0., 0.], 'back_path': []}
        alternative = dict(native, ghost_pos=[2., 0., 0.])
        graph = {'high_level_step': 0, 'graph_position': [0., 0., 0.], 'options': [
            {'index': 0, 'admissible': True, 'logit': -1., 'action': {'act': 0}},
            {'index': 1, 'admissible': True, 'logit': 1., 'action': native, 'current_proposal': True},
            {'index': 2, 'admissible': True, 'logit': 0., 'action': alternative, 'current_proposal': False}]}
        event = {'baseline_index': 1, 'alternative_index': 2, 'baseline_action': native, 'alternative_action': alternative}
        expected = pre_action_features(graph, event)
        modified = copy.deepcopy(graph)
        modified.update(distance_to_goal=999., reference_path=[[99., 99., 99.]], final_success=0)
        for option in modified['options']:
            option.update(future_collision=True, final_return=-999.)
        self.assertEqual(expected, pre_action_features(modified, event))
        self.assertGreater(expected['native_probability'], expected['alternative_probability'])
        self.assertNotIn('final_success', expected)


if __name__ == '__main__':
    unittest.main()
