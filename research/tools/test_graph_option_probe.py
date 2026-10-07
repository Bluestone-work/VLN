import copy
import unittest

from analyze_graph_options import analyze_state
from probe_graph_options import validate_join
from vlnce_baselines.adaptive_action.option_calibration import pack


class GraphOptionProbeTest(unittest.TestCase):
    def row(self):
        def action(index, progress, primitives, path, current):
            return {'index': index, 'graph_id': 'g{}'.format(index), 'action_type': 4,
                    'progress_m': progress, 'primitive_events': primitives, 'motion_path_m': path,
                    'current_proposal': current, 'collision_events': 0, 'target_error_horizontal_m': 0.}
        return {'scene_id': 's', 'episode_id': 'e', 'trajectory_id': 'r', 'high_level_step': 2,
                'selected_sentinel_passed': True, 'effective_index': 1, 'budget_stop': False,
                'forced_stop': False, 'options': [action(1, .5, 5, 1., True), action(2, 3., 20, 4., False),
                                                action(3, 1., 4, .8, True)]}

    def test_cost_controls_remove_long_action_artificial_advantage(self):
        state = analyze_state(self.row(), 1e-5)
        self.assertEqual(state['gain_m'], 2.5)
        self.assertEqual(state['primitive_capped_gain_m'], .5)
        self.assertEqual(state['path_capped_gain_m'], .5)
        self.assertEqual(state['both_costs_capped_gain_m'], .5)
        self.assertEqual(state['current_plus_selected_gain_m'], .5)
        self.assertEqual(state['historical_extra_gain_m'], 2.)

    def test_actual_stop_states_are_excluded_from_rank_only_gain(self):
        row = self.row()
        row['options'][0]['action_type'] = 0
        self.assertIsNone(analyze_state(row, 1e-5))
        row['options'][0]['action_type'] = 4
        row['budget_stop'] = True
        with self.assertRaises(ValueError):
            analyze_state(row, 1e-5)

    def test_no_duplicate_graph_actions_or_failed_sentinels(self):
        row = self.row()
        row['options'][1]['graph_id'] = row['options'][0]['graph_id']
        with self.assertRaises(ValueError):
            analyze_state(row, 1e-5)
        row = self.row()
        row['selected_sentinel_passed'] = False
        with self.assertRaises(ValueError):
            analyze_state(row, 1e-5)

    def test_join_rejects_missing_actions_wrong_pose_and_budget(self):
        action = {'act': 4, 'ghost_vp': 'g0'}
        graph = {'scene_id': 's', 'episode_id': 'e', 'high_level_step': 0,
                 'mask': [True, True, True], 'visited': [False, True, False],
                 'graph_position': pack([0., 0., 0.]), 'effective_index': 2,
                 'budget_stop': False, 'no_vp_left': False,
                 'options': [{'index': 0, 'graph_id': None, 'admissible': True, 'action': {'act': 0}},
                             {'index': 2, 'graph_id': 'g0', 'admissible': True, 'action': action}]}
        trace = {'scene_id': 's', 'episode_id': 'e', 'high_level_step': 0,
                 'pre_pose': {'position': [0., 0., 0.]}, 'action': action}
        validate_join(graph, trace, {'endpoint_tolerance_m': 1e-5})
        for mutation in ('missing', 'pose', 'budget', 'identity'):
            g, t = copy.deepcopy(graph), copy.deepcopy(trace)
            if mutation == 'missing':
                g['options'].pop(0)
            elif mutation == 'pose':
                t['pre_pose']['position'][0] = 1.
            elif mutation == 'budget':
                g['budget_stop'] = True
            else:
                t['episode_id'] = 'different'
            with self.assertRaises(ValueError):
                validate_join(g, t, {'endpoint_tolerance_m': 1e-5})


if __name__ == '__main__':
    unittest.main()
