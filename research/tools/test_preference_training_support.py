import copy
import unittest
from audit_preference_training_support import chosen_index, pair_kind, relation
from graph_value_feasibility import FEATURE_NAMES


class SupportTests(unittest.TestCase):
    def setUp(self):
        self.metrics = dict(success=0., spl=0., ndtw=.5, distance_to_goal=4., path_length=10., primitive_action_count=20)
        self.model = {'feature_names': list(FEATURE_NAMES), 'weights_standardized': [0.]*len(FEATURE_NAMES),
                      'mean': [0.]*len(FEATURE_NAMES), 'scale': [1.]*len(FEATURE_NAMES)}
        action = dict(act=4, ghost_pos=[1.,0.,0.], front_pos=[0.,0.,0.], back_path=[])
        self.graph = {'effective_index': 2, 'high_level_step': 0, 'budget_stop': False, 'no_vp_left': False,
                      'graph_position': [0.,0.,0.], 'visited': [False,False,False], 'mask': [True,True,True],
                      'options': [dict(index=0,admissible=True,logit=-1.,action={'act':0}),
                                  dict(index=1,admissible=True,logit=0.,action=action),
                                  dict(index=2,admissible=True,logit=1.,action=action)]}

    def test_success_with_route_quality_loss_is_mixed(self):
        changed = dict(self.metrics, success=1., spl=.6, ndtw=.4, distance_to_goal=1.)
        self.assertEqual(relation(changed, self.metrics), 'mixed')

    def test_tolerance_and_integer_cost_tie(self):
        self.assertEqual(relation(dict(self.metrics, ndtw=.5000001), self.metrics), 'tie')
        self.assertEqual(relation(dict(self.metrics, primitive_action_count=19), self.metrics), 'strict_improvement')
        self.assertEqual(relation(dict(self.metrics, primitive_action_count=21), self.metrics), 'strict_degradation')

    def test_stop_and_budget_are_protected(self):
        self.graph['effective_index'] = 0
        self.assertEqual(chosen_index(self.graph,self.model),(0,'protected_stop_or_budget'))
        self.graph['effective_index'] = 2; self.graph['budget_stop'] = True
        self.assertEqual(chosen_index(self.graph,self.model),(2,'protected_stop_or_budget'))

    def test_original_index_tie_break_and_visited_mask(self):
        self.assertEqual(chosen_index(self.graph,self.model),(1,'changed'))
        self.graph['visited'][1] = True
        self.assertEqual(chosen_index(self.graph,self.model),(2,'native_abstention'))
        self.graph['visited'][1] = False; self.graph['mask'][1] = False
        self.assertEqual(chosen_index(self.graph,self.model),(2,'native_abstention'))

    def test_future_labels_do_not_affect_choice(self):
        before = chosen_index(self.graph,self.model)
        altered = copy.deepcopy(self.graph)
        altered.update(goal_distance=0., final_success=1, reference_path=['privileged'])
        for option in altered['options']: option['full_return_success'] = 1
        self.assertEqual(chosen_index(altered,self.model),before)

    def test_stop_native_overlap_is_not_double_counted_in_partition(self):
        self.assertEqual(pair_kind(True,False,0,2,0),'stop_involving')
        self.assertEqual(pair_kind(False,False,1,2,2),'native_move')
        self.assertEqual(pair_kind(False,False,1,3,2),'two_non_native_moves')


if __name__ == '__main__': unittest.main()
