import copy
import unittest
import numpy as np
from prepare_route_disjoint_confirmation import route_key, record_route
from prepare_ranker_intervention_schedule import first_disagreement, admissible_moves
from graph_value_feasibility import FEATURE_NAMES
from test_graph_value_features import state, packed


class ConfirmationTests(unittest.TestCase):
    def test_sibling_instructions_and_prefix_have_same_route(self):
        a={'episode_id':'1','trajectory_id':3,'scene_id':'mp3d/a/a.glb'}
        b=dict(a, episode_id='2', scene_id='data/scene_datasets/mp3d/a/a.glb')
        self.assertEqual(route_key(a), route_key(b))
        self.assertEqual(record_route({'episode_id':'1'}, {'1':a}, 'fixture'), route_key(a))
        with self.assertRaises(ValueError): record_route({'episode_id':'9'}, {'1':a}, 'fixture')
        with self.assertRaises(ValueError): record_route({'episode_id':'1','scene_id':'mp3d/b/b.glb'}, {'1':a}, 'fixture')

    def test_first_disagreement_stop_masks_and_future_labels(self):
        row=state(); alt=copy.deepcopy(row['options'][1]); alt.update(index=2, logit=3.0)
        row['options'].append(alt)
        row.update(visited=[False]*3, mask=[True]*3, high_level_step=0, effective_index=1, budget_stop=False, no_vp_left=False)
        model={'feature_names':list(FEATURE_NAMES),'mean':[0.0]*13,'scale':[1.0]*13,'weights_standardized':[1.0]+[0.0]*12}
        self.assertEqual(first_disagreement([row],model)[2]['index'],2)
        polluted=copy.deepcopy(row); polluted.update(distance_to_goal=0,success=1,reference_path=[100])
        self.assertEqual(first_disagreement([polluted],model)[2]['index'],2)
        row['visited'][2]=True
        self.assertIsNone(first_disagreement([row],model))
        row['visited'][2]=False; row['budget_stop']=True
        self.assertIsNone(first_disagreement([row],model))
        row['budget_stop']=False; row['effective_index']=0
        self.assertIsNone(first_disagreement([row],model))

if __name__=='__main__': unittest.main()
