import copy
import unittest
from prepare_preference_replication_schedules import cost_at_full_state,random_at_full_state
from prepare_ranker_intervention_schedule import first_disagreement
from test_graph_value_features import state,packed
from graph_value_feasibility import FEATURE_NAMES
from analyze_preference_replication import point_gate

class ReplicationTests(unittest.TestCase):
    def fixture(self):
        row=state();alt=copy.deepcopy(row['options'][1]);alt.update(index=2,logit=2.0)
        alt['action']['ghost_pos']=packed([0,0,8]);row['options'].append(alt)
        row.update(visited=[False]*3,mask=[True]*3,high_level_step=2,effective_index=1,budget_stop=False,no_vp_left=False,scene_id='s',episode_id='e')
        model={'feature_names':list(FEATURE_NAMES),'mean':[0.]*13,'scale':[1.]*13,'weights_standardized':[1.]+[0.]*12}
        return row,model
    def test_matched_cost_can_abstain_without_searching_later(self):
        row,full=self.fixture();choice=first_disagreement([row],full)
        cost=copy.deepcopy(full);cost['weights_standardized']=[0.]*13;cost['weights_standardized'][7]=-1.
        matched=cost_at_full_state(choice,cost)
        self.assertEqual(matched[0]['high_level_step'],2)
        self.assertEqual(matched[2]['index'],1)
        self.assertIsNone(first_disagreement([row],cost))
        self.assertIsNone(cost_at_full_state(None,cost))
    def test_random_includes_native_and_excludes_stop(self):
        row,model=self.fixture();choice=first_disagreement([row],model)
        indices={random_at_full_state(choice,s)[2]['index'] for s in range(50)}
        self.assertEqual(indices,{1,2})
        self.assertEqual(random_at_full_state(choice,12)[2]['index'],random_at_full_state(choice,12)[2]['index'])
    def test_cost_worsening_blocks_replication(self):
        arm={'metrics':{k:{'delta':0.1,'intervention':0.95} for k in ['success','spl','ndtw','primitive_action_count']},'quality_rescue_scenes':['a','b']}
        self.assertFalse(all(point_gate(arm,0.9).values()))
        arm['metrics']['primitive_action_count']['delta']=-1.
        self.assertTrue(all(point_gate(arm,0.9).values()))
        arm['quality_rescue_scenes']=['a'];self.assertFalse(all(point_gate(arm,0.9).values()))

if __name__=='__main__':unittest.main()
