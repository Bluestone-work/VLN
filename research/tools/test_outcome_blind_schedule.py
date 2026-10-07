import copy
import unittest
from prepare_outcome_blind_schedule import choose_state, choose_alternative

def row(step):
    return {'scene_id':'s','episode_id':'e','high_level_step':step,'budget_stop':False,
            'no_vp_left':False,'effective_index':1,
            'options':[{'index':i,'admissible':True,'action':{'act':4},'graph_id':'g'+str(i),'logit':-i}
                       for i in [1,2,3,4]]}

class SelectionTests(unittest.TestCase):
    def test_order_independent_state_selection_with_step_hash(self):
        rows={i:row(i) for i in range(10)}
        chosen=choose_state(rows,42)['high_level_step']
        self.assertEqual(chosen,choose_state(dict(reversed(list(rows.items()))),42)['high_level_step'])
        self.assertGreater(len({choose_state(rows,s)['high_level_step'] for s in range(20)}),1)

    def test_no_privileged_field_affects_selection(self):
        a={i:row(i) for i in range(5)};b=copy.deepcopy(a)
        for r in b.values():r.update(distance_to_goal=-999,success=1,branch_progress=999)
        self.assertEqual(choose_state(a,42)['high_level_step'],choose_state(b,42)['high_level_step'])
        for k in ['top_logit','seeded_graph_id']:
            self.assertEqual(choose_alternative(a[0],k,42),choose_alternative(b[0],k,42))

    def test_alternatives_distinct_and_order_invariant(self):
        a=row(0);top=choose_alternative(a,'top_logit',42)
        second=choose_alternative(a,'seeded_graph_id',42)
        self.assertNotEqual(top['index'],second['index'])
        a['options'].reverse()
        self.assertEqual(second,choose_alternative(a,'seeded_graph_id',42))

    def test_missing_alternative_is_not_replaced(self):
        a=row(0);a['options']=a['options'][:2]
        self.assertIsNone(choose_alternative(a,'seeded_graph_id',42))
        a['options']=a['options'][:1]
        self.assertIsNone(choose_state({0:a},42))

if __name__=='__main__':unittest.main()
