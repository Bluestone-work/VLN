import unittest

from prepare_continuation_sample import choose_new_scenes, normalize_scene
from prepare_continuation_schedule import select_events
from continuation_labels import confirmation
from analyze_continuation_horizon import window
from analyze_prospective_continuation import aggregate


class SamplingTests(unittest.TestCase):
    def test_exclude_entire_prior_scene_and_deduplicate_route(self):
        rows = [{'scene_id': s, 'trajectory_id': r, 'episode_id': '{}{}{}'.format(s, r, i)}
                for s in ['mp3d/a/a.glb', 'mp3d/b/b.glb', 'mp3d/c/c.glb']
                for r in ['1', '2', '3'] for i in [0, 1]]
        selected = choose_new_scenes(rows, {'mp3d/a/a.glb'}, 2, 2, 42)
        self.assertEqual(len(selected), 4)
        self.assertEqual(len({(r['scene_id'], r['trajectory_id']) for r in selected}), 4)
        self.assertNotIn('mp3d/a/a.glb', {r['scene_id'] for r in selected})
        self.assertEqual(selected, choose_new_scenes(list(reversed(rows)), {'mp3d/a/a.glb'}, 2, 2, 42))

    def test_scene_prefix_is_normalized(self):
        self.assertEqual(normalize_scene('data/scene_datasets/mp3d/a/a.glb'), 'mp3d/a/a.glb')

    def test_insufficient_scenes_does_not_relax_exclusions(self):
        with self.assertRaises(ValueError):
            choose_new_scenes([{'scene_id': 'a', 'trajectory_id': 1}], {'a'}, 1, 1, 42)


class LabelTests(unittest.TestCase):
    def setUp(self):
        self.rule = {'minimum_progress_gain_m':0.25,'primitive_delta_max':0,'movement_delta_max_m':1e-5}
        self.base = {'distance_to_goal':4.,'primitive_count':12,'movement_m':2.}

    def test_progress_alone_cannot_hide_extra_cost(self):
        alt = dict(self.base, distance_to_goal=2., primitive_count=13)
        self.assertFalse(confirmation(self.base,alt,self.rule)['accepted'])
        alt.update(primitive_count=10, movement_m=2.01)
        self.assertFalse(confirmation(self.base,alt,self.rule)['accepted'])

    def test_absolute_threshold_and_finite_guard(self):
        alt = dict(self.base, distance_to_goal=3.75)
        self.assertTrue(confirmation(self.base,alt,self.rule)['accepted'])
        alt['distance_to_goal']=3.75001
        self.assertFalse(confirmation(self.base,alt,self.rule)['accepted'])
        alt['distance_to_goal']=float('nan')
        with self.assertRaises(ValueError):confirmation(self.base,alt,self.rule)

    def test_h2_label_does_not_read_future_final_metrics(self):
        alt = dict(self.base, distance_to_goal=3., final_success=0., final_primitives=999)
        first=confirmation(self.base,alt,self.rule)
        alt.update(final_success=1.,final_primitives=1)
        self.assertEqual(first,confirmation(self.base,alt,self.rule))

    def test_event_sampling_keeps_every_eligible_route_without_outcome_selection(self):
        rows=[{'scene_id':'s','episode_id':ep,'high_level_step':step,
               'route_matched_gain_m':gain,'baseline_episode_success':success}
              for ep,gain,success in [('a',.5,0),('b',.4,1),('c',.24,0)] for step in [0,1,2]]
        selected=select_events(rows,.25,42)
        self.assertEqual({r['episode_id'] for r in selected},{'a','b'})
        self.assertEqual(len(selected),2)
        changed=[dict(r,baseline_episode_success=1-r['baseline_episode_success']) for r in reversed(rows)]
        chosen2=select_events(changed,.25,42)
        self.assertEqual({(r['episode_id'],r['high_level_step']) for r in selected},
                         {(r['episode_id'],r['high_level_step']) for r in chosen2})

    def test_terminal_window_includes_stop_motion_without_fabricated_actions(self):
        record={'pre_pose':{'position':[0.,0.,0.]},'post_pose':{'position':[2.,0.,0.]},
                'primitives':[{'pose':{'position':[1.,0.,0.]}},{'pose':{'position':[2.,0.,0.]}},
                              {'pose':{'position':[2.,0.,0.]}}],
                'distance_after':3.,'done':True}
        w=window([record],0,2)
        self.assertEqual(w['actual_decisions'],1)
        self.assertEqual(w['primitive_count'],3)
        self.assertEqual(w['movement_m'],2.)
        self.assertTrue(w['terminal'])

    def test_empty_confirmation_group_is_not_zero_performance(self):
        result=aggregate([],set(),set())
        self.assertEqual(result['episodes'],0)
        self.assertIsNone(result['metrics']['success']['composed_mean'])


if __name__ == '__main__':
    unittest.main()
