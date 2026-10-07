import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from test_graph_option_capture import fixture
from vlnce_baselines.adaptive_action.graph_option_capture import GraphOptionCapture, full_action
from vlnce_baselines.adaptive_action.option_calibration import pack
from vlnce_baselines.adaptive_action.single_intervention import SingleInterventionHook


class SingleInterventionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        g, ids, mask, visited, logits = fixture()
        self.g, self.ids = g, ids
        ep = SimpleNamespace(episode_id='1', scene_id='scene', trajectory_id='2',
                             instruction=SimpleNamespace(instruction_text='go'))
        self.trainer = SimpleNamespace(gmaps=[g], max_len=15,
            config=SimpleNamespace(MODEL=SimpleNamespace(consume_ghost=True)),
            envs=SimpleNamespace(current_episodes=lambda: [ep]))
        self.nav = {'gmap_vp_ids': [ids], 'gmap_masks': torch.tensor([mask]),
                    'gmap_visited_masks': torch.tensor([visited])}
        self.logits = torch.tensor([logits])
        capture = GraphOptionCapture(Path(self.tmp.name) / 'reference')
        capture.prepare(self.trainer, 2, ['1'], [g.node_pos['1']], self.nav,
                        self.logits, np.array([3]), np.array([3]), [False], None)
        reference = capture.pending[0]
        self.baseline = {('scene', '1', 2): reference}
        self.event = {'episode_id': '1', 'scene_id': 'scene', 'high_level_step': 2,
            'baseline_index': 3, 'alternative_index': 4,
            'baseline_action': pack(full_action(g, '1', 'g0')),
            'alternative_action': pack(full_action(g, '1', 'g1')),
            'expected_graph_ids': ids, 'expected_graph_position': pack(g.node_pos['1'])}

    def tearDown(self):
        self.tmp.cleanup()

    def hook(self, enabled):
        return SingleInterventionHook({'events': [self.event]}, self.baseline,
                                      Path(self.tmp.name) / 'hook', enabled)

    def prepare(self, hook, chosen=None, step=2):
        chosen = np.array([3]) if chosen is None else chosen
        original = chosen.copy()
        hook.prepare(self.trainer, step, ['1'], [self.g.node_pos['1']], self.nav,
                     self.logits, chosen, original, [False], None)
        return chosen, original

    def commit(self, hook, graph_id):
        action = full_action(self.g, '1', graph_id)
        del self.g.ghost_pos[graph_id]
        hook.commit([{'action': action}])

    def test_disabled_retains_native_action_and_exact_identity(self):
        hook = self.hook(False)
        chosen, original = self.prepare(hook)
        self.assertEqual(chosen.tolist(), original.tolist())
        self.commit(hook, 'g0')
        hook.finish()
        self.assertEqual(hook.applied, set())

    def test_enabled_changes_only_index_and_consumes_correct_ghost(self):
        hook = self.hook(True)
        chosen, original = self.prepare(hook)
        self.assertEqual(chosen.tolist(), [4])
        self.assertEqual(original.tolist(), [3])
        self.commit(hook, 'g1')
        self.assertIn('g0', self.g.ghost_pos)
        hook.finish()
        self.assertEqual(hook.applied, {'1'})

    def test_continuation_uses_new_native_choice_without_archived_suffix(self):
        hook = self.hook(True)
        self.prepare(hook)
        self.commit(hook, 'g1')
        self.nav['gmap_masks'][0, 4] = False
        chosen, original = self.prepare(hook, step=3)
        self.assertEqual(chosen.tolist(), original.tolist())
        self.commit(hook, 'g0')
        hook.finish()
        with hook.log.open() as stream:
            rows = [json.loads(l) for l in stream]
        self.assertEqual([r['intervened'] for r in rows], [True, False])
        self.assertFalse(rows[-1]['baseline_prefix_verified'])

    def test_invalid_alternative_mask_rejected(self):
        hook = self.hook(True)
        self.nav['gmap_masks'][0, 4] = False
        with self.assertRaises(ValueError):
            self.prepare(hook)

    def test_stop_cannot_be_overridden(self):
        hook = self.hook(True)
        with self.assertRaises(ValueError):
            self.prepare(hook, np.array([0]))

    def test_horizon_and_changed_prefix_rejected(self):
        hook = self.hook(True)
        self.trainer.max_len = 16
        with self.assertRaises(ValueError):
            self.prepare(hook)
        self.trainer.max_len = 15
        self.logits[0, 3] += .01
        with self.assertRaises(ValueError):
            self.prepare(hook)

    def test_action_construction_mismatch_rejected(self):
        hook = self.hook(True)
        self.prepare(hook)
        with self.assertRaises(ValueError):
            hook.commit([{'action': full_action(self.g, '1', 'g0')}])

    def test_missing_ghost_consumption_rejected(self):
        hook = self.hook(True)
        self.prepare(hook)
        with self.assertRaises(ValueError):
            hook.commit([{'action': full_action(self.g, '1', 'g1')}])

    def test_changed_alternate_typed_action_rejected(self):
        self.event['alternative_action'] = copy.deepcopy(self.event['alternative_action'])
        self.event['alternative_action']['ghost_pos']['dtype'] = '<f4'
        hook = self.hook(True)
        with self.assertRaises(ValueError):
            self.prepare(hook)

    def test_missing_and_duplicate_schedule_rejected(self):
        hook = self.hook(True)
        with self.assertRaises(ValueError):
            hook.finish()
        with self.assertRaises(ValueError):
            SingleInterventionHook({'events': [self.event, self.event]}, self.baseline,
                                   Path(self.tmp.name) / 'duplicate', True)


if __name__ == '__main__':
    unittest.main()
