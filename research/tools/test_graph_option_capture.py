import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from vlnce_baselines.adaptive_action.graph_option_capture import full_action, enumerate_options, GraphOptionCapture
from vlnce_baselines.adaptive_action.option_calibration import pack


def fixture():
    g = SimpleNamespace(node_pos={'0': np.zeros(3, dtype=np.float32), '1': np.ones(3, dtype=np.float32)},
                        node_stop_scores={'0': .6, '1': .2},
                        shortest_path={'1': {'0': ['1', '0'], '1': ['1']}},
                        ghost_pos={'g0': None, 'g1': None},
                        ghost_aug_pos={'g0': np.array([1.2, 1., 1.], dtype=np.float64),
                                       'g1': np.array([.2, 0., 0.], dtype=np.float64)},
                        last_candidate_vps=['g0', 'g0', '1'])
    g.front_to_ghost_dist = lambda ghost: (.2, '1' if ghost == 'g0' else '0')
    ids = [None, '0', '1', 'g0', 'g1']
    mask = [True] * 5
    visited = [False, True, True, False, False]
    logits = [.1, -np.inf, -np.inf, 3., 2.]
    return g, ids, mask, visited, logits


class GraphOptionCaptureTest(unittest.TestCase):
    def test_full_stop_uses_historical_best_node_and_back_path(self):
        g, _, _, _, _ = fixture()
        a = full_action(g, '1', None)
        self.assertEqual(a['stop_vp'], '0')
        self.assertEqual([v[0] for v in a['back_path']], ['0'])
        self.assertEqual(a['stop_pos'].dtype, np.float32)

    def test_ghost_action_preserves_dtype_and_front_path(self):
        g, _, _, _, _ = fixture()
        self.assertEqual(full_action(g, '1', 'g0')['back_path'], [])
        a = full_action(g, '1', 'g1')
        self.assertEqual(a['ghost_pos'].dtype, np.float64)
        self.assertEqual(a['front_pos'].dtype, np.float32)
        self.assertEqual([v[0] for v in a['back_path']], ['0'])

    def test_masks_and_aliases_produce_one_action_per_valid_graph_id(self):
        g, ids, mask, visited, logits = fixture()
        rows = enumerate_options(g, '1', ids, mask, visited, logits, False, False)
        self.assertEqual([r['index'] for r in rows], [0, 3, 4])
        self.assertTrue(rows[1]['current_proposal'])
        self.assertFalse(rows[2]['current_proposal'])
        self.assertEqual(sum(r['graph_id'] == 'g0' for r in rows), 1)

    def test_final_budget_allows_only_stop(self):
        g, ids, mask, visited, logits = fixture()
        rows = enumerate_options(g, '1', ids, mask, visited, logits, True, False)
        self.assertEqual([r['index'] for r in rows if r['admissible']], [0])

    def test_padding_is_excluded_and_tensor_dimensions_are_checked(self):
        g, ids, mask, visited, logits = fixture()
        rows = enumerate_options(g, '1', ids + ['padding'], mask + [False], visited + [False], logits + [-np.inf], False, False)
        self.assertEqual(len(rows), 3)
        with self.assertRaises(ValueError):
            enumerate_options(g, '1', ids, mask[:-1], visited, logits, False, False)
        with self.assertRaises(ValueError):
            enumerate_options(g, '1', ids, mask, visited, logits, False, False, np.zeros((5, 767)))

    def test_valid_embedding_and_serialization_survive_ghost_deletion(self):
        g, ids, mask, visited, logits = fixture()
        rows = enumerate_options(g, '1', ids, mask, visited, logits, False, False, np.zeros((5, 768), dtype=np.float32))
        old = json.loads(json.dumps(rows))
        del g.ghost_pos['g0']
        g.ghost_aug_pos['g0'][0] = 999
        self.assertEqual(rows, old)

    def test_exposing_visited_node_or_missing_stop_fails(self):
        g, ids, mask, visited, logits = fixture()
        with self.assertRaises(ValueError):
            enumerate_options(g, '1', ids, [False] + mask[1:], visited, logits, False, False)
        with self.assertRaises(ValueError):
            enumerate_options(g, '1', ids, mask, [False] * 5, [0.] * 5, False, False)

    def test_forced_stop_identity_checked_before_write(self):
        g, ids, mask, visited, logits = fixture()
        ep = SimpleNamespace(episode_id=1, scene_id='scene', trajectory_id=2,
                             instruction=SimpleNamespace(instruction_text='go'))
        trainer = SimpleNamespace(gmaps=[g], max_len=15, envs=SimpleNamespace(current_episodes=lambda: [ep]))
        nav = {'gmap_vp_ids': [ids], 'gmap_masks': torch.tensor([mask]), 'gmap_visited_masks': torch.tensor([visited])}
        with tempfile.TemporaryDirectory() as tmp:
            capture = GraphOptionCapture(Path(tmp) / 'capture')
            capture.prepare(trainer, 14, ['1'], [g.node_pos['1']], nav, torch.tensor([logits]),
                            np.array([3]), np.array([3]), [False], torch.zeros(1, 5, 768))
            self.assertEqual(capture.pending[0]['effective_index'], 0)
            self.assertTrue(capture.pending[0]['forced_stop'])
            with self.assertRaises(ValueError):
                capture.commit([{'action': full_action(g, '1', 'g0')}])
            capture.commit([{'action': full_action(g, '1', None)}])
            row = json.loads(capture.path.read_text())
            self.assertTrue(row['native_selected_action_exact'])


if __name__ == '__main__':
    unittest.main()
