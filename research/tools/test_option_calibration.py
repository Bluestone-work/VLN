"""Focused tests for the full-controller calibration contract (no simulator)."""
import copy
import json
import random
import unittest
from pathlib import Path

import numpy as np

from replay_option_calibration import compare, rotation_error
from vlnce_baselines.adaptive_action.option_calibration import pack, unpack, rng_snapshot, restore_rng


class OptionCalibrationTest(unittest.TestCase):
    def setUp(self):
        self.gates = json.loads(Path('research/configs/OPTION_CALIBRATION_001.json').read_text())['gates']
        pose = {'position': [0., 0., 0.], 'rotation': [0., 0., 0., 1.]}
        self.record = {'pre_pose': copy.deepcopy(pose), 'post_pose': copy.deepcopy(pose),
                       'distance_before': 2., 'distance_after': 1., 'post_rng': {'state': 100},
                       'done': False, 'observation_hashes': {'rgb': 'abc'},
                       'primitives': [{'action': 1, 'collided': False, 'pose': copy.deepcopy(pose)}]}

    def test_typed_full_option_roundtrip_preserves_precision_and_path(self):
        action = {'act': 4, 'front_pos': np.array([.1, .2, .3], dtype=np.float32),
                  'ghost_pos': np.array([.10000000001, .2, .3], dtype=np.float64),
                  'back_path': [('old_node', np.array([1., 2., 3.], dtype=np.float32))],
                  'tryout': False, 'score': np.float32(.25)}
        restored = unpack(json.loads(json.dumps(pack(action))))
        self.assertEqual(restored['front_pos'].dtype, np.float32)
        self.assertEqual(restored['ghost_pos'].dtype, np.float64)
        self.assertIsInstance(restored['back_path'][0], tuple)
        self.assertIsInstance(restored['score'], np.float32)
        self.assertEqual(pack(restored), pack(action))

    def test_stop_and_empty_or_teleport_paths_remain_distinct(self):
        for path in (None, [], [('n2', np.array([1, 2, 3], dtype=np.float32))]):
            action = {'act': 0, 'stop_pos': np.zeros(3, dtype=np.float32), 'back_path': path, 'tryout': False}
            self.assertEqual(pack(unpack(json.loads(json.dumps(pack(action))))), pack(action))

    def test_python_numpy_rng_roundtrip(self):
        original = rng_snapshot()
        try:
            random.seed(37)
            np.random.seed(42)
            before = json.loads(json.dumps(rng_snapshot()))
            expected = [random.random(), np.random.random(), random.randint(0, 999)]
            restore_rng(before)
            self.assertEqual(expected, [random.random(), np.random.random(), random.randint(0, 999)])
        finally:
            restore_rng(original)

    def test_exact_replay_passes_and_quaternion_sign_is_equivalent(self):
        self.assertTrue(compare(self.record, copy.deepcopy(self.record), self.gates)['passed'])
        self.assertEqual(rotation_error([0, 0, 0, 1], [0, 0, 0, -1]), 0)

    def test_endpoint_and_progress_mismatches_fail(self):
        for kind in ('endpoint', 'pre_pose', 'progress', 'rotation'):
            actual = copy.deepcopy(self.record)
            if kind == 'endpoint':
                actual['post_pose']['position'][0] = .01
            elif kind == 'pre_pose':
                actual['pre_pose']['position'][0] = .01
            elif kind == 'progress':
                actual['distance_after'] += .01
            else:
                actual['post_pose']['rotation'] = [0, .1, 0, np.sqrt(.99)]
            self.assertFalse(compare(self.record, actual, self.gates)['passed'], kind)

    def test_action_collision_rng_and_terminal_mismatches_fail(self):
        for kind in ('action', 'collision', 'rng', 'done', 'length'):
            actual = copy.deepcopy(self.record)
            if kind == 'action':
                actual['primitives'][0]['action'] = 2
            elif kind == 'collision':
                actual['primitives'][0]['collided'] = True
            elif kind == 'rng':
                actual['post_rng']['state'] = 101
            elif kind == 'done':
                actual['done'] = True
            else:
                actual['primitives'].append(copy.deepcopy(actual['primitives'][0]))
            self.assertFalse(compare(self.record, actual, self.gates)['passed'], kind)

    def test_stop_marker_is_counted_without_cached_collision(self):
        self.record['primitives'].append({'action': 0, 'collided': None, 'pose': self.record['post_pose']})
        self.record['done'] = True
        actual = copy.deepcopy(self.record)
        self.assertTrue(compare(self.record, actual, self.gates)['passed'])
        actual['primitives'][-1]['collided'] = False
        self.assertFalse(compare(self.record, actual, self.gates)['passed'])

    def test_sensor_hash_difference_is_reported_separately(self):
        actual = copy.deepcopy(self.record)
        actual['observation_hashes']['rgb'] = 'different'
        result = compare(self.record, actual, self.gates)
        self.assertTrue(result['passed'])
        self.assertFalse(result['sensor_hashes_exact'])


if __name__ == '__main__':
    unittest.main()
