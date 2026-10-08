#!/usr/bin/env python3
import unittest
from prepare_interrupt_confirmation_sample import norm, route_key


class SampleIdentityTests(unittest.TestCase):
    def test_scene_normalization(self):
        self.assertEqual(norm('data/scene_datasets/mp3d/A/A.glb'), 'mp3d/A/A.glb')
        self.assertEqual(norm('mp3d/A/A.glb'), 'mp3d/A/A.glb')

    def test_route_identity_excludes_instruction_id(self):
        a = {'scene_id': 'mp3d/A/A.glb', 'trajectory_id': 7, 'episode_id': '1'}
        b = {'scene_id': 'data/scene_datasets/mp3d/A/A.glb', 'trajectory_id': '7', 'episode_id': '2'}
        self.assertEqual(route_key(a), route_key(b))


if __name__ == '__main__':
    unittest.main()
