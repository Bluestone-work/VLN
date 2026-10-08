#!/usr/bin/env python3
"""Eligibility boundaries, label noninterference, and immutable trace tests."""
import copy
import unittest
from audit_interrupt_coverage import eligible_cuts, symptoms


class CoverageTests(unittest.TestCase):
    def record(self, n=5):
        return {'action': {'act': 4, 'ghost_pos': {'__array__': [3, 0, 0]}},
                'pre_pose': {'position': [0, 0, 0]}, 'post_pose': {'position': [0, 0, 0]},
                'primitives': [{'action': 1, 'collided': True, 'pose': {'position': [0, 0, 0]}}
                               for _ in range(n)]}

    def test_all_interior_cuts_and_overlapping_stall(self):
        r = self.record(); old = copy.deepcopy(r)
        c = eligible_cuts(r, symptoms(r), ['ghost']*5)
        self.assertEqual([x['cut_primitive'] for x in c], [1, 2, 3, 4])
        self.assertEqual(c[2]['reasons'], ['collision', 'stall'])
        self.assertEqual(r, old)

    def test_backtrack_exclusion_and_streak_reset(self):
        r = self.record()
        for p in r['primitives']:
            p['collided'] = False
        c = eligible_cuts(r, symptoms(r), ['backtrack', 'backtrack', 'ghost', 'ghost', 'ghost'])
        self.assertFalse(any('stall' in x['reasons'] for x in c))

    def test_turns_do_not_trigger_stall_or_collision(self):
        r = self.record()
        for p in r['primitives']:
            p['action'] = 2
        self.assertEqual(eligible_cuts(r, symptoms(r), ['ghost']*5), [])

    def test_terminal_collision_only_is_not_cut(self):
        r = self.record()
        for i, p in enumerate(r['primitives']):
            p['collided'] = i == 4
            p['pose']['position'] = [(i+1)*0.25, 0, 0]
        self.assertEqual(eligible_cuts(r, {'collision': True, 'deviation': False, 'stall': False}, ['ghost']*5), [])

    def test_midpoint_fallback_and_collision_priority(self):
        r = self.record(4)
        for i, p in enumerate(r['primitives']):
            p['collided'] = False
            p['pose']['position'] = [0.1*(i+1), 0, 0]
        c = eligible_cuts(r, symptoms(r), ['ghost']*4)
        self.assertEqual(c, [{'cut_primitive': 2, 'reasons': ['midpoint_deviation_oracle']}])
        r['primitives'][0]['collided'] = True
        self.assertEqual(eligible_cuts(r, symptoms(r), ['ghost']*4),
                         [{'cut_primitive': 1, 'reasons': ['collision']}])

    def test_stop_unqualified_and_phase_shape(self):
        r = self.record()
        self.assertEqual(eligible_cuts(r, dict(collision=False, stall=False, deviation=False), ['ghost']*5), [])
        with self.assertRaises(AssertionError):
            eligible_cuts(r, symptoms(r), ['ghost'])
        r['action']['act'] = 0
        self.assertEqual(eligible_cuts(r, {'collision': True}, []), [])

    def test_return_labels_do_not_change_eligibility(self):
        r = self.record()
        a = eligible_cuts(r, symptoms(r), ['ghost']*5)
        r.update(native_success=True, future_return={'success': 1}, goal_distance=0.0)
        self.assertEqual(a, eligible_cuts(r, symptoms(r), ['ghost']*5))


if __name__ == '__main__':
    unittest.main()
