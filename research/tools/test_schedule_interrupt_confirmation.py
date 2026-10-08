#!/usr/bin/env python3
import unittest
from schedule_interrupt_confirmation import keyed


class ScheduleTests(unittest.TestCase):
    def test_seeded_order_is_deterministic(self):
        row = {'scene_id': 's', 'episode_id': 'e', 'high_level_step': 2}
        self.assertEqual(keyed(20261011, row, 4), keyed(20261011, row, 4))
        self.assertNotEqual(keyed(20261011, row, 4), keyed(20261012, row, 4))


if __name__ == '__main__': unittest.main()
