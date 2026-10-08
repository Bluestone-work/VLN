#!/usr/bin/env python3
import unittest


class PlanInvariantTests(unittest.TestCase):
    def test_expected_arm_count(self):
        routes, event_cuts, uniform_cuts, arms = 128, 43, 129, 2
        self.assertEqual(routes + (event_cuts + uniform_cuts) * arms, 472)
        self.assertEqual(uniform_cuts, event_cuts * 3)


if __name__ == '__main__': unittest.main()
