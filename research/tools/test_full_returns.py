import unittest
from unittest.mock import patch
from analyze_full_returns import classify, DIRECTIONS
from analyze_continuation_horizon import window


class FullReturnTests(unittest.TestCase):
    def vector(self, **changes):
        result = dict.fromkeys(DIRECTIONS, 0.)
        result.update(changes)
        return result

    def test_direction_of_each_metric(self):
        for key, direction in DIRECTIONS.items():
            self.assertEqual(classify(self.vector(**{key: direction})), 'dominates')
            self.assertEqual(classify(self.vector(**{key: -direction})), 'dominated')

    def test_no_reward_tradeoff_hides_success_loss(self):
        self.assertEqual(classify(self.vector(success=-1, primitive_action_count=-100)), 'mixed')
        self.assertEqual(classify(self.vector(success=1, path_length=1)), 'mixed')
        self.assertEqual(classify(self.vector(primitive_action_count=-1, distance_to_goal=0.01)), 'mixed')

    def test_tolerance_and_exact_integer_cost(self):
        self.assertEqual(classify(self.vector(ndtw=1e-7)), 'tied')
        self.assertEqual(classify(self.vector(ndtw=1e-7, primitive_action_count=1)), 'dominated')

    def test_invalid_labels_rejected(self):
        for value in [float('nan'), float('inf'), -float('inf')]:
            with self.assertRaises(ValueError):
                classify(self.vector(ndtw=value))
        with self.assertRaises(ValueError):
            classify({'success': 1})
        with self.assertRaises(ValueError):
            classify(self.vector(primitive_action_count=0.5))

    def test_terminal_absorbing_window_counts_stop_once(self):
        records = [dict(distance_after=3, primitives=[1, 2], done=False),
                   dict(distance_after=1, primitives=[1, 2, 3], done=True)]
        with patch('analyze_continuation_horizon.movement', return_value=0.5):
            result = window(records, 0, 3)
            self.assertEqual(result['primitive_count'], 5)
            self.assertEqual(result['movement_m'], 1.)
            self.assertEqual(result['actual_decisions'], 2)
            self.assertTrue(result['terminal'])
            self.assertEqual(window(records, 1, 2)['primitive_count'], 3)


if __name__ == '__main__':
    unittest.main()
