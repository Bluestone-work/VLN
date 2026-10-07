import copy
import unittest

from check_option_noninterference import compare_metrics
from prepare_balanced_calibration import select


class CalibrationProtocolTest(unittest.TestCase):
    def test_noninterference_requires_matching_ids_keys_and_values(self):
        baseline = {'1': {'sr': 1., 'spl': .7}, '2': {'sr': 0., 'spl': 0.}}
        self.assertTrue(compare_metrics(baseline, copy.deepcopy(baseline), 1e-6)['passed'])
        for mutation in ('id', 'key', 'value', 'nonfinite'):
            changed = copy.deepcopy(baseline)
            if mutation == 'id':
                changed['3'] = changed.pop('2')
            elif mutation == 'key':
                del changed['1']['spl']
            elif mutation == 'value':
                changed['1']['spl'] += 1e-3
            else:
                changed['1']['spl'] = float('nan')
            self.assertFalse(compare_metrics(baseline, changed, 1e-6)['passed'], mutation)
        self.assertFalse(compare_metrics({}, {}, 1e-6)['passed'])

    def test_balanced_selection_is_order_invariant_and_route_unique(self):
        rows = [{'scene_id': scene, 'trajectory_id': route, 'episode_id': '{}-{}-{}'.format(scene, route, instruction)}
                for scene in ['A', 'B'] for route in range(10) for instruction in range(3)]
        first = select(rows, 3, 100)
        second = select(list(reversed(rows)), 3, 100)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 6)
        self.assertEqual(len(set((r['scene_id'], r['trajectory_id']) for r in first)), 6)
        self.assertEqual(sum(r['scene_id'] == 'A' for r in first), 3)
        self.assertEqual(sum(r['scene_id'] == 'B' for r in first), 3)

    def test_missing_scene_routes_are_not_silently_duplicated(self):
        with self.assertRaises(ValueError):
            select([{'scene_id': 'A', 'trajectory_id': 1, 'episode_id': 1}], 2, 100)


if __name__ == '__main__':
    unittest.main()
