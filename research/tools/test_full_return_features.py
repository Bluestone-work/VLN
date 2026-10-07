import copy
import unittest
import numpy as np
from audit_full_return_features import features, command_length, spearman


def packed(x):
    return {'__array__': x}


def fixture():
    options = [{'index': 0, 'logit': -3., 'admissible': True}]
    for index, target, logit in [(2, [2., 0., 0.], 4.), (3, [1., 0., 2.], 2.)]:
        options.append({'index': index, 'logit': logit, 'admissible': True,
                        'current_proposal': True, 'action': {'act': 4, 'back_path': [],
                        'front_pos': packed([0., 0., 0.]), 'ghost_pos': packed(target)}})
    return {'privileged_labels_in_features': False, 'budget_stop': False,
            'no_vp_left': False, 'options': options, 'logits': [-3., None, 4., 2.],
            'mask': [True]*4, 'visited': [False, True, False, False],
            'effective_index': 2, 'graph_position': packed([0., 0., 0.])}


class FeatureContractTests(unittest.TestCase):
    def test_privileged_fields_cannot_change_features(self):
        row = fixture()
        expected = features(row, 3)
        row.update(goal_distance=999, reference_route=['poison'], success=-1,
                   instruction='changed text not used by these features')
        for o in row['options']:
            o.update(oracle_progress=999, embedding={'__array__': [999]*768})
        self.assertEqual(features(row, 3), expected)

    def test_rigid_coordinate_transform_preserves_cost(self):
        row = fixture()
        expected = command_length(row, row['options'][2])
        angle = .81
        rotation = np.array([[np.cos(angle), 0., -np.sin(angle)], [0., 1., 0.],
                             [np.sin(angle), 0., np.cos(angle)]])
        values = [row['graph_position']]
        for o in row['options'][1:]:
            values.extend([o['action']['front_pos'], o['action']['ghost_pos']])
        for v in values:
            v['__array__'] = (rotation.dot(v['__array__']) + [14., -2., 9.]).tolist()
        self.assertAlmostEqual(command_length(row, row['options'][2]), expected)

    def test_backtrack_cost_is_not_direct_target_distance(self):
        row = fixture()
        action = row['options'][2]['action']
        action['back_path'] = [{'__tuple__': ['old', packed([-2., 0., 0.])]}]
        action['front_pos'] = packed([-2., 0., 0.])
        self.assertAlmostEqual(command_length(row, row['options'][2]), 2. + np.sqrt(13.))

    def test_mask_nonfinite_and_privileged_flags_reject(self):
        for change in ['mask', 'nonfinite', 'privileged']:
            row = fixture()
            if change == 'mask': row['mask'][3] = False
            if change == 'nonfinite': row['logits'][3] = float('nan')
            if change == 'privileged': row['privileged_labels_in_features'] = True
            with self.assertRaises(ValueError): features(row, 3)

    def test_entropy_ignores_logit_offset_and_handles_constant_correlations(self):
        row = fixture()
        expected = features(row, 3)
        row['logits'] = [x+100 if x is not None else x for x in row['logits']]
        for o in row['options']: o['logit'] += 100
        self.assertEqual(features(row, 3), expected)
        self.assertIsNone(spearman([1, 1, 1], [1, 2, 3]))


if __name__ == '__main__':
    unittest.main()
