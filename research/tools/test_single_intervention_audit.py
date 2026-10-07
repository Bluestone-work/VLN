import copy
import json
import tempfile
import unittest
from pathlib import Path

from audit_single_intervention import read_rows, group_rows, paired_report


class SingleInterventionAuditTests(unittest.TestCase):
    def test_duplicate_trace_identity_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'trace.jsonl'
            row = {'scene_id': 's', 'episode_id': 'e', 'high_level_step': 0}
            path.write_text((json.dumps(row) + '\n') * 2)
            with self.assertRaises(ValueError):
                read_rows(path)

    def test_missing_prefix_early_terminal_and_budget_fail(self):
        rows = {('s', 'e', i): {'high_level_step': i, 'done': i == 1} for i in range(2)}
        self.assertEqual(len(group_rows(rows)['e']), 2)
        with self.assertRaises(ValueError):
            group_rows({('s', 'e', 1): rows[('s', 'e', 1)]})
        early = copy.deepcopy(rows)
        early[('s', 'e', 0)]['done'] = True
        with self.assertRaises(ValueError):
            group_rows(early)
        with self.assertRaises(ValueError):
            group_rows({('s', 'e', i): {'high_level_step': i, 'done': i == 15} for i in range(16)})

    def test_paired_report_keeps_both_rescue_and_harm(self):
        a = {e: {'success': float(e == 'b'), 'steps_taken': 2, 'high_level_steps': 1,
                 'path_length': 1.} for e in ['a', 'b', 'c']}
        b = copy.deepcopy(a)
        b['a']['success'], b['b']['success'] = 1., 0.
        trace = {e: [{'primitives': [{'collided': False}, {'collided': None}],
                       'post_metrics': {'collisions': {'count': 0}}}] for e in a}
        rows, groups = paired_report(a, b, {'a', 'b'}, trace, trace, 1e-6)
        self.assertEqual(groups['targeted_2']['success_rescued'], ['a'])
        self.assertEqual(groups['targeted_2']['success_lost'], ['b'])
        self.assertEqual(groups['targeted_2']['metrics']['success']['mean_delta'], 0.)
        self.assertEqual(groups['unaffected_1']['episodes'], 1)
        self.assertEqual(groups['all_3']['episodes'], 3)
        _, all_treated = paired_report(a, b, set(a), trace, trace, 1e-6)
        self.assertEqual(all_treated['unaffected_0']['metrics'], {})
        b['a']['steps_taken'] = 99
        with self.assertRaises(ValueError):
            paired_report(a, b, {'a', 'b'}, trace, trace, 1e-6)


if __name__ == '__main__':
    unittest.main()
