import json
import tempfile
import unittest
from pathlib import Path

from audit_unresolved_coverage import _route_summary


class UnresolvedCoverageTest(unittest.TestCase):
    def test_counts_non_stop_branches_and_native_symptoms(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            summary = root / "summary.json"
            run = root / "run" / "cases" / "control_1"
            run.mkdir(parents=True)
            summary.write_text(json.dumps({"routes": {"1": {"scene_id": "scene", "unresolved_after_tested_interventions": True}}}))
            (run / "result.json").write_text(json.dumps({"case": {"episode_id": "1", "mode": "control"}, "metrics": {"forced_stop_count": 1}, "primitive_collision_events": 2}))
            for idx, act in enumerate([0, 4, 4]):
                case = root / "run" / "cases" / ("e1_s0_a%d" % idx)
                case.mkdir()
                (case / "result.json").write_text(json.dumps({"case": {"episode_id": "1", "mode": "action", "high_level_step": 0, "action_index": idx, "action": {"act": act, "ghost_vp": "g%d" % idx}, "reasons": ["collision"]}, "metrics": {"success": 0, "ndtw": 0}, "baseline_metrics": {"ndtw": 0}, "primitive_collision_events": 1}))
            result = _route_summary(summary, root / "run")
            route = result["routes"][0]
            self.assertEqual(route["full_return_branch_count"], 3)
            self.assertEqual(route["full_return_nonstop_branch_count"], 2)
            self.assertTrue(route["has_nonstop_branch"])
            self.assertTrue(route["native_forced_stop"])


if __name__ == "__main__":
    unittest.main()
