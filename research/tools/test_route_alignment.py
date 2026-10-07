import unittest
import numpy as np
from route_alignment import unique_path, initial_row, extend_row, open_endpoint, route_gate, best_option
from prepare_option_expansion import choose


def xyz(x):
    return np.asarray([[v, 0., 0.] for v in x])


class RouteAlignmentTests(unittest.TestCase):
    def test_stationary_dedup_and_revisits(self):
        np.testing.assert_array_equal(unique_path(xyz([0, 0, 1, 1, 0])), xyz([0, 1, 0]))

    def test_exact_dtw_against_exhaustive_monotonic_paths(self):
        path, reference = xyz([0, .4, 1.8]), xyz([0, 1, 2])
        def enumerate_cost(i, j):
            cost = np.linalg.norm(path[i] - reference[j])
            if i == 0 and j == 0:
                return [cost]
            predecessors = [(a, b) for a, b in [(i - 1, j), (i, j - 1), (i - 1, j - 1)] if a >= 0 and b >= 0]
            return [cost + previous for a, b in predecessors for previous in enumerate_cost(a, b)]
        row = extend_row(initial_row(path[0], reference), path[1:], reference)
        for j in range(len(reference)):
            self.assertAlmostEqual(row[j], min(enumerate_cost(len(path) - 1, j)))

    def test_incremental_matches_whole_prefix(self):
        ref, path = xyz(range(8)), xyz([0, 1, 2, 1, 2, 3, 4])
        start = initial_row(path[0], ref)
        whole = extend_row(start, path[1:], ref)
        prefix = extend_row(start, path[1:4], ref)
        np.testing.assert_array_equal(whole, extend_row(prefix, path[4:], ref))

    def test_identical_branch_passes_and_no_mutation(self):
        ref = xyz([0, 1, 2, 3])
        before = initial_row(ref[0], ref)
        copy = before.copy()
        after = extend_row(before, ref[1:3], ref)
        self.assertTrue(route_gate(after, after, 0, 1e-5)['matched_prefix_route_gate'])
        np.testing.assert_array_equal(before, copy)

    def test_route_loop_does_not_jump_to_nearest_late_point(self):
        ref = xyz([0, 1, 2, 1, 0])
        row = initial_row(ref[0], ref)
        self.assertEqual(open_endpoint(row, 0), 0)
        self.assertEqual(open_endpoint(extend_row(row, xyz([1]), ref), 0), 1)

    def test_monotonic_cursor_and_earliest_tie(self):
        self.assertEqual(open_endpoint(np.array([0., 2., 2., 3.]), 1), 1)

    def test_shortcut_penalized_despite_same_endpoint(self):
        ref = np.array([[0., 0, 0], [0, 0, 1], [1, 0, 1], [1, 0, 0]])
        base = initial_row(ref[0], ref)
        correct = extend_row(base, ref[1:], ref)
        shortcut = extend_row(base, ref[3:], ref)
        gate = route_gate(shortcut, correct, 0, 1e-5)
        self.assertFalse(gate['matched_cost_gate'])
        self.assertGreater(gate['matched_endpoint_cost_delta'], 0.)

    def test_backward_motion_fails_progress_gate(self):
        ref = xyz(range(5))
        base = extend_row(initial_row(ref[0], ref), xyz([1]), ref)
        correct = extend_row(base, xyz([2, 3]), ref)
        back = extend_row(base, xyz([0]), ref)
        self.assertFalse(route_gate(back, correct, 1, 1e-5)['endpoint_gate'])

    def test_required_detour_can_match_ordered_route(self):
        ref = np.array([[0., 0, 0], [-1, 0, 0], [-1, 0, 1], [1, 0, 1]])
        base = initial_row(ref[0], ref)
        detour = extend_row(base, ref[1:3], ref)
        self.assertEqual(open_endpoint(detour, 0), 2)
        self.assertEqual(detour[2], 0.)

    def test_deterministic_goal_ties_prefer_selected(self):
        options = [{'index': 2, 'progress_m': 1.}, {'index': 3, 'progress_m': 1.}]
        self.assertEqual(best_option(options, 3)['index'], 3)

    def test_invalid_path_rejected(self):
        with self.assertRaises(ValueError):
            unique_path([[0, 0, float('nan')]])

    def test_metadata_sample_invariant_to_order_and_excludes_pilot(self):
        episodes = [{'scene_id': scene, 'trajectory_id': route, 'episode_id': '{}{}{}'.format(scene, route, ins)}
                    for scene in ['a', 'b', 'c'] for route in range(4) for ins in range(2)]
        excluded = {('a', '0')}
        a = choose(episodes, excluded, 2, 2, 42)
        b = choose(list(reversed(episodes)), excluded, 2, 2, 42)
        self.assertEqual(a, b)
        self.assertEqual(len({(e['scene_id'], e['trajectory_id']) for e in a}), 4)
        self.assertFalse({(e['scene_id'], str(e['trajectory_id'])) for e in a} & excluded)

    def test_native_float32_path_reconstruction(self):
        from analyze_route_options import prepare_prefixes
        ref = xyz([i * .23117 for i in range(171)]).astype(np.float32)
        trace = {'pre_pose': {'position': ref[0].tolist()}, 'post_pose': {'position': ref[-1].tolist()},
                 'primitives': [{'pose': {'position': p.tolist()}} for p in ref[1:]], 'done': True}
        expected = float(np.linalg.norm(ref[1:] - ref[:-1], axis=1).sum())
        native = {'e': {'path_length': expected, 'ndtw': 1., 'sdtw': 1., 'success': 1.}}
        _, checks = prepare_prefixes({('s', 'e', 0): trace}, {'e': {'locations': ref.tolist()}}, native, 0.)
        self.assertTrue(checks[0]['passed'])


if __name__ == '__main__':
    unittest.main()
