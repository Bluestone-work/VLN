"""Offline ordered-prefix diagnostics; no simulator, policy or training inputs."""
import numpy as np


def unique_path(points):
    result = []
    for point in points:
        point = np.asarray(point, dtype=np.float64)
        if point.shape != (3,) or not np.all(np.isfinite(point)):
            raise ValueError('Expected a finite 3D position')
        if not result or not np.array_equal(point, result[-1]):
            result.append(point)
    if not result:
        raise ValueError('Empty path')
    return np.asarray(result)


def trace_path(trace):
    points = [trace['pre_pose']['position']] + [p['pose']['position'] for p in trace['primitives']]
    if points[-1] != trace['post_pose']['position']:
        raise ValueError('Unrecorded controller motion')
    return unique_path(points)


def initial_row(start, reference):
    return np.cumsum(np.linalg.norm(reference - np.asarray(start), axis=1))


def extend_row(prefix_row, points, reference):
    """Extend exact DTW from a fixed prefix. points EXCLUDES shared starting pose."""
    previous = np.asarray(prefix_row).copy()
    for point in points:
        cost = np.linalg.norm(reference - point, axis=1)
        current = np.empty_like(previous)
        current[0] = cost[0] + previous[0]
        for j in range(1, len(previous)):
            current[j] = cost[j] + min(previous[j], previous[j - 1], current[j - 1])
        previous = current
    return previous


def open_endpoint(row, cursor):
    if cursor < 0 or cursor >= len(row) or not np.all(np.isfinite(row)):
        raise ValueError('Invalid DP row or monotonic cursor')
    return int(cursor + np.argmin(row[cursor:]))


def route_gate(branch_row, selected_row, cursor, tolerance):
    selected_end = open_endpoint(selected_row, cursor)
    endpoint = open_endpoint(branch_row, cursor)
    endpoint_ok = endpoint >= selected_end
    cost_delta = float(branch_row[selected_end] - selected_row[selected_end])
    cost_ok = cost_delta <= tolerance
    return {'reference_endpoint': endpoint, 'selected_reference_endpoint': selected_end,
            'matched_endpoint_cost': float(branch_row[selected_end]),
            'selected_matched_endpoint_cost': float(selected_row[selected_end]),
            'matched_endpoint_cost_delta': cost_delta,
            'endpoint_gate': bool(endpoint_ok), 'matched_cost_gate': bool(cost_ok),
            'matched_prefix_route_gate': bool(endpoint_ok and cost_ok)}


def best_option(options, selected_index):
    if not options:
        raise ValueError('No admissible options')
    return max(options, key=lambda o: (o['progress_m'], o['index'] == selected_index, -o['index']))
