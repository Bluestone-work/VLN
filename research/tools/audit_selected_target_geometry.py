#!/usr/bin/env python3
"""Measure raw-proposal versus selected-ghost geometry from existing logs.

Geometry audit only: does not replace full-controller counterfactual rollouts.
Supports the yaw-only poses used by the logged baseline and fails otherwise.
"""
import argparse
import json
import math
from pathlib import Path


def proposal_position(pose, candidate):
    x, y, z, w = pose['rotation']
    if abs(x) > 1e-5 or abs(z) > 1e-5:
        raise ValueError('Pose is not yaw-only')
    yaw = 2 * math.atan2(y, w)
    bearing = yaw + candidate['angle']
    p = pose['position']
    return [p[0] - candidate['distance'] * math.sin(bearing), p[1],
            p[2] - candidate['distance'] * math.cos(bearing)]


def analyze(path):
    ghost_actions, current_actions, distances = 0, 0, []
    examples = []
    with open(str(path)) as stream:
        for line in stream:
            row = json.loads(line)
            target = row['selected_target']
            if target['type'] != 'ghost':
                continue
            ghost_actions += 1
            matches = [m['candidate_index'] for m in row['candidate_graph_mapping']
                       if m['graph_id'] == target['id'] and m['graph_valid'] and not m['graph_visited']]
            if not matches:
                continue
            current_actions += 1
            local = []
            for i in matches:
                proposed = proposal_position(row['pose'], row['waypoint_candidates'][i])
                local.append(math.sqrt(sum((a - b) ** 2 for a, b in zip(proposed, target['position']))))
            delta = min(local)
            distances.append(delta)
            examples.append({'episode_id': row['episode_id'], 'step': row['high_level_step'],
                             'graph_id': target['id'], 'nearest_raw_target_offset_m': delta,
                             'matching_proposals': len(matches)})
    return {'ghost_decisions': ghost_actions, 'selected_ghost_represented_by_current_candidate': current_actions,
            'nearest_raw_target_offset_mean_m': sum(distances) / len(distances),
            'offset_gt_1mm_count': sum(d > .001 for d in distances),
            'offset_gt_10cm_count': sum(d > .1 for d in distances),
            'offset_max_m': max(distances),
            'largest_offsets': sorted(examples, key=lambda r: r['nearest_raw_target_offset_m'], reverse=True)[:5]}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = {'interpretation': 'Euclidean geometric mismatch, not execution failure rate. Probe label uses raw proposal; action uses merged ghost, front/back path, quantized turns and tryout.',
              'sources': {p: analyze(p) for p in args.inputs}}
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result, indent=2))
