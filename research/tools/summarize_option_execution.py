#!/usr/bin/env python3
"""Observed full-option symptoms only; no causal failure or alternative labels."""
import argparse
import collections
import hashlib
import json
from pathlib import Path

import numpy as np


def summary(values):
    a = np.asarray(values, dtype=np.float64)
    if not len(a):
        return {'count': 0}
    return {'count': len(a), 'mean': float(a.mean()), 'median': float(np.median(a)),
            'p90': float(np.percentile(a, 90)), 'max': float(a.max())}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--capture-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((args.capture_dir / 'manifest.json').read_text())
    sources = list((args.capture_dir / 'traces').glob('worker_seed*.jsonl'))
    if len(sources) != 1:
        raise ValueError('Expected one worker')
    paths = list((Path('data/logs/eval_results') / manifest['exp_name']).glob('stats_ep_*.json'))
    if len(paths) != 1:
        raise ValueError('Expected one episode result')
    metrics = json.loads(paths[0].read_text())
    observed = []
    with sources[0].open() as stream:
        for line in stream:
            r = json.loads(line)
            action = r['action']
            target = action['stop_pos' if action['act'] == 0 else 'ghost_pos']['__array__']
            residual = np.asarray(r['post_pose']['position']) - np.asarray(target)
            primitive_positions = [np.asarray(r['pre_pose']['position'])] + [np.asarray(v['pose']['position']) for v in r['primitives']]
            actual_path_length = sum(float(np.linalg.norm(b - a)) for a, b in zip(primitive_positions, primitive_positions[1:]))
            observed.append({'episode_id': r['episode_id'], 'high_level_step': r['high_level_step'],
                             'scene_id': r['scene_id'], 'action_type': action['act'],
                             'back_path_nodes': len(action.get('back_path') or []),
                             'target_error_3d_m': float(np.linalg.norm(residual)),
                             'target_error_horizontal_m': float(np.linalg.norm(residual[[0, 2]])),
                             'progress_m': r['distance_before'] - r['distance_after'],
                             'distance_before': r['distance_before'], 'distance_after': r['distance_after'],
                             'primitive_events': len(r['primitives']),
                             'collision_events': sum(x['collided'] is True for x in r['primitives']),
                             'motion_path_length_m': actual_path_length,
                             'stop_lost_goal_neighborhood': action['act'] == 0 and r['distance_before'] < 3 <= r['distance_after']})
    grouped = collections.defaultdict(list)
    for row in observed:
        grouped[row['episode_id']].append(row)
    if set(grouped) != set(metrics):
        raise ValueError('Episode mismatch between trace and metrics')
    # This is an observed cost check, not a change to the benchmark definition.
    for episode, rows in grouped.items():
        if sum(r['primitive_events'] for r in rows) != metrics[episode]['steps_taken']:
            raise ValueError('Primitive event count mismatch in ' + episode)
    by_action = {}
    for name, act in [('ghost', 4), ('stop', 0)]:
        rows = [r for r in observed if r['action_type'] == act]
        by_action[name] = {'options': len(rows), 'options_with_collision': sum(r['collision_events'] > 0 for r in rows),
                           'collision_events': sum(r['collision_events'] for r in rows),
                           'nonpositive_progress': sum(r['progress_m'] <= 0 for r in rows),
                           'progress_m': summary([r['progress_m'] for r in rows]),
                           'target_error_3d_m': summary([r['target_error_3d_m'] for r in rows]),
                           'target_error_horizontal_m': summary([r['target_error_horizontal_m'] for r in rows]),
                           'target_error_horizontal_threshold_counts': {str(t): sum(r['target_error_horizontal_m'] <= t for r in rows) for t in [.25, .5, 1.]}}
    failed_ids = sorted(k for k in grouped if metrics[k]['success'] == 0)
    failures = {'failed_episodes': len(failed_ids),
                'with_any_collision': sum(any(r['collision_events'] > 0 for r in grouped[k]) for k in failed_ids),
                'with_forced_stop': sum(metrics[k]['forced_stop_count'] > 0 for k in failed_ids),
                'with_native_oracle_success': sum(metrics[k]['oracle_success'] > 0 for k in failed_ids),
                'with_stop_lost_goal_neighborhood': sum(any(r['stop_lost_goal_neighborhood'] for r in grouped[k]) for k in failed_ids),
                'categories_overlap': True, 'causal_failure_counts': False}
    # Select examples by a declared observable, not by a favorable research outcome.
    examples = sorted(observed, key=lambda r: (-r['target_error_horizontal_m'], r['episode_id'], r['high_level_step']))[:5]
    result = {'experiment_id': manifest['config']['experiment_id'], 'episodes': len(grouped),
              'scenes': len(set(r['scene_id'] for r in observed)), 'options': len(observed),
              'primitive_events': sum(r['primitive_events'] for r in observed), 'by_action': by_action,
              'failure_symptoms': failures, 'largest_target_error_examples': examples,
              'scope': 'Observed selected-option geometry and events; residual does not prove an appropriate target or execution failure. Negative progress can be valid recovery.',
              'sources': {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in [sources[0], paths[0]]},
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (args.output_dir / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    with (args.output_dir / 'options.jsonl').open('x') as stream:
        for row in observed:
            stream.write(json.dumps(row, sort_keys=True) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'largest_target_error_examples'}, indent=2))


if __name__ == '__main__':
    main()
