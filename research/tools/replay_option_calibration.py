#!/usr/bin/env python3
"""Replay captured full options in a separate process using the real controller.

Sequential replay follows each recorded episode from reset. Isolated branches
reset task state, place the agent at the recorded pose, restore controller RNG,
and execute just the captured option. No live rollout worker is ever rewound.
"""
import argparse
import collections
import hashlib
import itertools
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(str(ROOT))

import numpy as np
import yaml
import habitat
import habitat_extensions
import vlnce_baselines
from vlnce_baselines.adaptive_action.option_calibration import OptionTraceEnv, unpack, restore_rng, state_snapshot


def rotation_error(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    cosine = abs(float(np.dot(a, b))) / float(np.linalg.norm(a) * np.linalg.norm(b))
    return 2 * math.acos(min(1.0, max(-1.0, cosine)))


def position_error(a, b):
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))


def compare(expected, actual, gate):
    expected_sequence = [p['action'] for p in expected['primitives']]
    actual_sequence = [p['action'] for p in actual['primitives']]
    expected_collisions = [p['collided'] for p in expected['primitives']]
    actual_collisions = [p['collided'] for p in actual['primitives']]
    ep, ap = expected['post_pose'], actual['post_pose']
    result = {'endpoint_error_m': position_error(ep['position'], ap['position']),
              'rotation_error_rad': rotation_error(ep['rotation'], ap['rotation']),
              'pre_pose_error_m': position_error(expected['pre_pose']['position'], actual['pre_pose']['position']),
              'progress_error_m': abs((expected['distance_before'] - expected['distance_after']) -
                                      (actual['distance_before'] - actual['distance_after'])),
              'primitive_sequence_exact': expected_sequence == actual_sequence,
              'collision_sequence_exact': expected_collisions == actual_collisions,
              'rng_post_exact': expected['post_rng'] == actual['post_rng'],
              'terminal_flag_exact': expected['done'] == actual['done'],
              'sensor_hashes_exact': expected['observation_hashes'] == actual['observation_hashes'],
              'expected_primitives': len(expected_sequence), 'replayed_primitives': len(actual_sequence)}
    if len(expected_sequence) == len(actual_sequence):
        result['max_primitive_position_error_m'] = max([position_error(e['pose']['position'], a['pose']['position'])
                                                      for e, a in zip(expected['primitives'], actual['primitives'])] + [0.0])
    else:
        result['max_primitive_position_error_m'] = None
    result['passed'] = all([
        result['endpoint_error_m'] <= gate['endpoint_tolerance_m'],
        result['pre_pose_error_m'] <= gate['endpoint_tolerance_m'],
        result['rotation_error_rad'] <= gate['rotation_tolerance_rad'],
        result['progress_error_m'] <= gate['progress_tolerance_m'],
        result['primitive_sequence_exact'], result['collision_sequence_exact'],
        result['rng_post_exact'], result['terminal_flag_exact']])
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--capture-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--max-episodes', type=int, default=-1)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((args.capture_dir / 'manifest.json').read_text())
    protocol = manifest['config']
    traces = list((args.capture_dir / 'traces').glob('worker_seed*.jsonl'))
    if len(traces) != 1:
        raise ValueError('First calibration requires one capture worker')
    source = traces[0]
    config = habitat.Config(yaml.safe_load(source.with_suffix('.yaml').read_text()))
    config.freeze()
    if config.VIDEO_OPTION:
        raise ValueError('Video mode is outside this calibration')
    current_sources = ['vlnce_baselines/common/environments.py', 'vlnce_baselines/adaptive_action/option_calibration.py']
    for path in current_sources:
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != manifest['hashes'][path]:
            raise ValueError('Controller or tracing source changed since capture: ' + path)
    grouped = collections.OrderedDict()
    with source.open() as stream:
        for line in stream:
            row = json.loads(line)
            grouped.setdefault(row['episode_id'], []).append(row)
    if args.max_episodes > 0:
        grouped = collections.OrderedDict(list(grouped.items())[:args.max_episodes])
    for episode, rows in grouped.items():
        if [r['high_level_step'] for r in rows] != list(range(len(rows))) or not rows[-1]['done']:
            raise ValueError('Incomplete or duplicate episode {}'.format(episode))
    dataset = habitat.make_dataset(config.TASK_CONFIG.DATASET.TYPE, config=config.TASK_CONFIG.DATASET)
    episodes = {str(e.episode_id): e for e in dataset.episodes}
    dataset.episodes = [episodes[key] for key in grouped]
    env = OptionTraceEnv(config, dataset=dataset, trace_dir=False)
    rows_output = []
    try:
        for mode in protocol['replay_modes']:
            for episode_id, expected_rows in grouped.items():
                episode = episodes[episode_id]
                env.habitat_env.episodes = [episode]
                env.habitat_env.episode_iterator = itertools.cycle([episode])
                if mode == 'sequential':
                    env.reset()
                for expected in expected_rows:
                    if mode == 'isolated_branch':
                        env.reset()
                        pose = expected['pre_pose']
                        q = pose['rotation']
                        env.habitat_env.sim.set_agent_state(np.asarray(pose['position'], dtype=np.float32),
                                                           np.quaternion(q[3], q[0], q[1], q[2]))
                        # Branch metrics start at this decision, not at the route origin.
                        env.habitat_env.task.measurements.reset_measures(episode=episode, task=env.habitat_env.task)
                    restore_rng(expected['pre_rng'])
                    env.step(unpack(expected['action']), vis_info=None)
                    result = compare(expected, env.last_option_trace, protocol['gates'])
                    action = unpack(expected['action'])
                    result.update({'mode': mode, 'episode_id': episode_id,
                                   'high_level_step': expected['high_level_step'],
                                   'action_type': int(action['act']),
                                   'back_path_nodes': len(action.get('back_path') or []),
                                   'tryout': bool(action['tryout']),
                                   'collided_primitives': sum(p['collided'] is True for p in expected['primitives'])})
                    rows_output.append(result)
                    with (args.output_dir / 'comparisons.jsonl').open('a') as stream:
                        stream.write(json.dumps(result, sort_keys=True) + '\n')
                print('{} episode={} options={} cumulative_failures={}'.format(mode, episode_id,
                      len(expected_rows), sum(not r['passed'] for r in rows_output)), flush=True)
    finally:
        env.close()
    summaries = {}
    for mode in protocol['replay_modes']:
        items = [r for r in rows_output if r['mode'] == mode]
        summaries[mode] = {'options': len(items), 'passed': sum(r['passed'] for r in items),
                           'max_endpoint_error_m': max(r['endpoint_error_m'] for r in items),
                           'max_rotation_error_rad': max(r['rotation_error_rad'] for r in items),
                           'max_progress_error_m': max(r['progress_error_m'] for r in items),
                           'sensor_hash_matches': sum(r['sensor_hashes_exact'] for r in items),
                           'primitive_events': sum(r['expected_primitives'] for r in items),
                           'collision_events': sum(r['collided_primitives'] for r in items),
                           'stop_options': sum(r['action_type'] == 0 for r in items),
                           'options_with_back_path': sum(r['back_path_nodes'] > 0 for r in items),
                           'tryout_options': sum(r['tryout'] for r in items)}
    summary = {'experiment_id': protocol['experiment_id'], 'capture_dir': str(args.capture_dir),
               'episodes': len(grouped), 'scenes': len({r[0]['scene_id'] for r in grouped.values()}),
               'modes': summaries, 'selected_option_replay_gate_passed': all(r['passed'] for r in rows_output),
               'baseline_noninterference_gate': 'must be checked separately against untraced evaluation',
               'all_action_counterfactual_validated': False,
               'gates': protocol['gates'], 'scope_limits': protocol['limits'],
               'capture_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
               'replay_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
