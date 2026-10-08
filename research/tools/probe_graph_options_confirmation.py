#!/usr/bin/env python3
"""Privileged analysis-only full graph-option branches in a separate simulator."""
import argparse
import collections
import hashlib
import itertools
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(str(ROOT))

import habitat
import habitat_extensions
import numpy as np
import yaml
import vlnce_baselines
from replay_option_calibration import compare, position_error
from vlnce_baselines.adaptive_action.option_calibration import OptionTraceEnv, unpack, restore_rng


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def key(row):
    return (row['scene_id'], row['episode_id'], row['high_level_step'])


def read_unique(path):
    result = collections.OrderedDict()
    with Path(path).open() as stream:
        for line in stream:
            row = json.loads(line)
            if key(row) in result:
                raise ValueError('Duplicate decision: {}'.format(key(row)))
            result[key(row)] = row
    return result


def validate_join(graph, trace, gates):
    if key(graph) != key(trace):
        raise ValueError('Graph and trace decision identity mismatch')
    valid = [i for i, (m, v) in enumerate(zip(graph['mask'], graph['visited'])) if m and not v]
    if valid != [o['index'] for o in graph['options']]:
        raise ValueError('Incomplete or reordered valid graph action set')
    if len(set(o['graph_id'] for o in graph['options'])) != len(valid):
        raise ValueError('Duplicate graph actions')
    selected = next(o for o in graph['options'] if o['index'] == graph['effective_index'])
    if not selected['admissible'] or selected['action'] != trace['action']:
        raise ValueError('Selected sentinel action differs from native trace')
    if position_error(unpack(graph['graph_position']), trace['pre_pose']['position']) > gates['endpoint_tolerance_m']:
        raise ValueError('Graph and controller poses differ')
    for option in graph['options']:
        expected = option['index'] == 0 or not (graph['budget_stop'] or graph['no_vp_left'])
        if option['admissible'] != expected:
            raise ValueError('Option violates episode-limit admissibility')


def branch(env, episode, original, action):
    env.habitat_env.episodes = [episode]
    env.habitat_env.episode_iterator = itertools.cycle([episode])
    env.reset()
    pose = original['pre_pose']
    q = pose['rotation']
    env.habitat_env.sim.set_agent_state(np.asarray(pose['position'], dtype=np.float32),
                                       np.quaternion(q[3], q[0], q[1], q[2]))
    env.habitat_env.task.measurements.reset_measures(episode=episode, task=env.habitat_env.task)
    restore_rng(original['pre_rng'])
    env.step(unpack(action), vis_info=None)
    return env.last_option_trace


def outcome(trace, option):
    poses = [np.asarray(trace['pre_pose']['position'])] + [np.asarray(p['pose']['position']) for p in trace['primitives']]
    target_key = 'stop_pos' if trace['action']['act'] == 0 else 'ghost_pos'
    target = unpack(trace['action'])[target_key]
    error = np.asarray(trace['post_pose']['position']) - target
    return {'index': option['index'], 'graph_id': option['graph_id'], 'current_proposal': option['current_proposal'],
            'action_type': trace['action']['act'], 'logit': option['logit'],
            'progress_m': trace['distance_before'] - trace['distance_after'],
            'distance_after': trace['distance_after'], 'primitive_events': len(trace['primitives']),
            'collision_events': sum(p['collided'] is True for p in trace['primitives']),
            'motion_path_m': sum(float(np.linalg.norm(b - a)) for a, b in zip(poses, poses[1:])),
            'target_error_horizontal_m': float(np.linalg.norm(error[[0, 2]])),
            'back_path_nodes': len(trace['action'].get('back_path') or []),
            'post_pose': trace['post_pose'], 'done': trace['done']}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--capture-dir', type=Path, required=True)
    p.add_argument('--noninterference-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--max-episodes', type=int, default=-1)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    capture = json.loads((args.capture_dir / 'manifest.json').read_text())
    base_protocol = dict(capture['config'])
    protocol = dict(base_protocol)
    protocol.setdefault('gates', {'endpoint_tolerance_m': 1e-5, 'rotation_tolerance_rad': 1e-5,
                                  'progress_tolerance_m': 1e-5, 'primitive_action_sequence_exact': True,
                                  'per_primitive_collision_sequence_exact': True, 'rng_post_exact': True,
                                  'terminal_flag_exact': True, 'baseline_episode_metrics_tolerance': 1e-6})
    protocol.setdefault('reverse_selection_seed', 20261009)
    protocol.setdefault('reverse_states_per_episode', 2)
    protocol.setdefault('limits', 'Confirmation capture; no outcome-based intervention labels.')
    control = json.loads((args.noninterference_dir / 'summary.json').read_text())
    if not control['baseline_noninterference_gate_passed'] or control['config'] != base_protocol:
        raise ValueError('Paired baseline noninterference gate has not passed')
    trace_paths = list((args.capture_dir / 'traces').glob('worker_seed*.jsonl'))
    if len(trace_paths) != 1:
        raise ValueError('Pilot requires a single capture worker')
    trace_path = trace_paths[0]
    graph_path = args.capture_dir / 'graph_options/graph_options.jsonl'
    for path in ['vlnce_baselines/common/environments.py', 'vlnce_baselines/adaptive_action/option_calibration.py',
                 'vlnce_baselines/adaptive_action/graph_option_capture.py', 'vlnce_baselines/ss_trainer_ETP.py']:
        if digest(path) != capture['hashes'][path]:
            raise ValueError('Captured code changed: ' + path)
    traces, graphs = read_unique(trace_path), read_unique(graph_path)
    if set(traces) != set(graphs):
        raise ValueError('Capture missing complete graph or controller records')
    grouped = collections.OrderedDict()
    for k, trace in traces.items():
        validate_join(graphs[k], trace, protocol['gates'])
        grouped.setdefault(k[1], []).append(k)
    if len(grouped) != protocol['episodes']:
        raise ValueError('Capture episode count differs from protocol')
    for episode, keys in grouped.items():
        if [k[2] for k in keys] != list(range(len(keys))) or not traces[keys[-1]]['done']:
            raise ValueError('Incomplete captured episode ' + episode)
    if args.max_episodes > 0:
        grouped = collections.OrderedDict(list(grouped.items())[:args.max_episodes])
    reverse_keys = set()
    for keys in grouped.values():
        def seeded_order(k):
            return hashlib.sha256('{}|{}|{}|{}'.format(protocol['reverse_selection_seed'], *k).encode()).hexdigest()
        reverse_keys.update(sorted(keys, key=seeded_order)[:protocol['reverse_states_per_episode']])
    worker_config = habitat.Config(yaml.safe_load(trace_path.with_suffix('.yaml').read_text()))
    worker_config.freeze()
    if worker_config.VIDEO_OPTION or not worker_config.TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING:
        raise ValueError('Unsupported simulator/controller configuration')
    if digest(control['sources']['capture_episodes']['path']) != control['sources']['capture_episodes']['sha256']:
        raise ValueError('Native capture results changed after noninterference check')
    manifest = {'created_utc': datetime.now(timezone.utc).isoformat(), 'experiment_id': protocol['experiment_id'],
                'capture_dir': str(args.capture_dir), 'config': protocol, 'max_episodes': args.max_episodes,
                'selected_episodes': list(grouped), 'reverse_decisions': sorted(reverse_keys),
                'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip(),
                'hardware': subprocess.check_output(['nvidia-smi', '--query-gpu=name,uuid,driver_version', '--format=csv,noheader']).decode(),
                'hashes': {str(path): digest(path) for path in [trace_path, graph_path, Path(__file__),
                    args.capture_dir / 'manifest.json', args.noninterference_dir / 'summary.json',
                    Path('research/tools/replay_option_calibration.py')]},
                'privileged_analysis_only': True, 'command_argv': sys.argv}
    (args.output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    dataset = habitat.make_dataset(worker_config.TASK_CONFIG.DATASET.TYPE, config=worker_config.TASK_CONFIG.DATASET)
    episodes = {str(e.episode_id): e for e in dataset.episodes}
    dataset.episodes = [episodes[k] for k in grouped]
    env = OptionTraceEnv(worker_config, dataset=dataset, trace_dir=False)
    sentinel_checks, order_checks = [], []
    decisions = []
    raw_path = args.output_dir / 'branch_traces.jsonl'
    try:
        with raw_path.open('x') as raw, (args.output_dir / 'decisions.jsonl').open('x') as output:
            for episode_id, keys in grouped.items():
                for k in keys:
                    graph, original = graphs[k], traces[k]
                    options = [o for o in graph['options'] if o['admissible']]
                    forward = {}
                    for option in options:
                        actual = branch(env, episodes[episode_id], original, option['action'])
                        forward[option['index']] = actual
                        raw.write(json.dumps({'decision_key': k, 'order': 'forward', 'index': option['index'], 'trace': actual}, allow_nan=False) + '\n')
                    sentinel = compare(original, forward[graph['effective_index']], protocol['gates'])
                    sentinel['decision_key'] = k
                    sentinel_checks.append(sentinel)
                    if not sentinel['passed']:
                        raise ValueError('Selected-action sentinel failed: {}'.format(k))
                    reversed_here = k in reverse_keys
                    if reversed_here:
                        for option in reversed(options):
                            actual = branch(env, episodes[episode_id], original, option['action'])
                            raw.write(json.dumps({'decision_key': k, 'order': 'reverse', 'index': option['index'], 'trace': actual}, allow_nan=False) + '\n')
                            check = compare(forward[option['index']], actual, protocol['gates'])
                            check.update({'decision_key': k, 'index': option['index']})
                            order_checks.append(check)
                            if not check['passed']:
                                raise ValueError('Enumeration order changed action outcome: {}'.format(k))
                    row = {'scene_id': k[0], 'episode_id': k[1], 'high_level_step': k[2],
                           'trajectory_id': graph['trajectory_id'], 'effective_index': graph['effective_index'],
                           'policy_index': graph['policy_index'], 'budget_stop': graph['budget_stop'],
                           'forced_stop': graph['forced_stop'], 'distance_before': original['distance_before'],
                           'structural_options': len(graph['options']), 'admissible_options': len(options),
                           'options': [outcome(forward[o['index']], o) for o in options],
                           'selected_sentinel_passed': sentinel['passed'], 'order_repeated': reversed_here,
                           'privileged_analysis_only': True}
                    decisions.append(row)
                    output.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')
                    output.flush()
                print('episode={} decisions={} forward_branches={} reversed_branches={}'.format(episode_id, len(decisions),
                      sum(r['admissible_options'] for r in decisions), len(order_checks)), flush=True)
    except Exception as error:
        (args.output_dir / 'FAILED.json').write_text(json.dumps({'error': repr(error), 'completed_decisions': len(decisions),
                                                               'sentinel_checks': sentinel_checks, 'order_checks': order_checks}, indent=2) + '\n')
        raise
    finally:
        env.close()
    for name, rows in [('sentinel_checks', sentinel_checks), ('order_checks', order_checks)]:
        with (args.output_dir / (name + '.jsonl')).open('x') as stream:
            for row in rows:
                stream.write(json.dumps(row, sort_keys=True) + '\n')
    summary = {'experiment_id': protocol['experiment_id'], 'episodes': len(grouped),
               'scenes': len({k[0] for keys in grouped.values() for k in keys}),
               'decisions': len(decisions), 'forward_branches': sum(r['admissible_options'] for r in decisions),
               'reversed_decisions': len(reverse_keys), 'reversed_branches': len(order_checks),
               'selected_sentinels_passed': sum(c['passed'] for c in sentinel_checks),
               'order_checks_passed': sum(c['passed'] for c in order_checks),
               'max_sentinel_endpoint_error_m': max(c['endpoint_error_m'] for c in sentinel_checks),
               'max_order_endpoint_error_m': max(c['endpoint_error_m'] for c in order_checks),
               'sentinel_sensor_matches': sum(c['sensor_hashes_exact'] for c in sentinel_checks),
               'order_sensor_matches': sum(c['sensor_hashes_exact'] for c in order_checks),
               'all_required_gates_passed': all(c['passed'] for c in sentinel_checks + order_checks),
               'baseline_noninterference_passed': True,
               'all_admissible_options_complete': True,
               'masked_budget_options_skipped': sum(r['structural_options'] - r['admissible_options'] for r in decisions),
               'limits': protocol['limits'], 'privileged_analysis_only': True}
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
