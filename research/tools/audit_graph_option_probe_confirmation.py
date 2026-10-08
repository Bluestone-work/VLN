#!/usr/bin/env python3
"""Confirmation variant: independent streaming integrity audit of every saved alternative branch."""
import argparse
import collections
import hashlib
import json
from pathlib import Path

from probe_graph_options import read_unique, key
from replay_option_calibration import position_error, rotation_error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--probe-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((args.probe_dir / 'manifest.json').read_text())
    capture = Path(manifest['capture_dir'])
    traces = read_unique(next((capture / 'traces').glob('worker_seed*.jsonl')))
    graphs = read_unique(capture / 'graph_options/graph_options.jsonl')
    options = {k: {o['index']: o for o in row['options'] if o['admissible']}
               for k, row in graphs.items() if k[1] in manifest['selected_episodes']}
    reverse = {tuple(k) for k in manifest['reverse_decisions']}
    expected = {('forward', k, index) for k, choices in options.items() for index in choices}
    expected.update(('reverse', k, index) for k in reverse for index in options[k])
    actual = set()
    gates = manifest['config'].get('gates', {'endpoint_tolerance_m': 1e-5, 'rotation_tolerance_rad': 1e-5, 'progress_tolerance_m': 1e-5})
    max_position = max_rotation = max_goal = 0.
    primitive_events = collision_events = back_paths = stop_branches = 0
    raw = args.probe_dir / 'branch_traces.jsonl'
    with raw.open() as stream:
        for line_number, line in enumerate(stream, 1):
            row = json.loads(line)
            k = tuple(row['decision_key'])
            identity = (row['order'], k, row['index'])
            if identity not in expected or identity in actual:
                raise ValueError('Missing, duplicate or inadmissible branch at line {}'.format(line_number))
            actual.add(identity)
            trace, original = row['trace'], traces[k]
            option = options[k][row['index']]
            if (trace['episode_id'] != k[1] or trace['scene_id'] != k[0]
                    or trace['action'] != option['action'] or trace['pre_rng'] != original['pre_rng']):
                raise ValueError('Branch identity, action or starting RNG mismatch')
            pe = position_error(trace['pre_pose']['position'], original['pre_pose']['position'])
            re = rotation_error(trace['pre_pose']['rotation'], original['pre_pose']['rotation'])
            de = abs(trace['distance_before'] - original['distance_before'])
            if pe > gates['endpoint_tolerance_m'] or re > gates['rotation_tolerance_rad'] or de > gates['progress_tolerance_m']:
                raise ValueError('Alternative starts from different physical state or goal distance')
            max_position, max_rotation, max_goal = max(max_position, pe), max(max_rotation, re), max(max_goal, de)
            is_stop = trace['action']['act'] == 0
            if trace['done'] != is_stop or trace['action']['tryout']:
                raise ValueError('Branch terminal/tryout behavior outside calibrated protocol')
            if is_stop and (not trace['primitives'] or trace['primitives'][-1]['action'] != 0 or trace['primitives'][-1]['collided'] is not None):
                raise ValueError('Invalid STOP marker')
            primitive_events += len(trace['primitives'])
            collision_events += sum(p['collided'] is True for p in trace['primitives'])
            back_paths += bool(trace['action'].get('back_path'))
            stop_branches += is_stop
    if actual != expected:
        raise ValueError('Not all required forward/reverse branches are present')
    result = {'experiment_id': manifest['experiment_id'], 'integrity_gate_passed': True,
              'branches': len(actual), 'forward_branches': sum(r[0] == 'forward' for r in actual),
              'reverse_branches': sum(r[0] == 'reverse' for r in actual),
              'max_start_position_error_m': max_position, 'max_start_rotation_error_rad': max_rotation,
              'max_start_goal_distance_error_m': max_goal, 'all_action_dictionaries_exact': True,
              'all_start_rng_exact': True, 'all_admissibility_and_coverage_checks_passed': True,
              'stop_branches': stop_branches, 'branches_with_back_path': back_paths,
              'primitive_events_including_repeats': primitive_events,
              'collision_events_including_repeats': collision_events,
              'raw_trace_sha256': hashlib.sha256(raw.read_bytes()).hexdigest(),
              'audit_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (args.output_dir / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
