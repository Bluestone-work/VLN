#!/usr/bin/env python3
"""Audit opt-in RGB-D semantic capture alignment without using future labels."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(value):
    return hashlib.sha256(np.asarray(value).tobytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--capture', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    graph_path = args.capture / 'graph_options' / 'graph_options.jsonl'
    trace_path = args.capture / 'traces' / 'worker_seed100.jsonl'
    graphs = [json.loads(x) for x in graph_path.open() if x.strip()]
    traces = [json.loads(x) for x in trace_path.open() if x.strip()]
    gkeys = [(str(r['episode_id']), int(r['high_level_step'])) for r in graphs]
    tkeys = [(str(r['episode_id']), int(r['high_level_step'])) for r in traces]
    result = {
        'capture': str(args.capture),
        'graph_rows': len(graphs),
        'trace_rows': len(traces),
        'unique_graph_keys': len(set(gkeys)),
        'unique_trace_keys': len(set(tkeys)),
        'key_sets_equal': set(gkeys) == set(tkeys),
        'semantic_visual_rows': 0,
        'rgb_files': 0,
        'depth_files': 0,
        'file_hash_matches': 0,
        'file_hash_mismatches': [],
        'state_metadata_mismatches': [],
        'trace_post_hash_comparisons': 0,
        'trace_post_hash_matches': 0,
        'trace_post_hash_nonmatches_expected_timing': 0,
        'selected_action_records': 0,
        'privileged_input_violations': [],
        'status': 'PASS_WITH_EXPLICIT_PRE_ACTION_TIMING',
    }
    for row in graphs:
        key = (str(row['episode_id']), int(row['high_level_step']))
        sv = row.get('semantic_visual')
        if not sv:
            result['state_metadata_mismatches'].append({'key': key, 'error': 'missing semantic_visual'})
            continue
        result['semantic_visual_rows'] += 1
        if row.get('privileged_labels_in_features'):
            result['privileged_input_violations'].append(key)
        if row.get('policy_index') is not None:
            result['selected_action_records'] += 1
        if sv.get('episode_id') != row.get('episode_id') or int(sv.get('high_level_step', -1)) != int(row.get('high_level_step', -2)):
            result['state_metadata_mismatches'].append({'key': key, 'error': 'episode/step mismatch'})
        if sv.get('current_vp') != row.get('current_vp') or sv.get('graph_ids') != row.get('graph_ids'):
            result['state_metadata_mismatches'].append({'key': key, 'error': 'graph identity mismatch'})
        for sensor in ('rgb', 'depth'):
            path = Path(sv.get('files', {}).get(sensor, ''))
            if not path.exists():
                result['state_metadata_mismatches'].append({'key': key, 'error': f'missing {sensor} file', 'path': str(path)})
                continue
            result[sensor + '_files'] += 1
            actual = sha(np.load(path, allow_pickle=False))
            expected = sv.get('observation_hashes', {}).get(sensor)
            if actual == expected:
                result['file_hash_matches'] += 1
            else:
                result['file_hash_mismatches'].append({'key': key, 'sensor': sensor, 'expected': expected, 'actual': actual})
    trace_by_key = {k: t for k, t in zip(tkeys, traces)}
    for row in graphs:
        key = (str(row['episode_id']), int(row['high_level_step']))
        sv = row.get('semantic_visual') or {}
        tr = trace_by_key.get(key)
        if not tr:
            continue
        for sensor in ('rgb', 'depth'):
            pre_hash = sv.get('observation_hashes', {}).get(sensor)
            post_hash = (tr.get('observation_hashes') or {}).get(sensor)
            if pre_hash and post_hash:
                result['trace_post_hash_comparisons'] += 1
                if pre_hash == post_hash:
                    result['trace_post_hash_matches'] += 1
                else:
                    result['trace_post_hash_nonmatches_expected_timing'] += 1
    if result['file_hash_mismatches'] or result['state_metadata_mismatches'] or result['privileged_input_violations']:
        result['status'] = 'FAIL'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    raise SystemExit(1 if result['status'] == 'FAIL' else 0)


if __name__ == '__main__':
    main()
