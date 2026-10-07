#!/usr/bin/env python3
"""Census terminal/recovery symptoms from accepted baseline traces only."""
import argparse
import json
from pathlib import Path
from audit_single_intervention import read_rows, group_rows, one
from run_option_capture import digest


def census(root, capture_tag='001', metric_tag='001'):
    root = Path(root)
    metrics = json.loads((root / ('noninterference_' + metric_tag) / 'capture_episodes.json').read_text())
    trace_path = one(root / ('capture_' + capture_tag) / 'traces', '*.jsonl')
    traces = group_rows(read_rows(trace_path))
    rows = []
    for episode, trace in sorted(traces.items()):
        metric = metrics[episode]
        distances = [float(r['distance_after']) for r in trace]
        inside = [i for i, value in enumerate(distances) if value < 3.0]
        final = trace[-1]
        forced = bool(metric['forced_stop_count'])
        failed = not bool(metric['success'])
        rows.append({'episode_id': episode, 'success': bool(metric['success']),
                     'endpoint_neighborhood_visited': bool(inside),
                     'first_endpoint_inside_step': inside[0] if inside else None,
                     'final_goal_distance_m': float(metric['distance_to_goal']),
                     'minimum_endpoint_distance_m': min(distances),
                     'arrived_then_left': bool(inside) and float(metric['distance_to_goal']) >= 3.0,
                     'never_arrived_failed': failed and not bool(inside),
                     'forced_stop': forced, 'forced_stop_failed': forced and failed,
                     'policy_stop': bool(metric['policy_stop_count']),
                     'final_action': final['action'].get('act'),
                     'final_back_path_nodes': len(final['action'].get('back_path') or []),
                     'final_primitives': len(final['primitives'])})
    failed_rows = [r for r in rows if not r['success']]
    return {'episodes': len(rows), 'successes': sum(r['success'] for r in rows),
            'failures': len(failed_rows),
            'arrived_then_left': sum(r['arrived_then_left'] for r in rows),
            'never_arrived_failed': sum(r['never_arrived_failed'] for r in rows),
            'forced_stop': sum(r['forced_stop'] for r in rows),
            'forced_stop_failed': sum(r['forced_stop_failed'] for r in rows),
            'policy_stop': sum(r['policy_stop'] for r in rows),
            'rows': rows, 'hashes': {str(trace_path): digest(trace_path)}}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, action='append', required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--capture-tag', default='001')
    p.add_argument('--metric-tag', default='001')
    args = p.parse_args()
    result = {'analysis': 'baseline terminal/recovery symptom census',
              'threshold_m': 3.0, 'capture_tag': args.capture_tag,
              'metric_tag': args.metric_tag,
              'cohorts': {str(r): census(r, args.capture_tag, args.metric_tag) for r in args.root},
              'limits': 'Descriptive endpoint-based labels from accepted baseline captures. Endpoint samples can miss between-decision proximity and do not identify STOP causality. No interventions or fitting.',
              'no_model_trained': True}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: {x: v for x, v in value.items() if x != 'rows'}
                      for k, value in result['cohorts'].items()}, indent=2))


if __name__ == '__main__':
    main()
