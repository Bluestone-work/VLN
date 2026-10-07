#!/usr/bin/env python3
"""Post hoc route-level accounting; does not fit or select a navigation policy."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    root = Path(cfg['replication']['root'])
    cycle = root / 'replication_cycle_001'
    if json.loads((cycle / 'status.json').read_text())['status'] != 'complete':
        raise ValueError('All registered arms must finish first')
    sample = json.loads(Path(cfg['sampling_manifest']).read_text())
    scenes = {str(e['episode_id']): e['scene_id'] for e in sample['episodes']}
    metrics = ['success', 'spl', 'ndtw', 'sdtw', 'distance_to_goal',
               'primitive_action_count', 'path_length', 'high_level_steps', 'collision_events']
    output_rows = []
    summaries = {}
    sources = [args.config, Path(__file__), cycle / 'status.json']
    native_trace_path = root / 'capture_001/traces/worker_seed100.jsonl'
    with native_trace_path.open() as stream:
        native_options = {(r['episode_id'], r['high_level_step']): len(r['primitives'])
                          for r in (json.loads(line) for line in stream)}
    sources.append(native_trace_path)
    for arm in cfg['replication']['arms']:
        schedule_path = root / 'schedules_001' / (arm + '.json')
        schedule = json.loads(schedule_path.read_text())
        path = cycle / (arm + '_enabled_audit') / 'paired_episodes.json'
        rows = json.loads(path.read_text())
        sources += [schedule_path, path]
        if {r['episode_id'] for r in rows} != set(scenes):
            raise ValueError('Population mismatch')
        events = {e['episode_id']: e for e in schedule['events']}
        audit_path = cycle / (arm + '_enabled_audit') / 'summary.json'
        audit = json.loads(audit_path.read_text())
        if not audit['all_required_gates_passed']:
            raise ValueError('Cannot attribute costs without exact prefixes')
        checks = {str(e['decision_key'][1]): e for e in audit['event_checks']}
        sources.append(audit_path)
        option_deltas = {}
        for r in rows:
            ep = r['episode_id']
            event = events.get(ep)
            out = {'arm': arm, 'episode_id': ep, 'scene_id': scenes[ep],
                   'changed': event is not None,
                   'intervention_step': event['high_level_step'] if event else '',
                   'baseline_action_index': event['baseline_index'] if event else '',
                   'alternative_action_index': event['alternative_index'] if event else ''}
            local_delta = (checks[ep]['replayed_primitives'] - native_options[(ep, event['high_level_step'])]) if event else 0
            option_deltas[ep] = local_delta
            out['changed_option_primitive_delta'] = local_delta
            out['remaining_continuation_primitive_delta'] = r['delta']['primitive_action_count'] - local_delta
            for metric in metrics:
                for kind in ['baseline', 'intervention', 'delta']:
                    out[kind + '_' + metric] = r[kind][metric]
            output_rows.append(out)
        categories = {
            'rescued': [r for r in rows if r['delta']['success'] > 0],
            'lost': [r for r in rows if r['delta']['success'] < 0],
            'success_unchanged': [r for r in rows if r['delta']['success'] == 0],
        }
        summaries[arm] = {
            'primitive_cost_increased_routes': [r['episode_id'] for r in rows if r['delta']['primitive_action_count'] > 0],
            'total_primitive_delta': sum(r['delta']['primitive_action_count'] for r in rows),
            'changed_options_total_primitive_delta': sum(option_deltas.values()),
            'remaining_continuations_total_primitive_delta': sum(r['delta']['primitive_action_count'] - option_deltas[r['episode_id']] for r in rows),
            'cheaper_option_but_more_expensive_episode': [r['episode_id'] for r in rows if option_deltas[r['episode_id']] < 0 and r['delta']['primitive_action_count'] > 0],
            'primitive_delta_by_success_outcome': {name: {
                'routes': len(selected),
                'total_primitive_delta': sum(r['delta']['primitive_action_count'] for r in selected)
            } for name, selected in categories.items()},
            'top_absolute_primitive_changes': [
                {'episode_id': r['episode_id'], 'scene_id': scenes[r['episode_id']],
                 'delta': {m: r['delta'][m] for m in metrics}}
                for r in sorted(rows, key=lambda r: (-abs(r['delta']['primitive_action_count']), r['episode_id']))[:8]],
        }
    args.output.mkdir(exist_ok=False)
    with (args.output / 'routes.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    result = {'post_hoc_descriptive_only': True, 'no_policy_fitting': True,
              'route_arm_rows': len(output_rows), 'arms': summaries,
              'hashes': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}
    with (args.output / 'summary.json').open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'route_arm_rows': len(output_rows), 'output': str(args.output)}, indent=2))


if __name__ == '__main__':
    main()
