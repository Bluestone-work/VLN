#!/usr/bin/env python3
"""Post-hoc H=1/2/3 and displaced-ghost observations; no policy or label tuning."""
import argparse
import json
from pathlib import Path
import numpy as np

from audit_single_intervention import read_rows, group_rows, one
from route_alignment import trace_path
from run_option_capture import digest


def movement(record):
    return float(np.linalg.norm(np.diff(trace_path(record), axis=0), axis=1).sum())


def window(records, start, horizon):
    # STOP is absorbing for this OFFLINE comparison only; no extra action is run.
    part = records[start:min(start + horizon, len(records))]
    return {'distance_to_goal': part[-1]['distance_after'],
            'primitive_count': sum(len(r['primitives']) for r in part),
            'movement_m': sum(movement(r) for r in part),
            'actual_decisions': len(part), 'terminal': bool(part[-1]['done'])}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--audit-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    audit_path = args.audit_dir / 'summary.json'
    audit = json.loads(audit_path.read_text())
    if not audit['all_required_gates_passed'] or audit['mode'] != 'enabled':
        raise ValueError('Accepted continuation required')
    manifest = json.loads((args.run_dir / 'manifest.json').read_text())
    old_path = one(Path(manifest['source_capture']) / 'traces', '*.jsonl')
    new_path = one(args.run_dir / 'traces', '*.jsonl')
    old, new = group_rows(read_rows(old_path)), group_rows(read_rows(new_path))
    hooks = read_rows(args.run_dir / 'hook/decisions.jsonl')
    pairs = {r['episode_id']: r for r in json.loads((args.audit_dir / 'paired_episodes.json').read_text())}
    output = []
    for event in manifest['schedule']['events']:
        episode, step = event['episode_id'], event['high_level_step']
        horizons = {}
        for horizon in [1, 2, 3]:
            a, b = window(old[episode], step, horizon), window(new[episode], step, horizon)
            horizons[str(horizon)] = {'baseline': a, 'intervention': b,
                'extra_goal_progress_m': a['distance_to_goal'] - b['distance_to_goal'],
                'primitive_delta': b['primitive_count'] - a['primitive_count'],
                'movement_delta_m': b['movement_m'] - a['movement_m']}
        first = horizons['1']
        if first['primitive_delta'] > 0 or first['movement_delta_m'] > 1e-5:
            raise ValueError('Frozen immediate cost cap no longer holds')
        displaced = event['baseline_action']['ghost_vp']
        reselected = [r['high_level_step'] for r in new[episode] if r['high_level_step'] > step and r['action'].get('ghost_vp') == displaced]
        final = new[episode][-1]
        hk = (final['scene_id'], episode, final['high_level_step'])
        output.append({'episode_id': episode, 'step': step, 'horizons': horizons,
                       'displaced_graph_id': displaced, 'displaced_ghost_reselected_at_steps': reselected,
                       'final_stop_native_index': hooks[hk]['native_index'],
                       'final_stop_budget': hooks[hk]['budget_stop'],
                       'final_stop_back_path_nodes': len(final['action'].get('back_path') or []),
                       'final_goal_distance_m': final['distance_after'],
                       'final_episode_delta': pairs[episode]['delta']})
    counts = {}
    for horizon in ['1', '2', '3']:
        rows = [r['horizons'][horizon] for r in output]
        counts[horizon] = {'extra_progress_negative': sum(r['extra_goal_progress_m'] < -1e-5 for r in rows),
                            'primitive_delta_positive': sum(r['primitive_delta'] > 0 for r in rows),
                            'movement_delta_positive': sum(r['movement_delta_m'] > 1e-5 for r in rows)}
    result = {'experiment_id': audit['experiment_id'], 'analysis_type': 'post-hoc, exploratory; horizons examined after continuation outcomes',
              'events': output, 'horizon_counts': counts, 'all_immediate_cost_caps_passed': True,
              'limits': 'Fixed high-level horizons are not equal primitive-time or execution cost. STOP is absorbing only in this offline comparison. No new model, tuned label rule, semantic proof or prospective validation.',
              'hashes': {str(path): digest(path) for path in [Path(__file__), audit_path, old_path, new_path,
                  args.run_dir / 'hook/decisions.jsonl', args.audit_dir / 'paired_episodes.json']}}
    (args.output_dir / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'horizon_counts': counts, 'episodes': [{k: r[k] for k in
        ['episode_id', 'displaced_ghost_reselected_at_steps', 'final_stop_native_index', 'final_stop_budget', 'final_stop_back_path_nodes']} for r in output]}, indent=2))


if __name__ == '__main__':
    main()
