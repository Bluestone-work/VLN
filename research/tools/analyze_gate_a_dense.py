#!/usr/bin/env python3
"""Analyze the frozen Gate A native-vs-dense full-return census."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


EPS = 1e-6


def read_jsonl(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream]


def quality_rescue(base, row):
    m = row['metrics']
    return (base['success'] < 1.0 - EPS and m['success'] >= 1.0 - EPS
            and m['ndtw'] + EPS >= base['ndtw'])


def cost_capped(base, row):
    return quality_rescue(base, row) and m_steps(row) <= m_steps(base) + EPS


def m_steps(row):
    return float(row.get('metrics', row).get('steps_taken', row.get('steps_taken', 0)))


def strict_dominates(base, row):
    m = row['metrics']
    conditions = [
        m['success'] + EPS >= base['success'],
        m['spl'] + EPS >= base['spl'],
        m['ndtw'] + EPS >= base['ndtw'],
        m['distance_to_goal'] <= base['distance_to_goal'] + EPS,
        m['path_length'] <= base['path_length'] + EPS,
        m_steps(row) <= m_steps(base) + EPS,
    ]
    strict = [
        m['success'] > base['success'] + EPS,
        m['spl'] > base['spl'] + EPS,
        m['ndtw'] > base['ndtw'] + EPS,
        m['distance_to_goal'] < base['distance_to_goal'] - EPS,
        m['path_length'] < base['path_length'] - EPS,
        m_steps(row) < m_steps(base) - EPS,
    ]
    return all(conditions) and any(strict)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--critical-plan', type=Path, required=True)
    p.add_argument('--native-results', type=Path, required=True)
    p.add_argument('--dense-results', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    plan = json.loads(args.critical_plan.read_text())
    native = read_jsonl(args.native_results)
    dense = read_jsonl(args.dense_results)
    controls = {str(r['case']['episode_id']): r for r in native if r['case']['mode'] == 'control'}
    native_states = defaultdict(list)
    selected_native = {}
    for r in native:
        c = r['case']
        if c['mode'] != 'action':
            continue
        key = (str(c['episode_id']), int(c['high_level_step']))
        native_states[key].append(r)
        if int(c.get('action_index', -1)) == int(c.get('effective_index', -2)):
            selected_native[key] = r
    dense_states = defaultdict(list)
    for r in dense:
        c = r['case']
        if c['mode'] == 'dense_action':
            dense_states[(str(c['episode_id']), int(c['high_level_step']))].append(r)

    state_rows = []
    route_rows = {}
    for state in plan['states']:
        ep, step = str(state['episode_id']), int(state['high_level_step'])
        if int(state.get('effective_index', 0)) == 0:
            continue
        key = (ep, step)
        base = controls[ep]['metrics']
        a0 = native_states[key]
        a1_new = dense_states[key]
        all_a1 = a0 + a1_new
        def best(rows, predicate):
            return any(predicate(base, r) for r in rows)
        row = {
            'episode_id': ep, 'scene_id': state['scene_id'], 'high_level_step': step,
            'reasons': '|'.join(state.get('reasons', [])),
            'baseline_success': base['success'], 'baseline_spl': base['spl'],
            'baseline_ndtw': base['ndtw'], 'baseline_goal_error': base['distance_to_goal'],
            'a0_candidates': len(a0), 'a1_new_candidates': len(a1_new),
            'a1_candidates': len(all_a1),
            'a0_strict': best(a0, strict_dominates), 'a1_strict': best(all_a1, strict_dominates),
            'a0_quality_rescue': best(a0, quality_rescue), 'a1_quality_rescue': best(all_a1, quality_rescue),
            'a0_cost_capped_rescue': best(a0, cost_capped), 'a1_cost_capped_rescue': best(all_a1, cost_capped),
            'new_candidate_quality_rescue': best(a1_new, quality_rescue),
            'new_candidate_cost_capped_rescue': best(a1_new, cost_capped),
            'native_selected_quality_rescue': quality_rescue(base, selected_native[key]) if key in selected_native else False,
            'native_selected_success': selected_native[key]['metrics']['success'] if key in selected_native else None,
            'new_candidate_steps_min': min((m_steps(r) for r in a1_new), default=None),
            'new_candidate_steps_max': max((m_steps(r) for r in a1_new), default=None),
            'new_candidate_collisions_mean': (sum(r['primitive_collision_events'] for r in a1_new) / len(a1_new) if a1_new else None),
        }
        state_rows.append(row)
        rr = route_rows.setdefault(ep, {'episode_id': ep, 'scene_id': state['scene_id'], 'states': 0})
        rr['states'] += 1
        for name in ['a0_strict','a1_strict','a0_quality_rescue','a1_quality_rescue','a0_cost_capped_rescue','a1_cost_capped_rescue','new_candidate_quality_rescue','new_candidate_cost_capped_rescue','native_selected_quality_rescue']:
            rr[name] = bool(rr.get(name, False) or row[name])
    route_rows = list(route_rows.values())
    with (args.output_dir / 'per_state.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(state_rows[0].keys()))
        writer.writeheader(); writer.writerows(state_rows)
    with (args.output_dir / 'per_route.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(route_rows[0].keys()))
        writer.writeheader(); writer.writerows(route_rows)

    scenes = lambda rows, key: len({r['scene_id'] for r in rows if r[key]})
    summary = {
        'protocol': 'Gate A Dense-Candidate Full-Return Oracle',
        'privileged_analysis_only': True,
        'unresolved_routes_total': len(plan['failed_episodes']),
        'critical_states_with_nonstop': len(state_rows),
        'scenes_with_critical_states': len({r['scene_id'] for r in state_rows}),
        'native_a0_branch_count': sum(r['a0_candidates'] for r in state_rows),
        'dense_a1_new_branch_count': sum(r['a1_new_candidates'] for r in state_rows),
        'a0_rescue_routes_strict': sum(r['a0_strict'] for r in route_rows),
        'a1_rescue_routes_strict': sum(r['a1_strict'] for r in route_rows),
        'a0_rescue_routes_quality': sum(r['a0_quality_rescue'] for r in route_rows),
        'a1_rescue_routes_quality': sum(r['a1_quality_rescue'] for r in route_rows),
        'a0_rescue_routes_cost_capped': sum(r['a0_cost_capped_rescue'] for r in route_rows),
        'a1_rescue_routes_cost_capped': sum(r['a1_cost_capped_rescue'] for r in route_rows),
        'new_candidate_rescue_routes_quality': sum(r['new_candidate_quality_rescue'] for r in route_rows),
        'new_candidate_rescue_routes_cost_capped': sum(r['new_candidate_cost_capped_rescue'] for r in route_rows),
        'native_rescue_but_not_policy_selected_routes': sum(r['a0_quality_rescue'] and not r['native_selected_quality_rescue'] for r in route_rows),
        'a1_rescue_scene_coverage_quality': scenes(state_rows, 'a1_quality_rescue'),
        'a1_rescue_scene_coverage_cost_capped': scenes(state_rows, 'a1_cost_capped_rescue'),
        'routes_with_no_dense_rescue': sum(not r['a1_quality_rescue'] for r in route_rows),
        'decision': 'GO for proposal-coverage mechanism' if sum(r['a1_quality_rescue'] for r in route_rows) > sum(r['a0_quality_rescue'] for r in route_rows) else 'NO-GO for proposal expansion / waypoint predictor modification',
        'interpretation_rule': 'A1 must add route-level quality or cost-capped rescue over A0 across multiple scenes; otherwise NO-GO.',
    }
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
