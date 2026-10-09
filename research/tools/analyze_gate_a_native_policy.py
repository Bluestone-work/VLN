#!/usr/bin/env python3
"""Paired analysis for the closed-loop A0/A1 native-policy experiment."""
import csv, json, math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'research/results/gate_a_native_policy'


def rows(path):
    with path.open() as f: return [json.loads(x) for x in f]


def metrics(path):
    with path.open() as f: return json.load(f)


def action_signature(option):
    a = option['action']
    def back_signature(item):
        pair = item.get('__tuple__', item)
        return (str(pair[0]), tuple(round(float(y), 5) for y in pair[1]['__array__']))
    # Ghost ids are implementation identities. Positions and traversed path
    # determine whether the native controller received the same target.
    if a['act'] == 0:
        return ('stop', tuple(round(float(x), 5) for x in a['stop_pos']['__array__']), tuple(back_signature(x) for x in a.get('back_path', [])))
    return ('move', tuple(round(float(x), 5) for x in a['ghost_pos']['__array__']), tuple(back_signature(x) for x in a.get('back_path', [])))


def target_signature(option):
    """Candidate identity for nested-set checks; ignore graph back-path rewrites."""
    a = option['action']
    if a['act'] == 0:
        return ('stop', tuple(round(float(x), 5) for x in a['stop_pos']['__array__']))
    return ('move', tuple(round(float(x), 5) for x in a['ghost_pos']['__array__']))


def action_at(row):
    return next(x for x in row['options'] if int(x['index']) == int(row['effective_index']))


def summarize(cohort, a0_dir, a1_dir, eval_cohort):
    """Compare paired episode outputs from two complete online rollouts.

    The capture directories intentionally live under ``runs/*`` while Habitat
    writes evaluator JSON under ``data/logs/eval_results/*``.  Keep that
    mapping explicit so the analysis cannot accidentally read a stale split.
    """
    eval_root = ROOT / 'data/logs/eval_results'
    a0_eval = eval_root / ('gate_a_native_policy_%s_a0_v2' % eval_cohort)
    a1_eval = eval_root / ('gate_a_native_policy_%s_a1_v6' % eval_cohort)
    a0m = metrics(a0_eval / ('stats_ckpt_59_%s.json' % cohort))
    a1m = metrics(a1_eval / ('stats_ckpt_59_%s.json' % cohort))
    a0_rows = {(str(r['episode_id']), int(r['high_level_step'])): r for r in rows(a0_dir / 'graph_options/graph_options.jsonl')}
    a1_rows = {(str(r['episode_id']), int(r['high_level_step'])): r for r in rows(a1_dir / 'graph_options/graph_options.jsonl')}
    a0_eps = {str(k): v for k, v in metrics(a0_eval / ('stats_ep_ckpt_59_%s_r0_w1.json' % cohort)).items()}
    a1_eps = {str(k): v for k, v in metrics(a1_eval / ('stats_ep_ckpt_59_%s_r0_w1.json' % cohort)).items()}
    manifest = {str(x['episode_id']): x for x in json.loads((BASE / 'manifest.json').read_text())['route_rows']}
    route_rows = []
    for ep in sorted(a0_eps, key=lambda x: int(x)):
        left, right = a0_eps[ep], a1_eps[ep]
        keys = sorted(k for k in a0_rows if k[0] == ep and k in a1_rows)
        first_diff = None; direct_new = 0; action_diffs = 0; compared = 0; nested_violations = 0
        first_a0_candidates = ''; first_a1_candidates = ''; first_nested = ''
        for key in keys:
            x, y = action_at(a0_rows[key]), action_at(a1_rows[key])
            same = target_signature(x) == target_signature(y)
            compared += 1
            native_targets_all = {target_signature(o) for o in a0_rows[key]['options'] if o.get('admissible') and int(o['action']['act']) != 0}
            dense_targets_all = {target_signature(o) for o in a1_rows[key]['options'] if o.get('admissible') and int(o['action']['act']) != 0}
            if not native_targets_all <= dense_targets_all:
                nested_violations += 1
            if not same:
                first_diff = key[1]
                first_a0_candidates = sum(1 for o in a0_rows[key]['options'] if o.get('admissible') and int(o['action']['act']) != 0)
                first_a1_candidates = sum(1 for o in a1_rows[key]['options'] if o.get('admissible') and int(o['action']['act']) != 0)
                action_diffs = 1
                # This is the only state at which a direct causal attribution
                # is valid: both arms started from the same simulator state.
                native_targets = {target_signature(o) for o in a0_rows[key]['options'] if o.get('admissible')}
                native_move_targets = {target_signature(o) for o in a0_rows[key]['options'] if o.get('admissible') and int(o['action']['act']) != 0}
                if target_signature(y) not in native_targets and int(y['action']['act']) != 0:
                    direct_new = 1
                a1_targets = {target_signature(o) for o in a1_rows[key]['options'] if o.get('admissible') and int(o['action']['act']) != 0}
                first_nested = int(native_move_targets <= a1_targets)
                break
        route_rows.append({
            'cohort': cohort, 'episode_id': ep,
            'role': manifest.get(ep, {}).get('role', ''),
            'scene_id': manifest.get(ep, {}).get('scene_id', ''),
            'source_cohort': manifest.get(ep, {}).get('cohort', ''),
            'a0_success': int(left['success'] >= 0.999999), 'a1_success': int(right['success'] >= 0.999999),
            'delta_success': int(right['success'] >= 0.999999) - int(left['success'] >= 0.999999),
            'a0_spl': left['spl'], 'a1_spl': right['spl'], 'delta_spl': right['spl'] - left['spl'],
            'a0_ndtw': left['ndtw'], 'a1_ndtw': right['ndtw'], 'delta_ndtw': right['ndtw'] - left['ndtw'],
            'a0_path_length': left['path_length'], 'a1_path_length': right['path_length'], 'delta_path_length': right['path_length'] - left['path_length'],
            'a0_steps': left['steps_taken'], 'a1_steps': right['steps_taken'], 'delta_steps': right['steps_taken'] - left['steps_taken'],
            'a0_collisions': left['collisions'], 'a1_collisions': right['collisions'], 'delta_collisions': right['collisions'] - left['collisions'],
            'first_matched_action_divergence': first_diff if first_diff is not None else '',
            'matched_action_differences': action_diffs, 'matched_prefix_steps': compared,
            'direct_new_candidate_selections': direct_new,
            'first_a0_candidates': first_a0_candidates, 'first_a1_candidates': first_a1_candidates,
            'first_state_a0_nested_in_a1': first_nested,
            'matched_prefix_nested_violations': nested_violations,
            'a0_high_level_steps': left.get('high_level_steps'), 'a1_high_level_steps': right.get('high_level_steps'),
        })
    return {'cohort': cohort, 'a0': a0m, 'a1': a1m, 'routes': route_rows,
            'route_count': len(route_rows),
            'a0_success_routes': sum(r['a0_success'] for r in route_rows),
            'a1_success_routes': sum(r['a1_success'] for r in route_rows),
            'rescued_routes': sum(r['a0_success'] == 0 and r['a1_success'] == 1 for r in route_rows),
            'destroyed_success_routes': sum(r['a0_success'] == 1 and r['a1_success'] == 0 for r in route_rows),
            'routes_with_direct_new_selection': sum(r['direct_new_candidate_selections'] > 0 for r in route_rows),
            'total_direct_new_selections': sum(r['direct_new_candidate_selections'] for r in route_rows),
            'routes_with_action_change': sum(r['matched_action_differences'] > 0 for r in route_rows)}


def main():
    out = BASE / 'analysis_002'; out.mkdir(parents=True, exist_ok=True)
    results = []
    for cohort, split in [('train', 'train'), ('val_unseen', 'val_unseen')]:
        run_cohort = 'train' if cohort == 'train' else 'unseen'
        results.append(summarize(split, BASE / ('runs/%s_a0_v2' % run_cohort),
                                 BASE / ('runs/%s_a1_v6' % run_cohort), run_cohort))
    all_routes = [r for result in results for r in result['routes']]
    with (out / 'per_route.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(all_routes[0])); w.writeheader(); w.writerows(all_routes)
    summary = {'protocol': 'Gate A closed-loop native policy A0/A1', 'oracle_used_for_selection': False,
               'candidate_policy': 'unchanged native ETPNav graph encoder and SAP argmax',
               'arms': {r['cohort']: {k: v for k, v in r.items() if k != 'routes'} for r in results},
               'route_rows': len(all_routes),
               'all_routes': all_routes}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({k: {x: v[x] for x in ('route_count','a0_success_routes','a1_success_routes','rescued_routes','destroyed_success_routes','routes_with_direct_new_selection','total_direct_new_selections')} for k,v in summary['arms'].items()}, indent=2))


if __name__ == '__main__': main()
