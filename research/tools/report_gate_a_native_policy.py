#!/usr/bin/env python3
"""Create stable summaries for the frozen online A0/A1 policy experiment."""
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'research/results/gate_a_native_policy'
ANALYSIS = BASE / 'analysis_002'


def mean(rows, key):
    vals = [float(r[key]) for r in rows if r[key] not in ('', None)]
    return sum(vals) / len(vals) if vals else None


def summarize(rows):
    n = len(rows)
    a0 = sum(int(r['a0_success']) for r in rows)
    a1 = sum(int(r['a1_success']) for r in rows)
    return {
        'route_count': n,
        'scene_count': len(set(r['scene_id'] for r in rows)),
        'a0_success_routes': a0,
        'a1_success_routes': a1,
        'a0_sr': a0 / n if n else None,
        'a1_sr': a1 / n if n else None,
        'rescued_routes': sum(int(r['a0_success']) == 0 and int(r['a1_success']) == 1 for r in rows),
        'destroyed_success_routes': sum(int(r['a0_success']) == 1 and int(r['a1_success']) == 0 for r in rows),
        'delta_spl_mean': mean(rows, 'delta_spl'),
        'delta_ndtw_mean': mean(rows, 'delta_ndtw'),
        'delta_path_length_mean': mean(rows, 'delta_path_length'),
        'delta_steps_mean': mean(rows, 'delta_steps'),
        'delta_collisions_mean': mean(rows, 'delta_collisions'),
        'direct_new_first_divergence': sum(int(r['direct_new_candidate_selections']) for r in rows),
        'first_state_nested_count': sum(int(r['first_state_a0_nested_in_a1']) for r in rows),
    }


def proposal_stats():
    out = {}
    for split, arm in [('train', 'a0_v2'), ('train', 'a1_v6'),
                       ('unseen', 'a0_v2'), ('unseen', 'a1_v6')]:
        path = BASE / 'runs' / ('%s_%s' % (split, arm)) / 'graph_options/graph_options.jsonl'
        records = [json.loads(line) for line in path.open()]
        counts = [len(r.get('candidate_metadata', [])) for r in records]
        out['%s_%s' % (split, arm)] = {
            'decision_rows': len(records),
            'route_count': len(set(str(r['episode_id']) for r in records)),
            'mean_heatmap_proposals': statistics.mean(counts),
            'min_heatmap_proposals': min(counts), 'max_heatmap_proposals': max(counts),
        }
    return out


def main():
    rows = list(csv.DictReader((ANALYSIS / 'per_route.csv').open()))
    for r in rows:
        for k in ('a0_success', 'a1_success', 'direct_new_candidate_selections',
                  'first_state_a0_nested_in_a1'):
            r[k] = int(r[k] or 0)
    per_scene = []
    by_scene = defaultdict(list)
    for r in rows:
        by_scene[r['scene_id']].append(r)
    for scene, group in sorted(by_scene.items()):
        s = summarize(group)
        s.update({'scene_id': scene, 'roles': sorted(set(r['role'] for r in group)),
                  'episode_ids': sorted(r['episode_id'] for r in group)})
        per_scene.append(s)
    with (ANALYSIS / 'per_scene.csv').open('w', newline='') as f:
        fields = ['scene_id', 'roles', 'episode_ids'] + [k for k in per_scene[0] if k not in ('scene_id', 'roles', 'episode_ids')]
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r in per_scene:
            r = dict(r); r['roles'] = '|'.join(r['roles']); r['episode_ids'] = '|'.join(r['episode_ids']); w.writerow(r)
    groups = {}
    for label, group in [('all', rows), ('gate_a_unresolved', [r for r in rows if r['role'] == 'gate_a_unresolved']),
                         ('metadata_control', [r for r in rows if r['role'] == 'metadata_control'])]:
        groups[label] = summarize(group)
    key_routes = {ep: next(r for r in rows if r['episode_id'] == ep) for ep in ('8343', '1052', '1584') if any(r['episode_id'] == ep for r in rows)}
    summary = {
        'protocol': 'Gate A dense candidates with unchanged native ETPNav policy',
        'oracle_used_for_selection': False,
        'candidate_policy': 'A0 native NMS or A1 native-union-dense NMS; unchanged graph encoder, SAP argmax, controller, STOP and budget',
        'a0': {'max_predictions': 5, 'sigma': [7.0, 5.0]},
        'a1': {'construction': 'A0 union dense NMS', 'max_predictions': 12, 'sigma': [4.0, 3.0]},
        'cohort': {'route_count': len(rows), 'scene_count': len(set(r['scene_id'] for r in rows)), 'seed': 100,
                   'checkpoint': 'data/logs/checkpoints/release_r2r/ckpt.iter12000.pth'},
        'groups': groups,
        'proposal_stats': proposal_stats(),
        'key_routes': key_routes,
        'interpretation': {
            'proposal_coverage_signal': 'supported on frozen Gate-A unresolved routes: A1 rescues routes while A0 remains unsuccessful',
            'control_harm_present': True,
            'online_result_is_not_oracle': True,
            'next_gate': 'replicate on a larger outcome-blind native-success cohort before any navigator modification',
        },
    }
    (ANALYSIS / 'policy_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(groups, indent=2))


if __name__ == '__main__':
    main()
