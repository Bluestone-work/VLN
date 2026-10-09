#!/usr/bin/env python3
"""Paired route and provenance analysis for the 99-route A0/A1 replication."""
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'research/results/gate_a_native_policy_replication_001'
EVAL = ROOT / 'data/logs/eval_results'
OUT = BASE / 'analysis_001'


def read_jsonl(path):
    return [json.loads(line) for line in path.open()]


def target_signature(option):
    action = option['action']
    if action['act'] == 0:
        return ('stop', tuple(round(float(x), 5) for x in action['stop_pos']['__array__']))
    return ('move', tuple(round(float(x), 5) for x in action['ghost_pos']['__array__']))


def selected(row):
    return next(option for option in row['options'] if int(option['index']) == int(row['effective_index']))


def metric_path(cohort, arm, per_episode=False):
    suffix = '_per_episode' if per_episode else ''
    if cohort == 'train':
        split = 'train'
    else:
        split = 'val_unseen'
    name = 'gate_a_native_policy_replication_%s_%s' % (cohort, arm)
    if per_episode:
        return EVAL / name / ('stats_ep_ckpt_59_%s_r0_w1.json' % split)
    return EVAL / name / ('stats_ckpt_59_%s.json' % split)


def analyze_cohort(cohort, count):
    a0_dir = BASE / 'runs' / ('%s_a0' % cohort)
    a1_dir = BASE / 'runs' / ('%s_a1' % cohort)
    a0_rows = {(str(row['episode_id']), int(row['high_level_step'])): row for row in read_jsonl(a0_dir / 'graph_options/graph_options.jsonl')}
    a1_rows = {(str(row['episode_id']), int(row['high_level_step'])): row for row in read_jsonl(a1_dir / 'graph_options/graph_options.jsonl')}
    a0_eps = {str(k): value for k, value in json.load(metric_path(cohort, 'a0', True).open()).items()}
    a1_eps = {str(k): value for k, value in json.load(metric_path(cohort, 'a1', True).open()).items()}
    manifest = {str(row['episode_id']): row for row in json.load((BASE / 'manifest.json').open())['route_rows']}
    routes = []
    for episode_id in sorted(a0_eps, key=lambda value: int(value)):
        left, right = a0_eps[episode_id], a1_eps[episode_id]
        keys = sorted(key for key in a0_rows if key[0] == episode_id and key in a1_rows)
        first_divergence = ''
        direct_new = 0
        prefix_steps = 0
        nested_violations = 0
        a0_candidate_counts = []
        a1_candidate_counts = []
        new_candidate_counts = []
        a0_selected_rank = ''
        a1_selected_rank = ''
        for key in keys:
            x, y = a0_rows[key], a1_rows[key]
            a0_moves = {target_signature(option) for option in x['options'] if option.get('admissible') and option['action']['act'] != 0}
            a1_moves = {target_signature(option) for option in y['options'] if option.get('admissible') and option['action']['act'] != 0}
            if not a0_moves <= a1_moves:
                nested_violations += 1
            prefix_steps += 1
            a0_candidate_counts.append(len(x.get('candidate_metadata', [])))
            a1_candidate_counts.append(len(y.get('candidate_metadata', [])))
            new_candidate_counts.append(int(y.get('dense_candidate_provenance', {}).get('new_a1_count', 0)))
            if target_signature(selected(x)) != target_signature(selected(y)):
                first_divergence = key[1]
                if target_signature(selected(y)) not in {target_signature(option) for option in x['options'] if option.get('admissible')} and selected(y)['action']['act'] != 0:
                    direct_new = 1
                a0_valid = [option for option in x['options'] if option.get('admissible')]
                a1_valid = [option for option in y['options'] if option.get('admissible')]
                a0_selected_rank = 1 + sorted(a0_valid, key=lambda option: float(option['logit']), reverse=True).index(selected(x))
                a1_selected_rank = 1 + sorted(a1_valid, key=lambda option: float(option['logit']), reverse=True).index(selected(y))
                break
        routes.append({
            'cohort': cohort, 'episode_id': episode_id, 'role': 'metadata_control_replication',
            'scene_id': manifest[episode_id]['scene_id'],
            'a0_success': int(left['success'] >= 0.999999), 'a1_success': int(right['success'] >= 0.999999),
            'delta_success': int(right['success'] >= 0.999999) - int(left['success'] >= 0.999999),
            'a0_spl': left['spl'], 'a1_spl': right['spl'], 'delta_spl': right['spl'] - left['spl'],
            'a0_ndtw': left['ndtw'], 'a1_ndtw': right['ndtw'], 'delta_ndtw': right['ndtw'] - left['ndtw'],
            'a0_path_length': left['path_length'], 'a1_path_length': right['path_length'], 'delta_path_length': right['path_length'] - left['path_length'],
            'a0_steps': left['steps_taken'], 'a1_steps': right['steps_taken'], 'delta_steps': right['steps_taken'] - left['steps_taken'],
            'a0_collisions': left['collisions'], 'a1_collisions': right['collisions'], 'delta_collisions': right['collisions'] - left['collisions'],
            'a0_high_level_steps': left.get('high_level_steps'), 'a1_high_level_steps': right.get('high_level_steps'),
            'first_matched_action_divergence': first_divergence, 'matched_prefix_steps': prefix_steps,
            'direct_new_candidate_at_first_divergence': direct_new,
            'matched_prefix_nested_violations': nested_violations,
            'mean_a0_heatmap_candidates': statistics.mean(a0_candidate_counts) if a0_candidate_counts else 0.0,
            'mean_a1_heatmap_candidates': statistics.mean(a1_candidate_counts) if a1_candidate_counts else 0.0,
            'mean_new_a1_candidates': statistics.mean(new_candidate_counts) if new_candidate_counts else 0.0,
            'first_divergence_a0_rank': a0_selected_rank,
            'first_divergence_a1_rank': a1_selected_rank,
        })
    return {'cohort': cohort, 'metrics_a0': json.load(metric_path(cohort, 'a0').open()),
            'metrics_a1': json.load(metric_path(cohort, 'a1').open()), 'routes': routes}


def aggregate(rows):
    n = len(rows)
    if not n:
        return {'route_count': 0, 'scene_count': 0}
    return {
        'route_count': n,
        'scene_count': len(set(row['scene_id'] for row in rows)),
        'a0_success_routes': sum(row['a0_success'] for row in rows),
        'a1_success_routes': sum(row['a1_success'] for row in rows),
        'a0_sr': sum(row['a0_success'] for row in rows) / n if n else None,
        'a1_sr': sum(row['a1_success'] for row in rows) / n if n else None,
        'rescued_routes': sum(row['a0_success'] == 0 and row['a1_success'] == 1 for row in rows),
        'destroyed_success_routes': sum(row['a0_success'] == 1 and row['a1_success'] == 0 for row in rows),
        'mean_delta_spl': statistics.mean(row['delta_spl'] for row in rows),
        'mean_delta_ndtw': statistics.mean(row['delta_ndtw'] for row in rows),
        'mean_delta_path_length': statistics.mean(row['delta_path_length'] for row in rows),
        'mean_delta_steps': statistics.mean(row['delta_steps'] for row in rows),
        'mean_delta_collisions': statistics.mean(row['delta_collisions'] for row in rows),
        'direct_new_first_divergence_routes': sum(row['direct_new_candidate_at_first_divergence'] for row in rows),
        'matched_prefix_nested_violations': sum(row['matched_prefix_nested_violations'] for row in rows),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    results = [analyze_cohort('train', 55), analyze_cohort('unseen', 44)]
    rows = [row for result in results for row in result['routes']]
    with (OUT / 'per_route.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    by_a0 = {
        'a0_success': [row for row in rows if row['a0_success']],
        'a0_failure': [row for row in rows if not row['a0_success']],
    }
    by_scene = defaultdict(list)
    for row in rows:
        by_scene[row['scene_id']].append(row)
    scene_rows = []
    for scene_id, scene_group in sorted(by_scene.items()):
        scene_summary = aggregate(scene_group)
        scene_summary.update({'scene_id': scene_id, 'episode_ids': '|'.join(sorted(row['episode_id'] for row in scene_group))})
        scene_rows.append(scene_summary)
    with (OUT / 'per_scene.csv').open('w', newline='') as stream:
        fields = list(scene_rows[0])
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(scene_rows)
    retained = [row for row in rows if row['a1_success']]
    harmed = [row for row in rows if not row['a1_success']]
    summary = {
        'protocol': '99-route outcome-blind native-success control replication',
        'oracle_used_for_selection': False,
        'candidate_policy': 'A0 native or A1 A0-union-dense NMS; unchanged ETPNav navigator/controller',
        'cohort': {'route_count': len(rows), 'scene_count': len(set(row['scene_id'] for row in rows)), 'seed': 100,
                   'checkpoint': 'data/logs/checkpoints/release_r2r/ckpt.iter12000.pth'},
        'groups': {'all': aggregate(rows), 'a0_success': aggregate(by_a0['a0_success']), 'a0_failure': aggregate(by_a0['a0_failure']),
                   'a1_retained_success': aggregate(retained), 'a1_destroyed_success': aggregate(harmed)},
        'provenance': {
            'all_mean_a0_heatmap_candidates': statistics.mean(row['mean_a0_heatmap_candidates'] for row in rows),
            'all_mean_a1_heatmap_candidates': statistics.mean(row['mean_a1_heatmap_candidates'] for row in rows),
            'all_mean_new_a1_candidates': statistics.mean(row['mean_new_a1_candidates'] for row in rows),
            'retained_mean_a1_heatmap_candidates': statistics.mean(row['mean_a1_heatmap_candidates'] for row in retained),
            'harmed_mean_a1_heatmap_candidates': statistics.mean(row['mean_a1_heatmap_candidates'] for row in harmed),
            'retained_new_candidate_first_divergence_rate': statistics.mean(row['direct_new_candidate_at_first_divergence'] for row in retained),
            'harmed_new_candidate_first_divergence_rate': statistics.mean(row['direct_new_candidate_at_first_divergence'] for row in harmed),
        },
        'arm_metrics': {result['cohort']: {'a0': result['metrics_a0'], 'a1': result['metrics_a1']} for result in results},
        'route_rows': rows,
        'interpretation': {
            'selection_is_outcome_blind': True,
            'native_success_harm_is_estimated_after_freeze': True,
            'next_step': 'combine with prior 16-control cohort only as descriptive replication; do not train a selector',
        },
    }
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary['groups'], indent=2))


if __name__ == '__main__':
    main()
