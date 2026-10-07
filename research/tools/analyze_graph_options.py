#!/usr/bin/env python3
"""Same-state full-option progress, with explicit STOP/budget/cost controls."""
import argparse
import collections
import hashlib
import json
from pathlib import Path

import numpy as np
from audit_ranker_validity import cluster_ci


def analyze_state(row, tolerance):
    if not row['selected_sentinel_passed']:
        raise ValueError('Uncalibrated selected action')
    selected = next(o for o in row['options'] if o['index'] == row['effective_index'])
    if selected['action_type'] == 0:
        return None
    if row['budget_stop'] or row['forced_stop']:
        raise ValueError('Non-STOP option executed beyond the native episode limit')
    actions = [o for o in row['options'] if o['action_type'] == 4]
    if len({o['graph_id'] for o in actions}) != len(actions):
        raise ValueError('Duplicate graph IDs in candidate labels')
    best = max(actions, key=lambda o: o['progress_m'])
    count_control = [o for o in actions if o['primitive_events'] <= selected['primitive_events']]
    path_control = [o for o in actions if o['motion_path_m'] <= selected['motion_path_m'] + tolerance]
    current_control = [o for o in actions if o['current_proposal'] or o['index'] == selected['index']]
    both_control = [o for o in count_control if o['motion_path_m'] <= selected['motion_path_m'] + tolerance]
    result = {'scene_id': row['scene_id'], 'episode_id': row['episode_id'], 'trajectory_id': row['trajectory_id'],
              'high_level_step': row['high_level_step'], 'nonstop_options': len(actions),
              'selected_index': selected['index'], 'best_index': best['index'],
              'selected_progress_m': selected['progress_m'], 'best_progress_m': best['progress_m'],
              'gain_m': best['progress_m'] - selected['progress_m'],
              'primitive_capped_gain_m': max(o['progress_m'] for o in count_control) - selected['progress_m'],
              'path_capped_gain_m': max(o['progress_m'] for o in path_control) - selected['progress_m'],
              'both_costs_capped_gain_m': max(o['progress_m'] for o in both_control) - selected['progress_m'],
              'current_plus_selected_gain_m': max(o['progress_m'] for o in current_control) - selected['progress_m'],
              'historical_extra_gain_m': best['progress_m'] - max(o['progress_m'] for o in current_control),
              'selected_collisions': selected['collision_events'], 'selected_primitives': selected['primitive_events'],
              'best_primitives': best['primitive_events'], 'selected_path_m': selected['motion_path_m'],
              'best_path_m': best['motion_path_m'], 'best_historical': not best['current_proposal'],
              'selected_target_error_horizontal_m': selected['target_error_horizontal_m']}
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--probe-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    probe = json.loads((args.probe_dir / 'summary.json').read_text())
    manifest = json.loads((args.probe_dir / 'manifest.json').read_text())
    if not probe['all_required_gates_passed'] or not probe['all_admissible_options_complete']:
        raise ValueError('Full-option completeness and replay gates must pass')
    cfg = manifest['config']
    decisions = [json.loads(l) for l in (args.probe_dir / 'decisions.jsonl').open()]
    if len(decisions) != probe['decisions']:
        raise ValueError('Incomplete probe output')
    tolerance = cfg['gates']['progress_tolerance_m']
    states = [analyze_state(row, tolerance) for row in decisions]
    states = [row for row in states if row is not None]
    if not states:
        raise ValueError('No eligible non-STOP states')
    measures = ['gain_m', 'primitive_capped_gain_m', 'path_capped_gain_m', 'both_costs_capped_gain_m',
                'current_plus_selected_gain_m', 'historical_extra_gain_m']
    metrics = {}
    for measure in measures:
        y = [r[measure] for r in states]
        metrics[measure] = {'mean': float(np.mean(y)), 'median': float(np.median(y)),
                            'p90': float(np.percentile(y, 90)), 'max': float(np.max(y)),
                            'positive_states': sum(v > tolerance for v in y),
                            'states_at_least_025m': sum(v >= cfg['analysis'].get('material_state_gain_m', 0.25) for v in y),
                            'episode_bootstrap': cluster_ci(y, [r['episode_id'] for r in states],
                                cfg['analysis']['bootstrap_seed'], cfg['analysis']['bootstrap_samples']),
                            'scene_bootstrap': cluster_ci(y, [r['scene_id'] for r in states],
                                cfg['analysis']['bootstrap_seed'], cfg['analysis']['bootstrap_samples'])}
    scenes = collections.defaultdict(list)
    for row in states:
        scenes[row['scene_id']].append(row)
    by_scene = {scene: {'states': len(rows), 'episodes': len(set(r['episode_id'] for r in rows)),
                        'gain_m': float(np.mean([r['gain_m'] for r in rows])),
                        'both_costs_capped_gain_m': float(np.mean([r['both_costs_capped_gain_m'] for r in rows]))}
                for scene, rows in scenes.items()}
    result = {'experiment_id': cfg['experiment_id'], 'probe_dir': str(args.probe_dir),
              'episodes': probe['episodes'], 'scenes': probe['scenes'], 'total_decisions': len(decisions),
              'eligible_nonstop_states': len(states), 'stop_states_excluded': len(decisions) - len(states),
              'masked_budget_options_skipped': probe['masked_budget_options_skipped'],
              'mean_selected_progress_m': float(np.mean([r['selected_progress_m'] for r in states])),
              'mean_best_progress_m': float(np.mean([r['best_progress_m'] for r in states])),
              'cost_controls_include_selected': True, 'metrics': metrics, 'by_scene': by_scene,
              'best_uses_more_primitives_states': sum(r['best_primitives'] > r['selected_primitives'] for r in states),
              'best_uses_longer_path_states': sum(r['best_path_m'] > r['selected_path_m'] + tolerance for r in states),
              'examples_by_largest_raw_gain': sorted(states, key=lambda r: -r['gain_m'])[:5],
              'privileged_analysis_only': True, 'learned_model_evaluated': False,
              'limits': 'Local goal-progress comparisons, not episode success improvement or a certified optimal bound; bootstrap is descriptive, not independent training seeds.',
              'hashes': {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in [Path(__file__),
                  args.probe_dir / 'manifest.json', args.probe_dir / 'summary.json', args.probe_dir / 'decisions.jsonl']}}
    (args.output_dir / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    with (args.output_dir / 'states.jsonl').open('x') as stream:
        for row in states:
            stream.write(json.dumps(row, sort_keys=True) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ['examples_by_largest_raw_gain', 'hashes']}, indent=2))


if __name__ == '__main__':
    main()
