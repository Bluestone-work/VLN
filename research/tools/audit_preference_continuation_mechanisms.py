#!/usr/bin/env python3
"""Retrospective execution/continuation census, without policy fitting."""
import argparse
import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from audit_single_intervention import group_rows, key, read_rows
from decompose_continuation_costs import breakdown, stop_details, vector
from analyze_continuation_horizon import movement
from graph_value_feasibility import action_features, entropy, FEATURE_NAMES


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def execution_symptoms(record, config):
    target = vector(record['action']['ghost_pos'])
    endpoint = vector(record['post_pose']['position'])
    residual = float(np.linalg.norm((endpoint - target)[[0, 2]]))
    path = movement(record)
    collisions = sum(p['collided'] is True for p in record['primitives'])
    flags = {'collision': collisions > 0,
             'large_endpoint_deviation': residual >= config['endpoint_deviation_m'],
             'stall': len(record['primitives']) >= config['stall_primitives_min'] and path <= config['stall_path_m_max']}
    return dict(flags, any_symptom=any(flags.values()), horizontal_residual_m=residual,
                vertical_residual_m=float(abs(endpoint[1] - target[1])),
                path_m=path, collision_events=collisions)


def pre_action_features(graph, event):
    options = {o['index']: o for o in graph['options'] if o.get('admissible')}
    native, alternative = options[event['baseline_index']], options[event['alternative_index']]
    if native['action'] != event['baseline_action'] or alternative['action'] != event['alternative_action']:
        raise ValueError('Captured graph action does not match schedule')
    ent, probs = entropy([o['logit'] for o in options.values()])
    probability = dict(zip(options, probs))
    return {'step': graph['high_level_step'], 'candidate_count_including_stop': len(options),
            'native_current_proposal': bool(native['current_proposal']),
            'alternative_current_proposal': bool(alternative['current_proposal']),
            'same_front': native['action']['front_vp'] == alternative['action']['front_vp'],
            'alternative_back_path_empty': not bool(alternative['action'].get('back_path')),
            'native_probability': float(probability[native['index']]),
            'alternative_probability': float(probability[alternative['index']]),
            'policy_entropy': ent,
            'native_features': dict(zip(FEATURE_NAMES, map(float, action_features(graph, native)))),
            'alternative_features': dict(zip(FEATURE_NAMES, map(float, action_features(graph, alternative))))}


def reselection(records, step, displaced_action, stable_drift):
    for record in records[step + 1:]:
        action = record['action']
        if action.get('ghost_vp') != displaced_action['ghost_vp']:
            continue
        drift = float(np.linalg.norm(vector(action['ghost_pos']) - vector(displaced_action['ghost_pos'])))
        return {'found': True, 'step': record['high_level_step'],
                'immediate_next_decision': record['high_level_step'] == step + 1,
                'target_drift_3d_m': drift, 'stable_target': drift <= stable_drift,
                'primitives': len(record['primitives']), 'movement_m': movement(record),
                'action': action}
    return {'found': False, 'immediate_next_decision': False, 'stable_target': False}


def summarize(rows):
    return {'events': len(rows), 'scenes': len({r['scene_id'] for r in rows}),
            'cost_reversal': sum(r['cost_reversal'] for r in rows),
            'native_successes': sum(r['baseline_success'] > 0 for r in rows),
            'success_rescued': sum(r['final_delta']['success'] > 0 for r in rows),
            'success_lost': sum(r['final_delta']['success'] < 0 for r in rows),
            'quality_rescued': sum(r['final_delta']['success'] > 0 and r['final_delta']['ndtw'] >= -1e-6 for r in rows),
            'ndtw_harmed': sum(r['final_delta']['ndtw'] < -1e-6 for r in rows),
            'primitive_delta': sum(r['final_delta']['primitive_action_count'] for r in rows),
            'cost_groups': {g: sum(r['cost_deltas'][g]['primitives'] for r in rows)
                            for g in ['immediate', 'later_navigation', 'later_stop', 'later_other']},
            'native_execution_symptom': sum(r['native_execution']['any_symptom'] for r in rows),
            'alternative_execution_symptom': sum(r['alternative_execution']['any_symptom'] for r in rows),
            'alternative_symptom_counts': {k: sum(r['alternative_execution'][k] for r in rows)
                                            for k in ['collision', 'large_endpoint_deviation', 'stall']},
            'displaced_ghost_reselected': sum(r['reselection']['found'] for r in rows),
            'stable_next_reselection': sum(r['reselection']['stable_target'] and r['reselection']['immediate_next_decision'] for r in rows),
            'budget_stop_after': sum(r['intervention_stop']['budget_stop'] for r in rows)}


def paths_for(config, config_path):
    paths = {config_path, Path(config['protocol']), Path(__file__)}
    paths.update(Path('research/tools') / n for n in [
        'audit_single_intervention.py', 'decompose_continuation_costs.py',
        'analyze_continuation_horizon.py', 'graph_value_feasibility.py', 'route_alignment.py'])
    for batch in config['batches']:
        root = Path(batch['root']); run = root / batch['run']; audit = root / batch['audit']
        paths.update(root / p for p in ['capture_001/manifest.json', 'capture_001/traces/worker_seed100.jsonl',
                                       'capture_001/graph_options/graph_options.jsonl'])
        paths.update(run / p for p in ['manifest.json', 'traces/worker_seed100.jsonl', 'hook/decisions.jsonl'])
        paths.update(audit / p for p in ['summary.json', 'paired_episodes.json'])
    return sorted(paths)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--register', action='store_true')
    args = parser.parse_args(); cfg = json.loads(args.config.read_text())
    root = Path(cfg['output']); regpath = root / 'registration.json'
    if args.register:
        root.mkdir(exist_ok=False)
        value = {'created_utc': datetime.now(timezone.utc).isoformat(),
                 'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip(),
                 'python': sys.version, 'config': cfg,
                 'scope': 'Analysis specification frozen after navigation outcomes were known; not prospective outcome evidence.',
                 'hashes': {str(p): digest(p) for p in paths_for(cfg, args.config)}}
        with regpath.open('x') as stream:
            json.dump(value, stream, indent=2); stream.write('\n')
        print('Registered retrospective specification:', str(regpath)); return
    registration = json.loads(regpath.read_text())
    for path, expected in registration['hashes'].items():
        if digest(path) != expected:
            raise ValueError('Frozen analysis input/source changed: ' + path)
    events = []; batches = {}; cache = {}
    for batch in cfg['batches']:
        source = Path(batch['root']); run = source / batch['run']; auditdir = source / batch['audit']
        audit = json.loads((auditdir / 'summary.json').read_text())
        if not audit['all_required_gates_passed'] or audit['mode'] != 'enabled':
            raise ValueError('Missing valid intervention audit')
        manifest = json.loads((run / 'manifest.json').read_text())
        schedule = manifest['schedule']['events']
        if len(schedule) != batch['expected_events']:
            raise ValueError('Scheduled population changed')
        if str(source) not in cache:
            cache[str(source)] = (group_rows(read_rows(source / 'capture_001/traces/worker_seed100.jsonl')),
                                  read_rows(source / 'capture_001/graph_options/graph_options.jsonl'))
        before, graphs = cache[str(source)]
        after = group_rows(read_rows(run / 'traces/worker_seed100.jsonl'))
        hooks = read_rows(run / 'hook/decisions.jsonl')
        pairs = {r['episode_id']: r for r in json.loads((auditdir / 'paired_episodes.json').read_text())}
        if set(before) != set(after) or set(before) != set(pairs) or len(before) != 96:
            raise ValueError('Paired route population changed')
        batch_rows = []
        for event in schedule:
            ep = event['episode_id']; step = event['high_level_step']; old = before[ep]; new = after[ep]
            if old[:step] != new[:step] or old[step]['pre_pose'] != new[step]['pre_pose'] or old[step]['pre_rng'] != new[step]['pre_rng']:
                raise ValueError('Intervention prefix changed')
            if old[step]['action'] != event['baseline_action'] or new[step]['action'] != event['alternative_action']:
                raise ValueError('Actual intervention action mismatch')
            left, right = breakdown(old, step), breakdown(new, step)
            delta = {g: {m: right[g][m] - left[g][m] for m in left[g]} for g in left}
            final = pairs[ep]['delta']
            if sum(x['primitives'] for x in delta.values()) != final['primitive_action_count']:
                raise ValueError('Primitive accounting mismatch')
            if sum(x['collision_events'] for x in delta.values()) != final['collision_events']:
                raise ValueError('Collision accounting mismatch')
            if abs(sum(x['movement_m'] for x in delta.values()) - final['path_length']) > 1e-4:
                raise ValueError('Path accounting mismatch')
            reselect = reselection(new, step, event['baseline_action'], cfg['stable_target_drift_m'])
            if reselect['immediate_next_decision']:
                reselect['first_two_vs_native_option_primitive_delta'] = len(new[step]['primitives']) + reselect['primitives'] - len(old[step]['primitives'])
            item = {'batch': batch['name'], 'episode_id': ep, 'scene_id': event['scene_id'],
                    'pre_action': pre_action_features(graphs[key(old[step])], event),
                    'baseline_success': pairs[ep]['baseline']['success'],
                    'cost_reversal': delta['immediate']['primitives'] < 0 and final['primitive_action_count'] > 0,
                    'cost_deltas': delta, 'native_execution': execution_symptoms(old[step], cfg),
                    'alternative_execution': execution_symptoms(new[step], cfg), 'reselection': reselect,
                    'baseline_stop': stop_details(old, graphs), 'intervention_stop': stop_details(new, hooks),
                    'final_delta': final, 'accounting_passed': True}
            events.append(item); batch_rows.append(item)
        categorical = ['native_current_proposal', 'alternative_current_proposal', 'same_front', 'alternative_back_path_empty']
        strata = {name: {str(v): summarize([r for r in batch_rows if r['pre_action'][name] == v])
                          for v in [False, True]} for name in categorical}
        reversals = [r for r in batch_rows if r['cost_reversal']]
        batches[batch['name']] = {'role': batch['role'], 'population_routes': len(before),
                                 'all_events': summarize(batch_rows), 'cost_reversals': summarize(reversals),
                                 'other_events': summarize([r for r in batch_rows if not r['cost_reversal']]),
                                 'structural_strata': strata,
                                 'execution_commit': manifest['git_commit'], 'simulator_seed': manifest['config']['seed'],
                                 'checkpoint': manifest['config']['checkpoint'], 'hardware': manifest['hardware']}
    output = root / 'analysis_001'; output.mkdir(exist_ok=False)
    summary = {'experiment_id': cfg['experiment_id'], 'all_accounting_checks_passed': True,
               'analysis_type': cfg['analysis_type'], 'batches': batches,
               'training_gate': 'NO-GO unchanged; observational audit does not establish a remedy',
               'registration_sha256': digest(regpath), 'source_sha256': digest(__file__)}
    for name, value in [('summary.json', summary), ('events.json', events)]:
        with (output / name).open('x') as stream:
            json.dump(value, stream, indent=2); stream.write('\n')
    rows = []
    for r in events:
        row = {'batch': r['batch'], 'episode_id': r['episode_id'], 'scene_id': r['scene_id'],
               'step': r['pre_action']['step'], 'cost_reversal': r['cost_reversal'],
               'native_success': r['baseline_success'],
               'alternative_execution_symptom': r['alternative_execution']['any_symptom'],
               'stable_next_reselection': r['reselection']['stable_target'] and r['reselection']['immediate_next_decision'],
               'next_target_drift_m': r['reselection'].get('target_drift_3d_m'),
               'success_delta': r['final_delta']['success'], 'ndtw_delta': r['final_delta']['ndtw']}
        row.update({g + '_primitive_delta': v['primitives'] for g, v in r['cost_deltas'].items()})
        row['episode_primitive_delta'] = r['final_delta']['primitive_action_count']; rows.append(row)
    with (output / 'events.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(json.dumps({name: {k: value[k] for k in ['all_events', 'cost_reversals']} for name, value in batches.items()}, indent=2))


if __name__ == '__main__':
    main()
