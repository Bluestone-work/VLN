#!/usr/bin/env python3
"""Exhaustively census captured graph actions on baseline-failed routes.

This is a frozen analysis of complete one-option branches. It does not claim
that local branch outcomes are full-episode counterfactuals or causal labels.
"""
import argparse
import hashlib
import json
import subprocess
from collections import Counter, OrderedDict
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_unique(path, key):
    rows = OrderedDict()
    with Path(path).open() as stream:
        for line in stream:
            row = json.loads(line)
            k = key(row)
            if k in rows:
                raise ValueError('Duplicate row: {}'.format(k))
            rows[k] = row
    return rows


def key(row):
    return (str(row['scene_id']), str(row['episode_id']), int(row['high_level_step']))


def critical(option, rules):
    collision = int(option.get('collision_events', 0)) >= rules['collision_events_min']
    endpoint = float(option.get('target_error_horizontal_m', 0.0)) >= rules['target_error_horizontal_m_min']
    stall = (int(option.get('primitive_events', 0)) >= rules['stall_primitive_events_min']
             and float(option.get('motion_path_m', 0.0)) <= rules['stall_motion_path_m_max'])
    return {'collision': collision, 'large_endpoint_deviation': endpoint,
            'stall': stall, 'any_execution_symptom': collision or endpoint or stall}


def analyze_cohort(spec, rules):
    capture = Path(spec['capture_dir']); probe = Path(spec['probe_dir']); nonint = Path(spec['noninterference'])
    capture_manifest = json.loads((capture / 'manifest.json').read_text())
    probe_summary = json.loads((probe / 'summary.json').read_text())
    nonint_summary = json.loads((nonint / 'summary.json').read_text())
    if not probe_summary.get('all_required_gates_passed') or not probe_summary.get('all_admissible_options_complete'):
        raise ValueError('Probe completeness gate failed for {}'.format(spec['name']))
    if not nonint_summary.get('baseline_noninterference_gate_passed'):
        raise ValueError('Baseline noninterference gate failed for {}'.format(spec['name']))
    metrics = json.loads((nonint / 'capture_episodes.json').read_text())
    failed = {str(ep): value for ep, value in metrics.items() if not bool(value['success'])}
    graph_path = capture / 'graph_options/graph_options.jsonl'
    decision_path = probe / 'decisions.jsonl'
    graphs = read_unique(graph_path, key)
    decisions = read_unique(decision_path, key)
    if set(graphs) != set(decisions):
        raise ValueError('Graph/branch decision keys differ for {}'.format(spec['name']))
    all_route_keys = {k for k in decisions if k[1] in failed}
    rows = []
    candidate_rows = []
    route_counts = OrderedDict()
    counts = Counter()
    for k in sorted(all_route_keys, key=lambda x: (x[1], x[2])):
        row = decisions[k]
        graph = graphs[k]
        options = list(row['options'])
        if not options or not row.get('selected_sentinel_passed'):
            raise ValueError('Missing complete options or selected sentinel at {}'.format(k))
        selected_index = int(row['effective_index'])
        selected = next(o for o in options if int(o['index']) == selected_index)
        symptoms = critical(selected, rules)
        terminal = bool(row.get('forced_stop') or selected.get('action_type') == 0 or int(k[2]) == 14)
        forward = [o for o in options if int(o.get('action_type', 4)) == 4]
        alt = [o for o in forward if int(o['index']) != selected_index]
        if terminal:
            counts['terminal_critical_decisions'] += 1
        if not alt:
            counts['states_without_alternative_forward_action'] += 1
        for name, value in symptoms.items():
            if value:
                counts[name] += 1
        state = {
            'cohort': spec['name'], 'scene_id': k[0], 'episode_id': k[1],
            'high_level_step': k[2], 'trajectory_id': row['trajectory_id'],
            'baseline_episode_success': False, 'selected_index': selected_index,
            'selected_graph_id': selected.get('graph_id'), 'selected_progress_m': selected.get('progress_m'),
            'selected_distance_after_m': selected.get('distance_after'),
            'selected_primitive_events': selected.get('primitive_events'),
            'selected_collision_events': selected.get('collision_events'),
            'selected_target_error_horizontal_m': selected.get('target_error_horizontal_m'),
            'selected_motion_path_m': selected.get('motion_path_m'),
            'selected_symptoms': symptoms, 'terminal_critical': terminal,
            'candidate_count_total': len(options), 'candidate_count_forward': len(forward),
            'alternative_forward_count': len(alt),
            'candidate_coverage_empty': len(forward) == 0,
        }
        # Complete option set is retained, with no outcome-based filtering.
        for option in options:
            c = critical(option, rules)
            better = (float(option.get('progress_m', 0.0)) - float(selected.get('progress_m', 0.0))
                      >= rules['material_gain_m'])
            lower_cost = (int(option.get('primitive_events', 0)) <= int(selected.get('primitive_events', 0))
                          and float(option.get('motion_path_m', 0.0)) <= float(selected.get('motion_path_m', 0.0)) + rules['cost_tolerance_m'])
            useful_local = int(option.get('action_type', 4)) == 4 and int(option['index']) != selected_index and better and lower_cost
            candidate = dict(option)
            candidate.update({'cohort': spec['name'], 'scene_id': k[0], 'episode_id': k[1],
                              'high_level_step': k[2], 'selected_index': selected_index,
                              'selected': int(option['index']) == selected_index,
                              'execution_symptoms': c, 'local_cost_capped_gain_m':
                              float(option.get('progress_m', 0.0)) - float(selected.get('progress_m', 0.0)),
                              'local_cost_capped_useful': useful_local})
            candidate_rows.append(candidate)
            if useful_local:
                counts['local_cost_capped_alternative'] += 1
        rows.append(state)
        route_counts.setdefault(k[1], {'scene_id': k[0], 'decisions': 0, 'critical_decisions': 0,
                                       'forward_options': 0, 'states_without_alternative_forward_action': 0})
        route_counts[k[1]]['decisions'] += 1
        route_counts[k[1]]['critical_decisions'] += int(symptoms['any_execution_symptom'] or terminal)
        route_counts[k[1]]['forward_options'] += len(alt)
        route_counts[k[1]]['states_without_alternative_forward_action'] += int(len(alt) == 0)
    return {
        'name': spec['name'], 'episodes_total': len(metrics), 'failed_episodes': len(failed),
        'failed_episode_ids': sorted(failed), 'captured_failed_route_decisions': len(rows),
        'captured_failed_route_options': len(candidate_rows), 'counts': dict(counts),
        'route_counts': route_counts, 'states': rows, 'candidates': candidate_rows,
        'source_hashes': {str(p): digest(p) for p in [capture / 'manifest.json', graph_path,
                                                        probe / 'summary.json', decision_path,
                                                        nonint / 'summary.json', nonint / 'capture_episodes.json']},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    reports = [analyze_cohort(spec, config['critical_rules']) for spec in config['cohorts']]
    out = Path(config['output_dir']); out.mkdir(parents=True, exist_ok=False)
    states = [state for report in reports for state in report.pop('states')]
    candidates = [candidate for report in reports for candidate in report.pop('candidates')]
    for name, records in [('states', states), ('candidates', candidates)]:
        with (out / (name + '.jsonl')).open('x') as stream:
            for record in records:
                stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + '\n')
    summary = {'experiment_id': config['experiment_id'], 'status': 'complete',
               'post_hoc': True, 'no_new_rollout': True, 'no_model_trained': True,
               'privileged_analysis_only': True, 'controls': config['controls'],
               'cohorts': reports, 'total_failed_route_decisions': len(states),
               'total_failed_route_options': len(candidates),
               'limits': ['Complete captured option execution is exhaustive over the retained graph action set.',
                          'These are one-option outcomes from native states, not complete episode-return counterfactuals.',
                          'Execution symptoms and local alternatives are diagnostic evidence, not causal failure labels.',
                          'Proposal coverage is measured only over exposed graph actions; unseen waypoints are not tested.'],
               'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip(),
               'config_sha256': digest(args.config)}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({'failed_route_decisions': len(states), 'failed_route_options': len(candidates),
                      'cohorts': {x['name']: {'failed_episodes': x['failed_episodes'],
                                               'decisions': x['captured_failed_route_decisions'],
                                               'options': x['captured_failed_route_options'], 'counts': x['counts']}
                                  for x in reports}}, indent=2))


if __name__ == '__main__':
    main()
