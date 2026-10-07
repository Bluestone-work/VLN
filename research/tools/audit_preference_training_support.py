#!/usr/bin/env python3
"""In-sample frozen-score support audit; no fitting or simulator rollout."""
import argparse
import collections
import csv
import hashlib
import itertools
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from graph_value_feasibility import FEATURE_NAMES, METRICS, TOL, action_features, dominates
from prepare_ranker_intervention_schedule import admissible_moves, first_disagreement


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def relation(a, b):
    if dominates(a, b):
        return 'strict_improvement'
    if dominates(b, a):
        return 'strict_degradation'
    if all(abs(a[k] - b[k]) <= (0 if k == 'primitive_action_count' else TOL) for k in METRICS):
        return 'tie'
    return 'mixed'


def pair_kind(left_stop, right_stop, left_index, right_index, native_index):
    if left_stop or right_stop:
        return 'stop_involving'
    return 'native_move' if native_index in [left_index, right_index] else 'two_non_native_moves'


def chosen_index(graph, model):
    native = graph['effective_index']
    if native <= 0 or graph['budget_stop'] or graph['no_vp_left']:
        return native, 'protected_stop_or_budget'
    choice = first_disagreement([graph], model)
    return (choice[2]['index'], 'changed') if choice else (native, 'native_abstention')


def state_summary(rows):
    supported = [r for r in rows if r['supported']]
    changed = [r for r in supported if r['chosen_index'] != r['native_index']]
    return {'states': len(rows), 'supported_states': len(supported),
            'routes': len({r['episode_id'] for r in rows}),
            'changed_states': len(changed),
            'protected_states': sum(r['selection_reason'] == 'protected_stop_or_budget' for r in rows),
            'native_abstentions': sum(r['selection_reason'] == 'native_abstention' for r in rows),
            'native_successful_states': sum(r['native_metrics']['success'] > 0 for r in supported),
            'chosen_successful_states': sum(r['chosen_metrics']['success'] > 0 for r in supported),
            'changed_relations': dict(collections.Counter(r['chosen_vs_native'] for r in changed)),
            'native_dominated_by_move_states': sum(r['native_dominated_by_move'] for r in supported),
            'chosen_dominated_by_move_states': sum(r['chosen_dominated_by_move'] for r in supported),
            'chosen_has_any_component_harm': sum(any(r['delta'][k] < -TOL for k in ['success', 'spl', 'ndtw']) or any(r['delta'][k] > TOL for k in ['distance_to_goal', 'path_length', 'primitive_action_count']) for r in changed),
            'component_harm_counts': {k: sum((r['delta'][k] < -TOL if k in ['success', 'spl', 'ndtw'] else r['delta'][k] > TOL) for r in changed) for k in METRICS},
            'note': 'Counts over correlated frozen states, not executed multi-step policy episode metrics.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--register', action='store_true')
    args = parser.parse_args(); cfg = json.loads(args.config.read_text())
    if cfg['fit_model'] or cfg['use_replication_outcomes'] or cfg['allow_simulator_rollout'] or cfg['metric_tolerance'] != TOL:
        raise ValueError('Fixed diagnostic information boundary changed')
    plan_path = Path(cfg['training_plan']); plan = json.loads(plan_path.read_text())
    graph_path = Path(plan['config']['source_root']) / 'capture_001/graph_options/graph_options.jsonl'
    paths = [args.config, Path(cfg['protocol']), plan_path, Path(cfg['training_returns']),
             Path(cfg['frozen_model']), graph_path, Path(__file__),
             Path('research/tools/graph_value_feasibility.py'), Path('research/tools/prepare_ranker_intervention_schedule.py'),
             Path('research/results/critical_graph_value_fresh/independent_audit_001/summary.json')]
    regpath = args.output / 'registration.json'
    if args.register:
        args.output.mkdir(exist_ok=False)
        registration = {'created_utc': datetime.now(timezone.utc).isoformat(),
                        'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip(),
                        'python': sys.version, 'config': cfg,
                        'scope': 'In-sample development postmortem. Navigation outcomes already exist; not prospective confirmation.',
                        'original_simulator_config': plan['config'],
                        'hashes': {str(p): digest(p) for p in paths}}
        with regpath.open('x') as stream:
            json.dump(registration, stream, indent=2); stream.write('\n')
        print('Registered training-support audit:', str(regpath)); return
    registration = json.loads(regpath.read_text())
    for path, expected in registration['hashes'].items():
        if digest(path) != expected:
            raise ValueError('Registered source/input changed: ' + path)
    independent = json.loads(paths[-1].read_text())
    if not independent['all_checks_passed'] or independent['result_sha256'] != digest(Path(cfg['training_returns'])):
        raise ValueError('Source full-return audit does not match')
    model = json.loads(Path(cfg['frozen_model']).read_text())
    if model['feature_names'] != list(FEATURE_NAMES):
        raise ValueError('Frozen feature schema mismatch')
    if model['hashes'][cfg['training_returns']] != digest(Path(cfg['training_returns'])):
        raise ValueError('Different fitting corpus')
    graphs = {}
    for line in graph_path.open():
        row = json.loads(line); key = (str(row['episode_id']), row['high_level_step'])
        if key in graphs: raise ValueError('Duplicate graph state')
        graphs[key] = row
    expected = {c['case_id']: c for c in plan['cases']}
    if len(expected) != len(plan['cases']): raise ValueError('Duplicate planned case')
    groups = collections.defaultdict(dict); seen = set(); feature_rows = []
    w = np.asarray(model['weights_standardized']); mean = np.asarray(model['mean']); scale = np.asarray(model['scale'])
    if w.shape != (len(FEATURE_NAMES),) or mean.shape != w.shape or scale.shape != w.shape or np.any(scale <= 0) or not all(np.all(np.isfinite(x)) for x in [w, mean, scale]):
        raise ValueError('Invalid frozen model dimensions or values')
    native_return_checks = 0
    for line in Path(cfg['training_returns']).open():
        result = json.loads(line); case = result['case']
        if case['mode'] != 'action': continue
        case_id = case['case_id']
        if case_id in seen or case_id not in expected or not result['validity_passed']:
            raise ValueError('Duplicate, unplanned or invalid action return')
        planned = expected[case_id]
        if any(case[k] != planned[k] for k in ['episode_id', 'high_level_step', 'action_index', 'action']):
            raise ValueError('Full-return case differs from plan')
        seen.add(case_id); key = (str(case['episode_id']), case['high_level_step']); graph = graphs[key]
        option = next(o for o in graph['options'] if o['index'] == case['action_index'])
        if not option['admissible'] or option['action'] != case['action']:
            raise ValueError('Graph action identity mismatch')
        metrics = dict(result['metrics']); metrics['primitive_action_count'] = int(round(metrics.get('primitive_action_count', metrics['steps_taken'])))
        if not all(np.isfinite(metrics[k]) for k in METRICS): raise ValueError('Nonfinite return')
        if option['index'] == graph['effective_index']:
            baseline = dict(result['baseline_metrics'])
            baseline['primitive_action_count'] = int(round(baseline.get('primitive_action_count', baseline['steps_taken'])))
            if any(abs(metrics[k] - baseline[k]) > TOL for k in METRICS):
                raise ValueError('Native full return does not reproduce baseline')
            native_return_checks += 1
        features = action_features(graph, option); feature_rows.append(features)
        if option['index'] in groups[key]: raise ValueError('Duplicate state/action')
        groups[key][option['index']] = {'index': option['index'], 'stop': option['action']['act'] == 0,
            'metrics': {k: metrics[k] for k in METRICS}, 'score': float(w.dot((features-mean)/scale)),
            'native_logit': float(option['logit']), 'case_id': case_id}
    planned_states = {(str(s['episode_id']), s['high_level_step']) for s in plan['states']}
    if seen != set(expected) or set(groups) != planned_states or len(groups) != cfg['expected_states'] or len(seen) != cfg['expected_actions']:
        raise ValueError('Incomplete frozen census')
    observed_mean = np.mean(feature_rows, axis=0); observed_scale = np.std(feature_rows, axis=0)
    observed_scale[observed_scale < 1e-8] = 1.
    if not np.allclose(mean, observed_mean, atol=1e-10, rtol=0) or not np.allclose(scale, observed_scale, atol=1e-10, rtol=0):
        raise ValueError('Frozen normalization does not match development features')
    pairs = []; states = []
    for key, actions in groups.items():
        graph = graphs[key]; native = graph['effective_index']; scene = graph['scene_id']
        admissible_indices = {o['index'] for o in graph['options'] if o['admissible']}
        if set(actions) != admissible_indices: raise ValueError('Incomplete admissible action coverage')
        for left, right in itertools.combinations(actions.values(), 2):
            rel = relation(left['metrics'], right['metrics']); strict = rel in ['strict_improvement','strict_degradation']
            winner, loser = (left, right) if rel == 'strict_improvement' else (right, left)
            pairs.append({'episode_id': key[0], 'step': key[1], 'scene_id': scene,
                'left_index': left['index'], 'right_index': right['index'],
                'relation': 'strict' if strict else rel,
                'kind': pair_kind(left['stop'], right['stop'], left['index'], right['index'], native),
                'involves_native': native in [left['index'], right['index']],
                'winner_index': winner['index'] if strict else None,
                'frozen_correct': winner['score'] > loser['score'] if strict else None,
                'logit_correct': winner['native_logit'] > loser['native_logit'] if strict else None,
                'frozen_score_tie': left['score'] == right['score']})
        chosen, reason = chosen_index(graph, model)
        record = {'episode_id': key[0], 'step': key[1], 'scene_id': scene, 'native_index': native,
                  'chosen_index': chosen, 'selection_reason': reason, 'supported': native in actions and chosen in actions}
        if record['supported']:
            a, b = actions[native], actions[chosen]; moves = {o['index'] for o in admissible_moves(graph)}
            record.update(native_metrics=a['metrics'], chosen_metrics=b['metrics'],
                chosen_vs_native=relation(b['metrics'], a['metrics']),
                delta={k: b['metrics'][k]-a['metrics'][k] for k in METRICS},
                native_dominated_by_move=any(dominates(actions[i]['metrics'], a['metrics']) for i in moves),
                chosen_dominated_by_move=any(dominates(actions[i]['metrics'], b['metrics']) for i in moves))
        states.append(record)
    strict_count = sum(p['relation'] == 'strict' for p in pairs)
    if strict_count != cfg['expected_strict_pairs']:
        raise ValueError('Strict pair count differs from frozen training support')
    pair_groups = {}
    for name in ['all_pairs','stop_involving','native_move','two_non_native_moves','all_native_involving']:
        subset = [p for p in pairs if name == 'all_pairs' or (name == 'all_native_involving' and p['involves_native']) or p['kind'] == name]
        strict = [p for p in subset if p['relation'] == 'strict']
        pair_groups[name] = {'pairs': len(subset), 'relations': dict(collections.Counter(p['relation'] for p in subset)),
            'strict_pairs': len(strict), 'frozen_correct': sum(p['frozen_correct'] for p in strict),
            'logit_correct': sum(p['logit_correct'] for p in strict),
            'frozen_accuracy': float(np.mean([p['frozen_correct'] for p in strict])) if strict else None,
            'logit_accuracy': float(np.mean([p['logit_correct'] for p in strict])) if strict else None}
    choices = state_summary(states)
    adverse_or_mixed = sum(choices['changed_relations'].get(k, 0) for k in ['mixed', 'strict_degradation'])
    result = {'experiment_id': cfg['experiment_id'], 'coverage_and_normalization_passed': True,
        'states': len(states), 'actions': len(seen), 'pairs': len(pairs), 'strict_pairs': strict_count,
        'native_return_checks': native_return_checks,
        'pair_groups': pair_groups, 'state_choices': choices,
        'by_route': {ep: state_summary([r for r in states if r['episode_id'] == ep]) for ep in sorted({r['episode_id'] for r in states})},
        'by_scene': {scene: state_summary([r for r in states if r['scene_id'] == scene]) for scene in sorted({r['scene_id'] for r in states})},
        'in_sample_only': True, 'model_refitted': False, 'replication_outcomes_read': False,
        'decision': 'NO-GO unchanged; this diagnostic cannot authorize training',
        'support_diagnosis': 'Retire all-pair accuracy as a sufficient gate: native-relative choices include mixed or harmful returns.' if adverse_or_mixed else 'In-sample choices are nondegrading; distribution/trigger transfer remains unresolved.',
        'limits': 'Frozen fitting corpus, failure-selected and correlated critical states. Not held-out accuracy or an executed episode-policy aggregate.',
        'registration_sha256': digest(regpath)}
    output = args.output / 'analysis_001'; output.mkdir(exist_ok=False)
    for name, data in [('summary.json', result), ('states.json', states)]:
        with (output/name).open('x') as stream:
            json.dump(data, stream, indent=2); stream.write('\n')
    with (output/'pairs.csv').open('x', newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(pairs[0])); writer.writeheader(); writer.writerows(pairs)
    print(json.dumps({'pair_groups': pair_groups, 'state_choices': result['state_choices']},indent=2))


if __name__ == '__main__': main()
