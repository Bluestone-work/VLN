#!/usr/bin/env python3
"""Paired continuation audit: prefix, execution, unaffected routes, final metrics."""
import argparse
import collections
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from run_option_capture import digest
from check_option_noninterference import compare_metrics
from replay_option_calibration import compare


def key(row):
    return row['scene_id'], str(row['episode_id']), row['high_level_step']


def read_rows(path):
    result = collections.OrderedDict()
    with Path(path).open() as stream:
        for line in stream:
            row = json.loads(line)
            k = key(row)
            if k in result:
                raise ValueError('Duplicate decision record')
            result[k] = row
    return result


def one(directory, pattern):
    paths = list(Path(directory).glob(pattern))
    if len(paths) != 1:
        raise ValueError('Expected exactly one {} in {}'.format(pattern, directory))
    return paths[0]


def group_rows(rows):
    episodes = collections.OrderedDict()
    for k, row in rows.items():
        episodes.setdefault(k[1], []).append(row)
    for episode, records in episodes.items():
        if ([r['high_level_step'] for r in records] != list(range(len(records)))
                or not records[-1]['done'] or len(records) > 15
                or any(r['done'] for r in records[:-1])):
            raise ValueError('Incomplete or budget-invalid episode: ' + episode)
    return episodes


def paired_report(baseline, current, selected, traces, old_traces, tolerance):
    extra = {}
    for episode in baseline:
        extra[episode] = {}
        for name, source in [('baseline', old_traces), ('intervention', traces)]:
            records = source[episode]
            extra[episode][name] = {
                'primitive_action_count': sum(len(r['primitives']) for r in records),
                'collision_events': sum(p['collided'] is True for r in records for p in r['primitives']),
                'native_collision_count': records[-1]['post_metrics']['collisions']['count']}
            native = baseline[episode] if name == 'baseline' else current[episode]
            if (extra[episode][name]['primitive_action_count'] != native['steps_taken']
                    or len(records) != native['high_level_steps']):
                raise ValueError('Native and traced action counts differ')
    metrics = sorted(set(baseline[next(iter(baseline))]) | set(extra[next(iter(extra))]['baseline']))
    lower_is_better = {'distance_to_goal', 'path_length', 'high_level_steps', 'steps_taken',
                       'primitive_action_count', 'collision_events', 'native_collision_count', 'collisions'}
    rows = []
    for episode in baseline:
        a, b = dict(baseline[episode]), dict(current[episode])
        a.update(extra[episode]['baseline'])
        b.update(extra[episode]['intervention'])
        rows.append({'episode_id': episode, 'scheduled': episode in selected,
                     'baseline': a, 'intervention': b, 'delta': {m: b[m] - a[m] for m in metrics}})
    groups = {}
    for label, ids in [('targeted', selected), ('unaffected', set(baseline) - selected), ('all', set(baseline))]:
        name = '{}_{}'.format(label, len(ids))
        subset = [r for r in rows if r['episode_id'] in ids]
        group = {'episodes': len(subset), 'metrics': {},
                 'success_rescued': [r['episode_id'] for r in subset if r['baseline']['success'] == 0 and r['intervention']['success'] == 1],
                 'success_lost': [r['episode_id'] for r in subset if r['baseline']['success'] == 1 and r['intervention']['success'] == 0]}
        for metric in metrics:
            if not subset:
                continue
            group['metrics'][metric] = {
                'baseline_mean': sum(r['baseline'][metric] for r in subset) / len(subset),
                'intervention_mean': sum(r['intervention'][metric] for r in subset) / len(subset),
                'mean_delta': sum(r['delta'][metric] for r in subset) / len(subset),
                'higher_count': sum(r['delta'][metric] > tolerance for r in subset),
                'lower_count': sum(r['delta'][metric] < -tolerance for r in subset),
                'tie_count': sum(abs(r['delta'][metric]) <= tolerance for r in subset),
                'lower_is_better': metric in lower_is_better}
        groups[name] = group
    return rows, groups


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--source-noninterference', type=Path, required=True)
    p.add_argument('--source-probe', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = args.run_dir / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    source = Path(manifest['source_capture'])
    source_manifest_path = source / 'manifest.json'
    source_manifest = json.loads(source_manifest_path.read_text())
    old_gate_path = args.source_noninterference / 'summary.json'
    old_gate = json.loads(old_gate_path.read_text())
    if not old_gate['baseline_noninterference_gate_passed'] or old_gate['config'] != manifest['config']:
        raise ValueError('Archived untraced baseline gate is invalid')
    if manifest['config'] != source_manifest['config']:
        raise ValueError('Changed baseline configuration')
    for name, value in source_manifest['overrides'].items():
        if name != 'TRAINER_NAME' and manifest['overrides'][name] != value:
            raise ValueError('Changed observation/evaluation overrides')
    for name in ['hardware', 'python', 'environment', 'git_commit']:
        if manifest[name] != source_manifest[name]:
            raise ValueError('Environment provenance changed: ' + name)
    for name, expected in manifest['hashes'].items():
        if digest(Path(name)) != expected:
            raise ValueError('Run source modified before audit: ' + name)
    cfg, schedule = manifest['config'], manifest['schedule']
    selected = {str(e['episode_id']) for e in schedule['events']}
    expected_events = schedule.get('expected_event_count', 7)
    if not selected or len(selected) != expected_events or len(schedule['events']) != expected_events:
        raise ValueError('Frozen intervention population changed')
    mode = 'enabled' if manifest['enabled'] else 'disabled'
    trace_path = one(args.run_dir / 'traces', 'worker_seed*.jsonl')
    baseline_path = one(source / 'traces', 'worker_seed*.jsonl')
    traces, originals = read_rows(trace_path), read_rows(baseline_path)
    current_grouped, old_grouped = group_rows(traces), group_rows(originals)
    if list(current_grouped) != list(old_grouped) or len(current_grouped) != cfg['episodes']:
        raise ValueError('Episode population/order changed')
    hook = read_rows(args.run_dir / 'hook/decisions.jsonl')
    hook_summary = json.loads((args.run_dir / 'hook/summary.json').read_text())
    if set(hook) != set(traces) or not hook_summary['hook_completed']:
        raise ValueError('Missing native hook or primitive traces')
    expected_interventions = {key(e) for e in schedule['events']} if manifest['enabled'] else set()
    actual_interventions = {k for k, r in hook.items() if r['intervened']}
    if actual_interventions != expected_interventions:
        raise ValueError('Wrong intervention events')
    for k, row in hook.items():
        if (not row['native_stop_preserved'] or not row['native_action_identity_verified']
                or row['expected_action'] != traces[k]['action']):
            raise ValueError('Native STOP/action mismatch')
        if not row['intervened'] and row['output_index'] != row['effective_native_index']:
            raise ValueError('Unscheduled action override')
    event_by_episode = {e['episode_id']: e for e in schedule['events']}
    exact_keys = set()
    for k in originals:
        if (not manifest['enabled'] or k[1] not in selected
                or k[2] < event_by_episode[k[1]]['high_level_step']):
            exact_keys.add(k)
    trace_differences = []
    for k in sorted(exact_keys):
        if k not in traces or originals[k] != traces[k]:
            differences = ['missing'] if k not in traces else [name for name in originals[k] if originals[k][name] != traces[k].get(name)]
            trace_differences.append({'decision_key': k, 'different_fields': differences})
    expected_metrics_path = args.source_noninterference / 'capture_episodes.json'
    if digest(expected_metrics_path) != old_gate['sources']['capture_episodes']['sha256']:
        raise ValueError('Archived native results changed')
    baseline_metrics = json.loads(expected_metrics_path.read_text())
    results = Path('data/logs/eval_results') / manifest['exp_name']
    current_metrics_path = one(results, 'stats_ep_*.json')
    aggregate_path = one(results, 'stats_ckpt_*.json')
    current_metrics = json.loads(current_metrics_path.read_text())
    if set(current_metrics) != set(baseline_metrics):
        raise ValueError('Native result episodes differ')
    unchanged = set(baseline_metrics) - selected if manifest['enabled'] else set(baseline_metrics)
    metrics_check = compare_metrics({e: baseline_metrics[e] for e in unchanged},
                                    {e: current_metrics[e] for e in unchanged},
                                    cfg['gates']['baseline_episode_metrics_tolerance'])
    if not unchanged:
        metrics_check = {'passed': True, 'episodes': 0, 'metric_comparisons': 0,
                         'max_absolute_delta': None, 'issues': [],
                         'note': 'No unaffected episodes; no control metric comparisons claimed.'}
    aggregate_check = None
    if not manifest['enabled']:
        aggregate_check = compare_metrics({'aggregate': old_gate['metrics']},
            {'aggregate': json.loads(aggregate_path.read_text())}, cfg['gates']['baseline_episode_metrics_tolerance'])
    event_checks = []
    if manifest['enabled']:
        probe_manifest = json.loads((args.source_probe / 'manifest.json').read_text())
        probe_summary = json.loads((args.source_probe / 'summary.json').read_text())
        if probe_manifest['capture_dir'] != str(source) or not probe_summary['all_required_gates_passed']:
            raise ValueError('Uncalibrated alternative source')
        expected_branches = {(key(e), e['alternative_index']): e for e in schedule['events']}
        found = set()
        with (args.source_probe / 'branch_traces.jsonl').open() as stream:
            for line in stream:
                branch = json.loads(line)
                identity = (tuple(branch['decision_key']), branch['index'])
                if branch['order'] != 'forward' or identity not in expected_branches:
                    continue
                if identity in found:
                    raise ValueError('Duplicate measured alternative')
                found.add(identity)
                k, _ = identity
                event, actual, previous = expected_branches[identity], traces[k], originals[k]
                start_exact = all(actual[name] == previous[name] for name in
                    ['pre_pose', 'pre_rng', 'pre_metrics', 'distance_before', 'episode_id', 'scene_id', 'high_level_step'])
                action_exact = actual['action'] == event['alternative_action'] == branch['trace']['action']
                result = compare(branch['trace'], actual, cfg['gates'])
                result.update(decision_key=k, start_matches_baseline_exact=start_exact,
                              alternative_action_exact=action_exact,
                              local_progress_m=actual['distance_before'] - actual['distance_after'],
                              local_progress_gain_m=previous['distance_after'] - actual['distance_after'])
                result['passed'] = bool(result['passed'] and result['sensor_hashes_exact'] and start_exact and action_exact
                    and result['max_primitive_position_error_m'] <= cfg['gates']['endpoint_tolerance_m'])
                event_checks.append(result)
        if found != set(expected_branches):
            raise ValueError('Missing measured alternatives')
    passed = (not trace_differences and metrics_check['passed']
              and (aggregate_check is None or aggregate_check['passed']) and all(e['passed'] for e in event_checks))
    rows, groups = paired_report(baseline_metrics, current_metrics, selected,
                                current_grouped, old_grouped, cfg['gates']['baseline_episode_metrics_tolerance'])
    result = {'experiment_id': schedule['experiment_id'], 'mode': mode,
              'all_required_gates_passed': bool(passed), 'source_capture': str(source),
              'schedule_sha256': digest(Path(manifest['schedule_path'])),
              'source_capture_manifest_sha256': digest(source_manifest_path),
              'episodes': len(current_grouped), 'decisions': len(traces),
              'unchanged_metric_comparison': metrics_check, 'aggregate_comparison': aggregate_check,
              'exact_trace_comparisons': len(exact_keys), 'trace_differences': trace_differences,
              'event_checks': event_checks, 'interventions': len(actual_interventions),
              'hook_summary': hook_summary, 'paired_groups': groups,
              'intervention_source_hashes': {name: manifest['hashes'][name] for name in
                  ['vlnce_baselines/adaptive_action/single_intervention.py', 'research/tools/run_single_intervention.py']},
              'privileged_analysis_only': schedule.get('analysis_only_oracle', True),
              'learned_model_evaluated': bool(manifest['enabled'] and schedule.get('learned_model_evaluated', False)),
              'limits': schedule.get('limits', '{} training routes selected for local gains, one simulator seed. Paired descriptive continuation outcomes; not an unbiased benchmark or deployable method.'.format(len(selected)))}
    for name, payload in [('summary.json', result), ('paired_episodes.json', rows), ('event_checks.json', event_checks)]:
        (args.output_dir / name).write_text(json.dumps(payload, indent=2) + '\n')
    provenance = {'created_utc': datetime.now(timezone.utc).isoformat(),
                  'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip(),
                  'hashes': {str(path): digest(path) for path in [Path(__file__), manifest_path, source_manifest_path,
                      old_gate_path, trace_path, baseline_path, expected_metrics_path, current_metrics_path, aggregate_path,
                      args.run_dir / 'hook/decisions.jsonl', args.run_dir / 'hook/summary.json']}}
    (args.output_dir / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ['paired_groups', 'event_checks']}, indent=2))
    if not passed:
        raise ValueError('Single-intervention fidelity gate failed; see saved summary')


if __name__ == '__main__':
    main()
