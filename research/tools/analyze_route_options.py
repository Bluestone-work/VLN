#!/usr/bin/env python3
"""Audit full graph actions against ordered reference prefixes, analysis only."""
import argparse
import collections
import gzip
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from fastdtw import fastdtw
from audit_ranker_validity import cluster_ci
from route_alignment import trace_path, initial_row, extend_row, open_endpoint, route_gate, best_option


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def key(row):
    return row['scene_id'], row['episode_id'], row['high_level_step']


def load_unique(path):
    rows = collections.OrderedDict()
    with Path(path).open() as stream:
        for line in stream:
            row = json.loads(line)
            k = key(row)
            if k in rows:
                raise ValueError('Duplicate decision')
            rows[k] = row
    return rows


def prepare_prefixes(traces, references, native, tolerance):
    """Freeze every true baseline prefix; never update it using alternatives."""
    grouped = collections.OrderedDict()
    for k, trace in traces.items():
        grouped.setdefault(k[1], []).append((k, trace))
    if set(grouped) != set(native):
        raise ValueError('Native and trace episodes differ')
    prefixes, checks = {}, []
    for episode, records in grouped.items():
        if [k[2] for k, _ in records] != list(range(len(records))) or not records[-1][1]['done']:
            raise ValueError('Incomplete native episode')
        reference = np.asarray(references[episode]['locations'], dtype=np.float64)
        if reference.ndim != 2 or reference.shape[1] != 3 or not np.all(np.isfinite(reference)):
            raise ValueError('Invalid reference')
        first = records[0][1]['pre_pose']['position']
        row, cursor, path = initial_row(first, reference), 0, [np.asarray(first)]
        for k, trace in records:
            option_path = trace_path(trace)
            if not np.array_equal(option_path[0], path[-1]):
                raise ValueError('Broken baseline prefix continuity')
            selected_row = extend_row(row, option_path[1:], reference)
            prefixes[k] = {'row': row, 'cursor': cursor, 'selected_row': selected_row,
                           'reference': reference, 'start': option_path[0], 'selected_path': option_path}
            cursor = open_endpoint(selected_row, cursor)
            row = selected_row
            path.extend(option_path[1:])
        # Position stores Habitat's float32 positions. Match native arithmetic
        # exactly here; the separate diagnostic DP intentionally uses float64.
        path = np.asarray(path, dtype=np.float32)
        length = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
        dtw_distance = fastdtw(path, reference, dist=lambda a, b: np.linalg.norm(b - a))[0]
        ndtw = float(np.exp(-dtw_distance / (len(reference) * 3.)))
        metrics = {'path_length': length, 'ndtw': ndtw, 'sdtw': ndtw * native[episode]['success']}
        deltas = {name: abs(value - native[episode][name]) for name, value in metrics.items()}
        checks.append({'episode_id': episode, 'distinct_positions': len(path), 'reference_positions': len(reference),
                       'recomputed': metrics, 'absolute_deltas': deltas, 'passed': max(deltas.values()) <= tolerance})
    return prefixes, checks


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--probe-dir', type=Path, required=True)
    p.add_argument('--integrity-dir', type=Path, required=True)
    p.add_argument('--noninterference-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    cfg = json.loads(args.config.read_text())
    manifest = json.loads((args.probe_dir / 'manifest.json').read_text())
    summary = json.loads((args.probe_dir / 'summary.json').read_text())
    integrity = json.loads((args.integrity_dir / 'summary.json').read_text())
    control = json.loads((args.noninterference_dir / 'summary.json').read_text())
    raw_path = args.probe_dir / 'branch_traces.jsonl'
    raw_hash = digest(raw_path)
    if not (summary['all_required_gates_passed'] and summary['all_admissible_options_complete']
            and integrity['integrity_gate_passed'] and control['baseline_noninterference_gate_passed']):
        raise ValueError('Prerequisite fidelity gates must pass')
    if integrity['raw_trace_sha256'] != raw_hash or control['config'] != manifest['config']:
        raise ValueError('Integrity/control belongs to different inputs')
    capture = Path(manifest['capture_dir'])
    trace_pathname = next((capture / 'traces').glob('worker_seed*.jsonl'))
    traces = load_unique(trace_pathname)
    decisions = load_unique(args.probe_dir / 'decisions.jsonl')
    if set(traces) != set(decisions) or len(decisions) != summary['decisions']:
        raise ValueError('Capture/probe coverage differs')
    split = manifest['config']['split']
    reference_path = Path('data/datasets/R2R_VLNCE_v1-2_preprocessed') / split / (split + '_gt.json.gz')
    references = json.load(gzip.open(str(reference_path), 'rt'))
    native_path = args.noninterference_dir / 'capture_episodes.json'
    native = json.loads(native_path.read_text())
    files = [args.config, Path(cfg['protocol']), Path(__file__), Path(__file__).with_name('route_alignment.py'),
             Path(__file__).with_name('audit_ranker_validity.py'), reference_path, native_path, trace_pathname,
             args.probe_dir / 'manifest.json', args.probe_dir / 'summary.json', args.probe_dir / 'decisions.jsonl',
             args.integrity_dir / 'summary.json', args.noninterference_dir / 'summary.json']
    provenance = {'created_utc': datetime.now(timezone.utc).isoformat(), 'argv': sys.argv, 'config': cfg,
                  'probe_experiment': manifest['experiment_id'], 'python': sys.version,
                  'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip(),
                  'hardware': subprocess.check_output(['nvidia-smi', '--query-gpu=name,uuid,driver_version', '--format=csv,noheader']).decode(),
                  'hashes': {str(path): digest(path) for path in files}, 'raw_trace_sha256': raw_hash,
                  'privileged_analysis_only': True}
    (args.output_dir / 'manifest.json').write_text(json.dumps(provenance, indent=2) + '\n')
    prefixes, checks = prepare_prefixes(traces, references, native, cfg['baseline_metric_tolerance'])
    (args.output_dir / 'baseline_reconstruction.json').write_text(json.dumps(checks, indent=2) + '\n')
    if not all(c['passed'] for c in checks):
        raise ValueError('Reconstructed baseline path/fastdtw metrics disagree')
    del traces, references
    print('Baseline reconstruction passed: {} episodes'.format(len(checks)), flush=True)
    tolerance = cfg['cost_and_dtw_tolerance']
    expected, outcomes, selected_outcomes = {}, collections.defaultdict(list), {}
    for k, decision in decisions.items():
        selected = next(o for o in decision['options'] if o['index'] == decision['effective_index'])
        if selected['action_type'] == 0:
            continue
        if decision['forced_stop'] or decision['budget_stop']:
            raise ValueError('Inadmissible non-STOP state')
        selected_outcomes[k] = selected
        for option in decision['options']:
            if option['action_type'] == 4:
                expected[(k, option['index'])] = option
    seen = set()
    with raw_path.open() as raw, (args.output_dir / 'options.jsonl').open('x') as output:
        for line in raw:
            branch = json.loads(line)
            k = tuple(branch['decision_key'])
            identity = (k, branch['index'])
            if branch['order'] != 'forward' or identity not in expected:
                continue
            if identity in seen:
                raise ValueError('Duplicate eligible branch')
            seen.add(identity)
            prefix, option, selected = prefixes[k], expected[identity], selected_outcomes[k]
            points = trace_path(branch['trace'])
            if not np.array_equal(points[0], prefix['start']):
                raise ValueError('Alternative shares wrong prefix')
            row = extend_row(prefix['row'], points[1:], prefix['reference'])
            if option['index'] == selected['index']:
                if not (np.array_equal(points, prefix['selected_path']) and np.array_equal(row, prefix['selected_row'])):
                    raise ValueError('Selected route sentinel differs')
            gates = route_gate(row, prefix['selected_row'], prefix['cursor'], tolerance)
            arc = np.concatenate(([0.], np.cumsum(np.linalg.norm(np.diff(prefix['reference'], axis=0), axis=1))))
            result = dict(option)
            result.update(gates)
            result.update(scene_id=k[0], episode_id=k[1], high_level_step=k[2],
                          selected_index=selected['index'], cursor=prefix['cursor'],
                          reference_arc_advance_m=float(arc[gates['reference_endpoint']] - arc[prefix['cursor']]),
                          primitive_cap=option['primitive_events'] <= selected['primitive_events'],
                          path_cap=option['motion_path_m'] <= selected['motion_path_m'] + tolerance)
            result['joint_cost_cap'] = bool(result['primitive_cap'] and result['path_cap'])
            outcomes[k].append(result)
            output.write(json.dumps(result, sort_keys=True, allow_nan=False) + '\n')
    if seen != set(expected):
        raise ValueError('Missing eligible alternatives')
    states = []
    for k, options in outcomes.items():
        selected = selected_outcomes[k]
        selected_diagnostic = next(o for o in options if o['index'] == selected['index'])
        if not selected_diagnostic['matched_prefix_route_gate']:
            raise ValueError('Selected action must pass route gate')
        capped = [o for o in options if o['joint_cost_cap']]
        goal_best = best_option(capped, selected['index'])
        route_best = best_option([o for o in capped if o['matched_prefix_route_gate']], selected['index'])
        endpoint_best = best_option([o for o in capped if o['endpoint_gate']], selected['index'])
        state = {'scene_id': k[0], 'episode_id': k[1], 'high_level_step': k[2],
                 'trajectory_id': decisions[k]['trajectory_id'], 'baseline_episode_success': native[k[1]]['success'],
                 'selected_index': selected['index'], 'goal_best_index': goal_best['index'],
                 'route_best_index': route_best['index'], 'selected_progress_m': selected['progress_m'],
                 'joint_cost_gain_m': goal_best['progress_m'] - selected['progress_m'],
                 'route_matched_gain_m': route_best['progress_m'] - selected['progress_m'],
                 'endpoint_only_gain_m': endpoint_best['progress_m'] - selected['progress_m'],
                 'goal_best_endpoint_gate': goal_best['endpoint_gate'],
                 'goal_best_matched_cost_gate': goal_best['matched_cost_gate'],
                 'goal_best_route_gate': goal_best['matched_prefix_route_gate'],
                 'goal_best_route_cost_delta': goal_best['matched_endpoint_cost_delta'],
                 'selected_reference_endpoint': goal_best['selected_reference_endpoint'],
                 'goal_best_reference_endpoint': goal_best['reference_endpoint'],
                 'route_best_reference_endpoint': route_best['reference_endpoint'],
                 'selected_primitives': selected['primitive_events'], 'route_best_primitives': route_best['primitive_events'],
                 'selected_path_m': selected['motion_path_m'], 'route_best_path_m': route_best['motion_path_m']}
        states.append(state)
    if not states:
        raise ValueError('No eligible states')
    metrics = {}
    for name in ['joint_cost_gain_m', 'route_matched_gain_m', 'endpoint_only_gain_m']:
        values = [s[name] for s in states]
        material = [s for s in states if s[name] >= cfg['material_gain_m']]
        metrics[name] = {'mean_m': float(np.mean(values)), 'median_m': float(np.median(values)),
                         'material_states': len(material), 'material_episodes': len({s['episode_id'] for s in material}),
                         'material_scenes': len({s['scene_id'] for s in material}),
                         'positive_states': sum(v > tolerance for v in values),
                         'episode_bootstrap': cluster_ci(values, [s['episode_id'] for s in states], cfg['bootstrap_seed'], cfg['bootstrap_samples']),
                         'scene_bootstrap': cluster_ci(values, [s['scene_id'] for s in states], cfg['bootstrap_seed'], cfg['bootstrap_samples'])}
    by_scene = {}
    for scene in sorted({s['scene_id'] for s in states}):
        rows = [s for s in states if s['scene_id'] == scene]
        by_scene[scene] = {'states': len(rows), 'joint_cost_gain_m': float(np.mean([s['joint_cost_gain_m'] for s in rows])),
                           'route_matched_gain_m': float(np.mean([s['route_matched_gain_m'] for s in rows])),
                           'route_material_states': sum(s['route_matched_gain_m'] >= cfg['material_gain_m'] for s in rows)}
    material = [s for s in states if s['joint_cost_gain_m'] >= cfg['material_gain_m']]
    result = {'experiment_id': cfg['experiment_id'], 'source_experiment': manifest['experiment_id'],
              'episodes': len(checks), 'scenes': len(by_scene), 'eligible_states': len(states),
              'eligible_nonstop_branches': len(seen), 'baseline_reconstruction_gate_passed': True,
              'native_metric_comparisons': 3 * len(checks),
              'max_native_metric_delta': max(max(c['absolute_deltas'].values()) for c in checks),
              'selected_route_sentinels_passed': len(states), 'metrics': metrics, 'by_scene': by_scene,
              'material_goal_best_gate_counts': {'states': len(material),
                  'endpoint_pass': sum(s['goal_best_endpoint_gate'] for s in material),
                  'cost_pass': sum(s['goal_best_matched_cost_gate'] for s in material),
                  'both_pass': sum(s['goal_best_route_gate'] for s in material)},
              'examples_route_retained': sorted(states, key=lambda s: -s['route_matched_gain_m'])[:5],
              'examples_route_rejected': sorted([s for s in material if not s['goal_best_route_gate']], key=lambda s: -s['joint_cost_gain_m'])[:5],
              'privileged_analysis_only': True, 'learned_model_evaluated': False,
              'limits': 'Conservative geometric ordered-prefix proxy; no semantic certificate, causal decomposition or counterfactual full-episode score. Monotonic commitment and fixed selected endpoint can penalize recovery/further progress. Bootstrap is descriptive.'}
    (args.output_dir / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    with (args.output_dir / 'states.jsonl').open('x') as stream:
        for row in states:
            stream.write(json.dumps(row, sort_keys=True) + '\n')
    print(json.dumps({k: v for k, v in result.items() if not k.startswith('examples') and k != 'by_scene'}, indent=2))


if __name__ == '__main__':
    main()
