#!/usr/bin/env python3
"""Build outcome-labeled native-relative Gate B samples.

The labels use full-return outcomes, while all feature columns are read from
the pre-action graph capture.  This is a diagnostic dataset builder; it does
not fit a model or alter navigation.
"""
import argparse
import csv
import itertools
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from vlnce_baselines.adaptive_action.option_calibration import unpack

EPS = 1e-6


def read_jsonl(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream]


def strict(base, row):
    m = row['metrics']
    return (m['success'] + EPS >= base['success'] and m['spl'] + EPS >= base['spl']
            and m['ndtw'] + EPS >= base['ndtw']
            and m['distance_to_goal'] <= base['distance_to_goal'] + EPS
            and m['path_length'] <= base['path_length'] + EPS
            and m['steps_taken'] <= base['steps_taken'] + EPS
            and any([m['success'] > base['success'] + EPS, m['spl'] > base['spl'] + EPS,
                     m['ndtw'] > base['ndtw'] + EPS,
                     m['distance_to_goal'] < base['distance_to_goal'] - EPS,
                     m['path_length'] < base['path_length'] - EPS,
                     m['steps_taken'] < base['steps_taken'] - EPS]))


def action_path_length(action, current):
    action = unpack(action)
    points = [np.asarray(current, dtype=np.float64)]
    for item in action.get('back_path') or []:
        points.append(np.asarray(item[1], dtype=np.float64))
    target_key = 'stop_pos' if action['act'] == 0 else 'ghost_pos'
    if target_key in action:
        points.append(np.asarray(action[target_key], dtype=np.float64))
    return float(sum(np.linalg.norm(b - a) for a, b in zip(points, points[1:])))


def softmax_stats(logits, valid):
    values = [float(logits[i]) for i in valid if logits[i] is not None and math.isfinite(float(logits[i]))]
    if not values:
        return 0.0, 0.0
    values = np.asarray(values, dtype=np.float64)
    shifted = values - values.max(); probs = np.exp(shifted) / np.exp(shifted).sum()
    entropy = float(-(probs * np.log(np.maximum(probs, 1e-12))).sum())
    ordered = np.sort(values)[::-1]
    margin = float(ordered[0] - ordered[1]) if len(ordered) > 1 else float(ordered[0])
    return entropy, margin


def build_cohort(name, graph_path, results_path):
    graphs = {(str(r['episode_id']), int(r['high_level_step'])): r for r in read_jsonl(graph_path)}
    results = read_jsonl(results_path)
    controls = {str(r['case']['episode_id']): r['metrics'] for r in results if r['case']['mode'] == 'control'}
    branches = defaultdict(dict)
    for r in results:
        case = r['case']
        if case['mode'] == 'action':
            branches[(str(case['episode_id']), int(case['high_level_step']))][int(case['action_index'])] = r
    samples = []
    state_seen = set()
    for key, graph in graphs.items():
        ep, step = key
        if key not in branches or ep not in controls:
            continue
        native_index = int(graph['effective_index'])
        native_result = branches[key].get(native_index)
        if native_result is None:
            continue
        base = controls[ep]
        valid = [o['index'] for o in graph['options'] if o['admissible']]
        entropy, margin = softmax_stats(graph['logits'], valid)
        native_option = next(o for o in graph['options'] if o['index'] == native_index)
        native_logit = float(native_option['logit'])
        current = unpack(graph['graph_position'])
        current_pos = np.asarray(current, dtype=np.float64)
        state_seen.add(key)
        for option in graph['options']:
            idx = int(option['index'])
            if idx == native_index or idx not in branches[key]:
                continue
            result = branches[key][idx]
            alt = option['action']
            alt_metrics = result['metrics']
            # Gate B is native-relative: the alternative must dominate the
            # native action from this exact captured state.  The episode
            # control is retained only for stratification; it is not the
            # intervention label reference.
            if strict(native_result['metrics'], result):
                label = 'INTERVENE'
            elif strict(result['metrics'], native_result):
                label = 'KEEP'
            else:
                label = 'AMBIGUOUS'
            alt_unpacked = unpack(alt)
            native_unpacked = unpack(native_option['action'])
            alt_target = np.asarray(alt_unpacked.get('stop_pos' if alt_unpacked['act'] == 0 else 'ghost_pos'), dtype=np.float64)
            native_target = np.asarray(native_unpacked.get('stop_pos' if native_unpacked['act'] == 0 else 'ghost_pos'), dtype=np.float64)
            valid_logits = [float(graph['logits'][i]) for i in valid]
            rank = 1 + sum(v > float(option['logit']) for v in valid_logits)
            native_rank = 1 + sum(v > native_logit for v in valid_logits)
            samples.append({
                'cohort': name, 'split': name, 'episode_id': ep, 'scene_id': graph['scene_id'],
                'high_level_step': step, 'native_index': native_index, 'candidate_index': idx,
                'label': label, 'native_success': int(base['success'] >= 1.0 - EPS),
                'native_failure': int(base['success'] < 1.0 - EPS),
                'candidate_is_stop': int(alt_unpacked['act'] == 0),
                'native_is_stop': int(native_unpacked['act'] == 0),
                'candidate_current_proposal': int(bool(option.get('current_proposal', False))),
                'native_current_proposal': int(bool(native_option.get('current_proposal', False))),
                'candidate_logit': float(option['logit']), 'native_logit': native_logit,
                'logit_difference': float(option['logit']) - native_logit,
                'candidate_rank_norm': float(rank / max(len(valid), 1)),
                'native_rank_norm': float(native_rank / max(len(valid), 1)),
                'candidate_distance_m': float(np.linalg.norm(alt_target - current_pos)),
                'native_distance_m': float(np.linalg.norm(native_target - current_pos)),
                'candidate_path_estimate_m': action_path_length(alt, current_pos),
                'native_path_estimate_m': action_path_length(native_option['action'], current_pos),
                'candidate_count': len(valid), 'high_level_step_norm': float(step / 15.0),
                'policy_entropy': entropy, 'top1_top2_margin': margin,
                'visited_or_history': int(not bool(option.get('current_proposal', False))),
                'native_full_success': int(native_result['metrics']['success'] >= 1.0 - EPS),
                'alternative_full_success': int(alt_metrics['success'] >= 1.0 - EPS),
            })
    return samples, state_seen


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train-graph', type=Path, required=True); p.add_argument('--train-results', type=Path, required=True)
    p.add_argument('--unseen-graph', type=Path, required=True); p.add_argument('--unseen-results', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=False)
    train, train_states = build_cohort('train', args.train_graph, args.train_results)
    unseen, unseen_states = build_cohort('val_unseen', args.unseen_graph, args.unseen_results)
    all_samples = train + unseen
    fields = list(all_samples[0].keys())
    with (args.output_dir / 'samples.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(all_samples)
    route_rows = []
    for (cohort, episode_id, scene_id), group in sorted(
            ((key, list(rows)) for key, rows in itertools.groupby(
                sorted(all_samples, key=lambda r: (r['cohort'], r['episode_id'])),
                key=lambda r: (r['cohort'], r['episode_id'], r['scene_id']))),
            key=lambda item: item[0]):
        labels = Counter(r['label'] for r in group)
        route_rows.append({
            'cohort': cohort, 'episode_id': episode_id, 'scene_id': scene_id,
            'state_count': len({(r['episode_id'], r['high_level_step']) for r in group}),
            'sample_count': len(group),
            'intervene_count': labels['INTERVENE'], 'keep_count': labels['KEEP'],
            'ambiguous_count': labels['AMBIGUOUS'],
            'native_success_sample_count': sum(r['native_success'] for r in group),
        })
    route_fields = list(route_rows[0].keys()) if route_rows else [
        'cohort', 'episode_id', 'scene_id', 'state_count', 'sample_count',
        'intervene_count', 'keep_count', 'ambiguous_count',
        'native_success_sample_count']
    with (args.output_dir / 'per_route.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=route_fields)
        writer.writeheader(); writer.writerows(route_rows)
    by_label = Counter(r['label'] for r in all_samples)
    summary = {'protocol': 'Gate B native-relative long-horizon intervention feasibility',
               'privileged_labels_analysis_only': True, 'sample_count': len(all_samples),
               'state_count': len(train_states | unseen_states),
               'route_count': len({r['episode_id'] for r in all_samples}),
               'scene_count': len({r['scene_id'] for r in all_samples}),
               'class_counts': dict(by_label),
               'cohort_counts': {c: dict(Counter(r['label'] for r in all_samples if r['cohort'] == c)) for c in ['train','val_unseen']},
               'state_counts_by_class': {c: len({(r['episode_id'], r['high_level_step']) for r in all_samples if r['label'] == c}) for c in ['INTERVENE','KEEP','AMBIGUOUS']},
               'route_counts_by_class': {c: len({r['episode_id'] for r in all_samples if r['label'] == c}) for c in ['INTERVENE','KEEP','AMBIGUOUS']},
               'scene_counts_by_class': {c: len({r['scene_id'] for r in all_samples if r['label'] == c}) for c in ['INTERVENE','KEEP','AMBIGUOUS']},
               'native_success_samples': sum(r['native_success'] for r in all_samples),
               'native_failure_samples': sum(r['native_failure'] for r in all_samples),
               'positive_scene_support_sufficient': len({r['scene_id'] for r in all_samples if r['label'] == 'INTERVENE'}) >= 3,
               'split': 'train cohort versus held-out val_unseen scenes; no random row split',
               'feature_policy': 'pre-action graph/logit/geometry only; no goal/reference/future outcome fields',
               'next_gate': 'fit only simple selective models if INTERVENE support is not sparse or scene-concentrated'}
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__': main()
