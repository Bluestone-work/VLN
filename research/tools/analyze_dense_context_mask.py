#!/usr/bin/env python3
"""Analyze the frozen four-arm dense-context execution-mask experiment."""
import csv
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'research/results/dense_a2_mask_control_val_unseen_001'
EVAL = ROOT / 'data/logs/eval_results'
ARMS = {
    'A0': 'gate_a_native_policy_replication_unseen_a0',
    'A1': 'gate_a_native_policy_replication_unseen_a1',
    'A2-ISO': 'dense_a2_control_val_unseen_001',
    'A2-MASK': 'dense_a2_mask_control_val_unseen_001',
}
METRICS = ['success', 'spl', 'ndtw', 'sdtw', 'path_length', 'steps_taken', 'collisions']


def load_arm(name):
    p = EVAL / ARMS[name] / 'stats_ep_ckpt_59_val_unseen_r0_w1.json'
    return {str(k): v for k, v in json.load(p.open()).items()}


def mean(d, metric):
    return statistics.mean(float(v[metric]) for v in d.values())


def bootstrap(groups, n=10000, seed=100):
    rng = random.Random(seed)
    scenes = list(groups)
    values = []
    for _ in range(n):
        sample = [rng.choice(scenes) for _ in scenes]
        values.append(statistics.mean(statistics.mean(groups[s]) for s in sample))
    values.sort()
    return [values[int(.025 * n)], values[int(.975 * n)]]


def main():
    arms = {name: load_arm(name) for name in ARMS}
    manifest = json.load((ROOT / 'research/results/gate_a_native_policy_replication_001/manifest.json').open())
    scene = {str(r['episode_id']): r['scene_id'] for r in manifest['route_rows'] if r['cohort'] == 'val_unseen'}
    rows = []
    for eid in sorted(arms['A0'], key=int):
        row = {'episode_id': eid, 'scene_id': scene[eid]}
        for name, data in arms.items():
            for metric in METRICS:
                row['%s_%s' % (name, metric)] = float(data[eid][metric])
        for name in ('A1', 'A2-ISO', 'A2-MASK'):
            row['%s_destroyed' % name] = int(row['A0_success'] >= 1 and row['%s_success' % name] < 1)
        rows.append(row)
    with (OUT / 'four_arm_per_route.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    summary = {'experiment_id': 'DENSE-A2-MASK-CONTROL-VAL-UNSEEN-001', 'route_count': len(rows),
               'scene_count': len(set(scene.values())), 'arms': {}, 'destroyed_route_ids': {},
               'bootstrap_scene_ci': {}, 'mask_fidelity': {}}
    for name, data in arms.items():
        summary['arms'][name] = {m: mean(data, m) for m in METRICS}
        summary['arms'][name]['success_routes'] = sum(float(v['success']) >= 1 for v in data.values())
    for name in ('A1', 'A2-ISO', 'A2-MASK'):
        summary['destroyed_route_ids'][name] = [r['episode_id'] for r in rows if r['%s_destroyed' % name]]
    for left, right in [('A2-MASK', 'A0'), ('A2-MASK', 'A1'), ('A2-ISO', 'A1')]:
        key = '%s_minus_%s' % (left, right)
        summary['bootstrap_scene_ci'][key] = {}
        for metric in ['success', 'spl', 'ndtw', 'steps_taken', 'collisions']:
            groups = defaultdict(list)
            for r in rows:
                groups[r['scene_id']].append(r['%s_%s' % (left, metric)] - r['%s_%s' % (right, metric)])
            summary['bootstrap_scene_ci'][key][metric] = bootstrap(groups)
    trace = OUT / 'run/graph_options/graph_options.jsonl'
    dense_selected = 0; full_rows = 0; bad = []
    with trace.open() as f:
        for line in f:
            r = json.loads(line); dc = r.get('dense_context'); full_rows += int(dc is not None)
            native = {i for i, flag in enumerate(dc.get('native_mask', [])) if flag}
            dense_selected += int(r['effective_index'] not in native)
            if r['effective_index'] not in native:
                bad.append([r['episode_id'], r['high_level_step'], r['effective_index']])
    summary['mask_fidelity'] = {'trace_rows': sum(1 for _ in trace.open()), 'full_context_rows': full_rows,
                                'dense_selected_count': dense_selected, 'bad_selected_rows': bad,
                                'all_dense_execution_blocked': dense_selected == 0 and not bad}
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
