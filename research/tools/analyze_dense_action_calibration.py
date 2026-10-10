#!/usr/bin/env python3
"""Read-only calibration/OOD analysis for the frozen A0/A1/A2 control traces."""
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
A0 = ROOT / 'research/results/gate_a_native_policy_replication_001/runs/unseen_a0/graph_options/graph_options.jsonl'
A1 = ROOT / 'research/results/dense_a1_isolation_capture_control_val_unseen_001/run2/graph_options/graph_options.jsonl'
A2 = ROOT / 'research/results/dense_a2_control_val_unseen_001/run/graph_options/graph_options.jsonl'
EVAL = ROOT / 'data/logs/eval_results'
OUT = ROOT / 'research/results/dense_action_calibration_001'


def rows(path):
    return [json.loads(line) for line in path.open()]


def sig(action):
    if int(action['act']) == 0:
        p = action.get('stop_pos', {}).get('__array__', [])
    else:
        p = action.get('ghost_pos', {}).get('__array__', [])
    return (int(action['act']), tuple(round(float(x), 4) for x in p))


def stats(xs):
    xs = [float(x) for x in xs if math.isfinite(float(x))]
    if not xs:
        return {'n': 0}
    ys = sorted(xs)
    return {'n': len(xs), 'mean': statistics.mean(xs), 'median': statistics.median(xs),
            'p05': ys[max(0, int(.05 * len(ys)) - 1)],
            'p95': ys[max(0, int(.95 * len(ys)) - 1)],
            'min': ys[0], 'max': ys[-1]}


def safe_mean(xs):
    return statistics.mean(xs) if xs else None


def main():
    a0 = {(str(r['episode_id']), int(r['high_level_step'])): r for r in rows(A0)}
    a1 = {(str(r['episode_id']), int(r['high_level_step'])): r for r in rows(A1)}
    a2 = {(str(r['episode_id']), int(r['high_level_step'])): r for r in rows(A2)}
    ep0 = {str(k): v for k, v in json.load((EVAL / 'gate_a_native_policy_replication_unseen_a0/stats_ep_ckpt_59_val_unseen_r0_w1.json').open()).items()}
    ep1 = {str(k): v for k, v in json.load((EVAL / 'gate_a_native_policy_replication_unseen_a1/stats_ep_ckpt_59_val_unseen_r0_w1.json').open()).items()}
    ep2 = {str(k): v for k, v in json.load((EVAL / 'dense_a2_control_val_unseen_001/stats_ep_ckpt_59_val_unseen_r0_w1.json').open()).items()}
    # Only compare common prefixes. Once A0 and A1 choose different actions,
    # later rows describe different states and cannot support a calibration claim.
    common_keys = []
    for episode_id in sorted({k[0] for k in a0} & {k[0] for k in a1} & {k[0] for k in a2}):
        for step in sorted(k[1] for k in a0 if k[0] == episode_id):
            key = (episode_id, step)
            if key not in a1 or key not in a2:
                break
            common_keys.append(key)
            s0 = next((o for o in a0[key]['options'] if int(o['index']) == int(a0[key]['effective_index'])), None)
            s1 = next((o for o in a1[key]['options'] if int(o['index']) == int(a1[key]['effective_index'])), None)
            if s0 is None or s1 is None or sig(s0['action']) != sig(s1['action']):
                break
    records = []
    for key in common_keys:
        x, y, z = a0[key], a1[key], a2[key]
        native_sigs = {sig(o['action']) for o in x['options'] if o.get('admissible')}
        opts = [o for o in y['options'] if o.get('admissible')]
        iso = y.get('dense_isolation', {}).get('isolated_native_logits', [])
        native_mask = y.get('dense_isolation', {}).get('native_mask', [])
        full = {int(o['index']): float(o['logit']) for o in opts}
        selected = next((o for o in opts if int(o['index']) == int(y['effective_index'])), None)
        native_scores, dense_scores, native_shifts = [], [], []
        native_ranks, dense_ranks = [], []
        for o in opts:
            is_native = sig(o['action']) in native_sigs
            score = float(o['logit'])
            if is_native:
                native_scores.append(score)
            else:
                dense_scores.append(score)
            i = int(o['index'])
            if is_native and i < len(iso) and i < len(native_mask) and native_mask[i] and iso[i] is not None:
                native_shifts.append(abs(score - float(iso[i])))
        ranked = sorted(opts, key=lambda o: float(o['logit']), reverse=True)
        for rank, o in enumerate(ranked, 1):
            if sig(o['action']) in native_sigs:
                native_ranks.append(rank)
            else:
                dense_ranks.append(rank)
        dense_top = max(dense_scores) if dense_scores else None
        native_top = max(native_scores) if native_scores else None
        selected_dense = selected is not None and sig(selected['action']) not in native_sigs
        records.append({
            'episode_id': key[0], 'high_level_step': key[1],
            'a0_success': int(ep0[key[0]]['success'] >= 1),
            'a1_success': int(ep1[key[0]]['success'] >= 1),
            'a2_success': int(ep2[key[0]]['success'] >= 1),
            'a1_destroyed': int(ep0[key[0]]['success'] >= 1 and ep1[key[0]]['success'] < 1),
            'a2_destroyed': int(ep0[key[0]]['success'] >= 1 and ep2[key[0]]['success'] < 1),
            'native_count': len(native_scores), 'dense_count': len(dense_scores),
            'native_top': native_top, 'dense_top': dense_top,
            'dense_minus_native_top': (dense_top - native_top) if dense_top is not None and native_top is not None else None,
            'native_score_mean': safe_mean(native_scores), 'dense_score_mean': safe_mean(dense_scores),
            'native_shift_mean': safe_mean(native_shifts),
            'native_rank_mean': safe_mean(native_ranks), 'dense_rank_mean': safe_mean(dense_ranks),
            'selected_dense': int(selected_dense),
            'selected_score': float(selected['logit']) if selected else None,
            'selected_rank': (ranked.index(selected) + 1) if selected else None,
            'a1_effective_index': int(y['effective_index']),
            'a2_effective_index': int(z['effective_index']),
            'a1_a2_changed': int(y['effective_index'] != z['effective_index']),
            'a0_a1_changed': int(sig(next(o for o in x['options'] if int(o['index']) == int(x['effective_index']))['action']) != sig(next(o for o in y['options'] if int(o['index']) == int(y['effective_index']))['action'])),
        })
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'decision_records.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(records[0])); w.writeheader(); w.writerows(records)
    seen = set()
    for r in records:
        if r['episode_id'] not in seen and r['a0_a1_changed']:
            r['first_divergence'] = 1
            seen.add(r['episode_id'])
        else:
            r['first_divergence'] = 0
    def group(name, rs):
        gaps = [r['dense_minus_native_top'] for r in rs if r['dense_minus_native_top'] is not None]
        return {'name': name, 'rows': len(rs), 'states': len(rs),
                'selected_dense_rate': safe_mean([r['selected_dense'] for r in rs]),
                'a1_a2_change_rate': safe_mean([r['a1_a2_changed'] for r in rs]),
                'dense_top_above_native_top_rate': safe_mean([int(x > 0) for x in gaps]),
                'dense_minus_native_top': stats([r['dense_minus_native_top'] for r in rs if r['dense_minus_native_top'] is not None]),
                'native_shift_mean': stats([r['native_shift_mean'] for r in rs if r['native_shift_mean'] is not None]),
                'native_top': stats([r['native_top'] for r in rs if r['native_top'] is not None]),
                'dense_top': stats([r['dense_top'] for r in rs if r['dense_top'] is not None]),
                'dense_score_mean': stats([r['dense_score_mean'] for r in rs if r['dense_score_mean'] is not None]),
                'native_score_mean': stats([r['native_score_mean'] for r in rs if r['native_score_mean'] is not None])}
    groups = [group('all_matched_states', records),
              group('first_a0_a1_divergence_states', [r for r in records if r['first_divergence']]),
              group('a1_destroyed_route_prefixes', [r for r in records if r['a1_destroyed']]),
              group('a1_retained_route_prefixes', [r for r in records if not r['a1_destroyed']]),
              group('a2_destroyed_route_prefixes', [r for r in records if r['a2_destroyed']])]
    summary = {'experiment_id': 'DENSE-ACTION-CALIBRATION-001', 'read_only': True,
               'matched_state_count': len(records), 'groups': groups,
               'interpretation': {
                   'native_reference': 'A0 admissible action signatures at the same episode/high-level prefix',
                   'dense_reference': 'A1 admissible signatures absent from A0 at the matched prefix',
                   'no_outcome_used_as_feature': True,
                   'next_gate': 'calibration/OOD evidence only; no policy threshold or intervention was fitted'}}
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
