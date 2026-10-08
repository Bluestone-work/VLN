#!/usr/bin/env python3
"""Full-return, route-level timing opportunity and successful-route harm."""
import argparse
import hashlib
import json
from pathlib import Path


def read(p):
    return json.loads(Path(p).read_text())


def classify(row):
    m, d = row['metrics'], row['delta']
    native_success = bool(row['baseline_metrics']['success'])
    rescue = not native_success and bool(m['success'])
    quality = rescue and d['ndtw'] >= -1e-6
    return {'sr_rescue': rescue, 'quality_rescue': quality,
            'cost_capped_quality_rescue': quality and d['steps_taken'] <= 0,
            'success_loss': native_success and not bool(m['success']),
            'ndtw_loss': d['ndtw'] < -1e-6, 'cost_increase': d['steps_taken'] > 0,
            'identical_metrics': all(abs(v) <= 1e-6 for v in d.values())}


def main():
    p = argparse.ArgumentParser(); p.add_argument('--run', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path); a = p.parse_args()
    manifest, status = read(a.run/'manifest.json'), read(a.run/'summary.json')
    assert status['status'] == 'complete' and status['all_validity_checks_passed']
    plan = manifest['plan']; rows = [json.loads(l) for l in (a.run/'results.jsonl').open()]
    modes = ['sense_only', 'interrupt_consume', 'interrupt_retain']
    expected = {'control_'+ep for ep in plan['replay_episodes']}
    expected.update('{}_{}_s{}_p{}'.format(mode, e['episode_id'], e['high_level_step'], e['cut_primitive'])
                    for mode in modes for e in plan['interrupt_events'])
    assert len(rows) == len(expected) and {r['case']['case_id'] for r in rows} == expected
    assert all(r['validity_passed'] and r['metric_reconstruction']['passed'] for r in rows)
    assert all(all(abs(v) <= 1e-6 for v in r['delta'].values()) for r in rows if r['case']['mode'] in ['control', 'sense_only'])
    details = []
    for r in rows:
        if r['case']['mode'] in ['control', 'sense_only']:
            continue
        assert r['case']['native_success'] == bool(r['baseline_metrics']['success'])
        details.append(dict(case=r['case'], metrics=r['metrics'], baseline=r['baseline_metrics'],
                            delta=r['delta'], classification=classify(r),
                            policy_calls=r['policy_calls'], observation_at_calls=r['observation_at_calls'],
                            collision_events=r['primitive_collision_events']))
    route_rows, summary = [], {}
    tags = ['sr_rescue', 'quality_rescue', 'cost_capped_quality_rescue', 'success_loss', 'ndtw_loss', 'cost_increase', 'identical_metrics']
    for mode in modes[1:]:
        for ep in plan['replay_episodes']:
            rr = sorted([r for r in details if r['case']['episode_id'] == ep and r['case']['mode'] == mode],
                        key=lambda r: (r['case']['high_level_step'], r['case']['cut_primitive']))
            assert rr
            item = {'episode_id': ep, 'scene_id': rr[0]['case']['scene_id'], 'mode': mode,
                    'native_success': bool(rr[0]['baseline']['success']), 'tested_cuts': len(rr),
                    'any': {tag: any(r['classification'][tag] for r in rr) for tag in tags},
                    'case_counts': {tag: sum(r['classification'][tag] for r in rr) for tag in tags},
                    'earliest': {'case_id': rr[0]['case']['case_id'], 'delta': rr[0]['delta'], 'classification': rr[0]['classification']}}
            quality = [r for r in rr if r['classification']['quality_rescue']]
            item['quality_rescue_examples'] = [r['case']['case_id'] for r in quality]
            route_rows.append(item)
        summary[mode] = {}
        for success in [False, True]:
            rr = [r for r in route_rows if r['mode'] == mode and r['native_success'] == success]
            quality_scenes = {r['scene_id'] for r in rr if r['any']['quality_rescue']}
            summary[mode]['success' if success else 'failure'] = {
                'routes': len(rr), 'scenes': len({r['scene_id'] for r in rr}),
                'any_cut_routes': {tag: [r['episode_id'] for r in rr if r['any'][tag]] for tag in tags},
                'quality_rescue_scenes': len(quality_scenes),
                'earliest_cut_routes': {tag: [r['episode_id'] for r in rr if r['earliest']['classification'][tag]] for tag in tags},
                'earliest_mean_deltas': {key: sum(r['earliest']['delta'][key] for r in rr)/len(rr) for key in ['success', 'spl', 'ndtw', 'steps_taken', 'path_length']}}
    passed = any(len(s['failure']['any_cut_routes']['quality_rescue']) >= 2 and s['failure']['quality_rescue_scenes'] >= 2 for s in summary.values())
    result = {'status': 'complete', 'experiment_id': plan['config']['experiment_id'],
              'rollouts': len(rows), 'events': len(plan['interrupt_events']), 'route_controls': len(plan['replay_episodes']),
              'metric_reconstruction_checks': sum(r['metric_reconstruction']['comparisons'] for r in rows),
              'arms': summary, 'gate_passed_for_independent_confirmation_only': passed,
              'decision': 'CONDITIONAL GO for independent confirmation only' if passed else 'NO-GO for expanding this collision-interrupt recipe',
              'limits': ['Privileged retrospective timing oracle on previously inspected training routes.',
                         'Correlated cuts; no trained trigger and no benchmark claim.',
                         'First eligible event evaluated on selected symptom-exposed routes only.',
                         'Does not test backtracking, semantic events or repeated interruptions.'],
              'hashes': {str(q): hashlib.sha256(q.read_bytes()).hexdigest() for q in [a.run/'manifest.json', a.run/'results.jsonl', Path(__file__)]}}
    a.output.mkdir(parents=True, exist_ok=False)
    for name, data in [('summary', result), ('routes', route_rows), ('cases', details)]:
        (a.output/(name+'.json')).write_text(json.dumps(data, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
