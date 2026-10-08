#!/usr/bin/env python3
"""Offline interrupt exposure/coverage; never imports a simulator or learns."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
from datetime import datetime, timezone


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    with Path(path).open() as f:
        return [json.loads(line) for line in f]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def distance(a, b, axes=(0, 1, 2)):
    return math.sqrt(sum((a[i]-b[i])**2 for i in axes))


def symptoms(rec):
    seq = rec['primitives']
    positions = [rec['pre_pose']['position']] + [p['pose']['position'] for p in seq]
    target = rec['action']['ghost_pos']['__array__']
    motion = sum(distance(a, b) for a, b in zip(positions, positions[1:]))
    error = distance(rec['post_pose']['position'], target, (0, 2))
    return {'collision': any(p['collided'] for p in seq),
            'deviation': error >= 0.5, 'stall': len(seq) >= 3 and motion <= 0.1}


def eligible_cuts(rec, flags, phases):
    """Old eligibility extended to all collision/stall times within each option."""
    if rec['action']['act'] != 4 or not any(flags.values()):
        return []
    seq = rec['primitives']
    assert len(seq) == len(phases)
    positions = [rec['pre_pose']['position']] + [p['pose']['position'] for p in seq]
    cuts = []
    streak = 0
    for i, (p, phase) in enumerate(zip(seq, phases)):
        if phase != 'ghost':
            streak = 0
            continue
        forward = p['action'] == 1
        streak = streak + 1 if forward and distance(positions[i+1], positions[i]) <= 0.01 else 0
        if i+1 == len(seq):
            continue
        reasons = []
        if forward and p['collided']:
            reasons.append('collision')
        if streak >= 3:
            reasons.append('stall')
        if reasons:
            cuts.append({'cut_primitive': i+1, 'reasons': reasons})
    if not cuts and flags['deviation']:
        indices = [i for i, p in enumerate(seq) if phases[i] == 'ghost' and p['action'] == 1]
        if len(indices) >= 2:
            j = (len(indices)-1)//2
            i = indices[j]
            residual = distance(rec['action']['ghost_pos']['__array__'], positions[i+1], (0, 2))
            if residual > (len(indices)-j-1)*0.25+0.5 and i+1 < len(seq):
                cuts.append({'cut_primitive': i+1, 'reasons': ['midpoint_deviation_oracle']})
    return cuts


def input_paths(cfg):
    paths = [cfg['protocol'], __file__, 'research/tools/test_interrupt_coverage.py',
             'research/tools/critical_case_driver.py',
             'vlnce_baselines/adaptive_action/critical_census.py',
             'vlnce_baselines/common/environments.py']
    for c in cfg['cohorts']:
        source, census = Path(c['source']), Path(c['census'])
        paths += [source/'capture_001/manifest.json', source/'capture_001/traces/worker_seed100.jsonl',
                  source/'noninterference_001/summary.json', source/'noninterference_001/capture_episodes.json',
                  census/'plan_001.json', census/c['run']/'interrupt_plan.json',
                  census/'independent_audit_001/summary.json']
        plan = read(census/'plan_001.json')
        paths += [census/c['run']/'cases'/('control_'+ep)/'trace.jsonl' for ep in plan['failed_episodes']]
    return sorted(set(str(Path(p)) for p in paths))


def analyze(c):
    source, census = Path(c['source']), Path(c['census'])
    assert read(source/'noninterference_001/summary.json')['baseline_noninterference_gate_passed']
    assert read(census/'independent_audit_001/summary.json')['all_checks_passed']
    manifest = read(source/'capture_001/manifest.json')
    assert manifest['config']['split'] == 'train' and manifest['config']['seed'] == 100
    metrics = read(source/'noninterference_001/capture_episodes.json')
    plan = read(census/'plan_001.json')
    failed = {ep for ep, m in metrics.items() if not m['success']}
    assert failed == set(plan['failed_episodes'])
    native = rows(source/'capture_001/traces/worker_seed100.jsonl')
    assert {str(r['episode_id']) for r in native} == set(metrics)
    controls = {}
    for ep in sorted(failed):
        for r in rows(census/c['run']/'cases'/('control_'+ep)/'trace.jsonl'):
            key = (str(r['episode_id']), r['high_level_step'])
            assert key not in controls
            controls[key] = r
    states = {(s['episode_id'], s['high_level_step']): s for s in plan['states']}
    options, cuts, route_rows = [], [], []
    checked = empty_checked = 0
    seen_controls = set()
    for r in native:
        ep, step = str(r['episode_id']), r['high_level_step']
        key = (ep, step)
        phases = None
        if key in controls:
            old = controls[key]
            for name in ['action', 'primitives', 'pre_pose', 'post_pose', 'pre_rng', 'post_rng',
                         'observation_hashes', 'distance_before', 'distance_after', 'done']:
                assert old[name] == r[name], (key, name)
            phases = old['phases']
            # Habitat STOP bypasses wrap_act, hence has no phase entry.
            if r['action']['act'] == 4:
                assert len(phases) == len(r['primitives'])
            checked += 1
            seen_controls.add(key)
        if r['action']['act'] != 4:
            continue
        assert r['action']['back_path'] is not None, 'Teleporting configuration not supported'
        empty = r['action']['back_path'] == []
        if empty:
            if phases is not None:
                assert all(p == 'ghost' for p in phases)
                empty_checked += 1
            else:
                phases = ['ghost'] * len(r['primitives'])
        flags = symptoms(r)
        if ep in failed:
            registered = set(states.get(key, {}).get('reasons', [])) & {'collision', 'deviation', 'stall'}
            assert registered == {k for k, v in flags.items() if v}, (key, registered, flags)
        events = eligible_cuts(r, flags, phases) if phases is not None else None
        option = {'cohort': c['name'], 'episode_id': ep, 'scene_id': r['scene_id'],
                  'high_level_step': step, 'native_success': bool(metrics[ep]['success']),
                  'empty_back_path': empty, 'phase_known': phases is not None,
                  'symptoms': flags, 'any_symptom': any(flags.values()),
                  'eligible_cuts': None if events is None else len(events),
                  'primitive_count': len(r['primitives'])}
        options.append(option)
        for event in events or []:
            cuts.append(dict(event, cohort=c['name'], episode_id=ep, scene_id=r['scene_id'],
                             high_level_step=step, native_success=option['native_success'],
                             empty_back_path=empty, baseline_primitive_count=len(r['primitives'])))
    assert seen_controls == set(controls)
    failed_cuts = sorted((e for e in cuts if not e['native_success']),
                         key=lambda e: (e['episode_id'], e['high_level_step'], e['cut_primitive']))
    first = {}
    for e in failed_cuts:
        first.setdefault(e['episode_id'], e)
    reconstructed = []
    for e in list(first.values())[:plan['config']['max_interrupt_routes']]:
        reconstructed.append({k: e[k] for k in ['episode_id', 'scene_id', 'high_level_step', 'cut_primitive', 'baseline_primitive_count']})
        reconstructed[-1]['cut_reason'] = e['reasons'][0]
    tested = read(census/c['run']/'interrupt_plan.json')['events']
    assert reconstructed == tested, 'Historical interrupt plan does not reproduce'
    tested_keys = {(e['episode_id'], e['high_level_step'], e['cut_primitive']) for e in tested}
    for e in cuts:
        e['historically_tested'] = (e['episode_id'], e['high_level_step'], e['cut_primitive']) in tested_keys
    for ep in sorted(metrics):
        rr = [o for o in options if o['episode_id'] == ep]
        ec = [e for e in cuts if e['episode_id'] == ep]
        scene = next(r['scene_id'] for r in native if str(r['episode_id']) == ep)
        route_rows.append({'cohort': c['name'], 'episode_id': ep, 'scene_id': scene,
                           'native_success': bool(metrics[ep]['success']), 'move_options': len(rr),
                           'symptom_options': sum(o['any_symptom'] for o in rr),
                           'phase_unknown_options': sum(not o['phase_known'] for o in rr),
                           'known_eligible_cuts': len(ec),
                           'known_eligible_options': len({e['high_level_step'] for e in ec}),
                           'historically_tested_cuts': sum(e['historically_tested'] for e in ec)})
    summary = {'cohort': c['name'], 'native_control_options_exact': checked,
               'empty_path_phase_checks': empty_checked, 'historical_plan_exact': True,
               'historically_tested_cuts': len(tested), 'strata': {}}
    for success in [False, True]:
        rr = [r for r in route_rows if r['native_success'] == success]
        oo = [o for o in options if o['native_success'] == success]
        ee = [e for e in cuts if e['native_success'] == success]
        summary['strata']['success' if success else 'failure'] = {
            'routes': len(rr), 'scenes': len({r['scene_id'] for r in rr}),
            'routes_with_symptoms': sum(r['symptom_options'] > 0 for r in rr),
            'move_options': len(oo), 'symptom_options': sum(o['any_symptom'] for o in oo),
            'symptom_type_counts': {k: sum(o['symptoms'][k] for o in oo) for k in ['collision', 'deviation', 'stall']},
            'phase_unknown_options': sum(not o['phase_known'] for o in oo),
            'known_eligible_cuts': len(ee),
            'known_eligible_options': len({(e['episode_id'], e['high_level_step']) for e in ee}),
            'known_eligible_routes': len({e['episode_id'] for e in ee}),
            'known_eligible_scenes': len({e['scene_id'] for e in ee}),
            'cut_reasons_overlapping': dict(Counter(reason for e in ee for reason in e['reasons'])),
            'empty_back_path_options': sum(o['empty_back_path'] for o in oo),
            'empty_back_path_eligible_options': sum(o['empty_back_path'] and bool(o['eligible_cuts']) for o in oo),
            'empty_back_path_eligible_routes': len({e['episode_id'] for e in ee if e['empty_back_path']})}
    return summary, options, cuts, route_rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True, type=Path)
    p.add_argument('--register', action='store_true')
    a = p.parse_args()
    cfg = read(a.config)
    out = Path(cfg['output'])
    if a.register:
        out.mkdir(parents=True, exist_ok=False)
        paths = input_paths(cfg) + [str(a.config)]
        registration = {'created_utc': datetime.now(timezone.utc).isoformat(),
                        'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip(),
                        'python': platform.python_version(), 'hardware': platform.platform(),
                        'config': cfg, 'hashes': {path: digest(path) for path in paths}}
        (out/'registration.json').write_text(json.dumps(registration, indent=2)+'\n')
        print('Registered {} inputs/sources before analysis'.format(len(paths)))
        return
    reg = read(out/'registration.json')
    assert reg['config'] == cfg
    for path, h in reg['hashes'].items():
        assert digest(path) == h, path
    dest = out/'analysis_001'
    dest.mkdir(exist_ok=False)
    summaries, options, cuts, routes = [], [], [], []
    for c in cfg['cohorts']:
        s, oo, ee, rr = analyze(c)
        summaries.append(s); options.extend(oo); cuts.extend(ee); routes.extend(rr)
    assert len({r['episode_id'] for r in routes}) == len(routes), 'Cohort episode overlap'
    result = {'experiment_id': cfg['experiment_id'], 'status': 'complete', 'cohorts': summaries,
              'no_training_or_rollouts': True, 'future_interrupt_outcomes_not_read': True,
              'limits': ['Old training cohorts, not held-out evidence.',
                         'Success eligibility is a lower bound; nonempty back-path phases unknown.',
                         'Symptoms are not failure labels; untested cuts have no inferred returns.',
                         'No online trigger or causal benefit is established.']}
    for name, data in [('summary', result), ('options', options), ('cuts', cuts), ('routes', routes)]:
        (dest/(name+'.json')).write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
