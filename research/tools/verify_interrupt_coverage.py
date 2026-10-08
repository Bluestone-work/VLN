#!/usr/bin/env python3
"""Independent cut predicates plus the exact historical first-event function."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import numpy as np


def read(p):
    return json.loads(Path(p).read_text())


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', required=True, type=Path); a = p.parse_args()
    reg = read(a.root/'registration.json')
    for name, h in reg['hashes'].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == h
    tree = ast.parse(Path('research/tools/critical_case_driver.py').read_text())
    func = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'find_interrupts')
    module = ast.Module(body=[func]); scope = {'np': np}
    exec(compile(module, 'historical_find_interrupts', 'exec'), scope)
    actual = read(a.root/'analysis_001/cuts.json')
    expected = set(); option_checks = 0; plan_checks = 0
    for c in reg['config']['cohorts']:
        source, census = Path(c['source']), Path(c['census'])
        plan = read(census/'plan_001.json'); controls = {}
        for ep in plan['failed_episodes']:
            controls[ep] = [json.loads(l) for l in (census/c['run']/'cases'/('control_'+ep)/'trace.jsonl').open()]
        historical = scope['find_interrupts'](controls, plan['states'], plan['config']['max_interrupt_routes'])
        assert historical == read(census/c['run']/'interrupt_plan.json')['events']; plan_checks += 1
        control_map = {(ep, r['high_level_step']): r for ep, records in controls.items() for r in records}
        for line in (source/'capture_001/traces/worker_seed100.jsonl').open():
            r = json.loads(line); ep = str(r['episode_id']); step = r['high_level_step']
            if r['action']['act'] != 4:
                continue
            if (ep, step) in control_map:
                phases = control_map[(ep, step)]['phases']
            elif r['action']['back_path'] == []:
                phases = ['ghost']*len(r['primitives'])
            else:
                continue
            seq = r['primitives']; pts = np.array([r['pre_pose']['position']]+[v['pose']['position'] for v in seq])
            target = np.array(r['action']['ghost_pos']['__array__'])
            deviation = np.linalg.norm((target-np.array(r['post_pose']['position']))[[0, 2]]) >= 0.5
            qualified = any(v['collided'] for v in seq) or deviation or (len(seq) >= 3 and np.linalg.norm(np.diff(pts, axis=0), axis=1).sum() <= 0.1)
            local = []
            if qualified:
                for j in range(1, len(seq)):
                    collision = phases[j-1] == 'ghost' and seq[j-1]['action'] == 1 and seq[j-1]['collided']
                    stall = j >= 3 and all(phases[k] == 'ghost' and seq[k]['action'] == 1 and np.linalg.norm(pts[k+1]-pts[k]) <= 0.01 for k in range(j-3, j))
                    if collision or stall:
                        reasons = tuple(k for k, yes in [('collision', collision), ('stall', stall)] if yes)
                        local.append((j, reasons))
                if not local and deviation:
                    ix = [k for k, v in enumerate(seq) if phases[k] == 'ghost' and v['action'] == 1]
                    if len(ix) >= 2:
                        middle = (len(ix)-1)//2; k = ix[middle]
                        if k+1 < len(seq) and np.linalg.norm((target-pts[k+1])[[0, 2]]) > (len(ix)-middle-1)*0.25+0.5:
                            local.append((k+1, ('midpoint_deviation_oracle',)))
            for cut, reasons in local:
                expected.add((c['name'], ep, step, cut, reasons))
            option_checks += 1
    got = {(e['cohort'], e['episode_id'], e['high_level_step'], e['cut_primitive'], tuple(e['reasons'])) for e in actual}
    assert len(got) == len(actual) and got == expected
    dest = a.root/'verification_001'; dest.mkdir(exist_ok=False)
    result = {'all_checks_passed': True, 'known_phase_options_checked': option_checks,
              'cut_identity_and_reason_checks': len(got), 'historical_plan_checks': plan_checks,
              'independence': 'Separate vector geometry and three-step window predicates; exact historical function isolated by AST, no simulator import.'}
    (dest/'summary.json').write_text(json.dumps(result, indent=2)+'\n'); print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
