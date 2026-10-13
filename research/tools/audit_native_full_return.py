#!/usr/bin/env python3
"""Fail-closed audit of every frozen full-return branch; no outcome fitting."""
import argparse
import hashlib
import json
from pathlib import Path
from collections import Counter
from build_native_full_return_pairs import rows, digest, metrics, outcome, label


def key(r):
    return (r['scene_id'], str(r['episode_id']), int(r['high_level_step']))


def indexed(rs):
    out = {key(r): r for r in rs}
    if len(out) != len(rs):
        raise ValueError('duplicate exact state key')
    return out


def prefix_checks(event, disabled, enabled, dh, ah):
    k = key(event)
    prior = [q for q in disabled if q[:2] == k[:2] and q[2] < k[2]]
    through = prior + [k]
    checks = {'prefix_exact': all(q in enabled and disabled[q] == enabled[q] for q in prior)}
    for field in ('pre_pose', 'pre_rng', 'pre_metrics', 'distance_before'):
        checks[field + '_exact'] = k in enabled and disabled[k][field] == enabled[k][field]
    for field in ('graphmap_sha256', 'policy_rng_sha256', 'graph_ids', 'mask', 'visited',
                  'instruction', 'trajectory_id', 'native_action', 'native_index', 'effective_native_index'):
        checks[field + '_exact'] = all(q in ah and dh[q][field] == ah[q][field] for q in through)
    checks['native_action_exact'] = disabled[k]['action'] == event['baseline_action']
    checks['alternative_action_exact'] = k in enabled and enabled[k]['action'] == event['alternative_action']
    checks['event_exact'] = k in ah and (ah[k]['scheduled'] and ah[k]['intervened'] and
        ah[k]['output_index'] == event['alternative_index'] and
        dh[k]['output_index'] == event['baseline_index'] and
        dh[k]['graph_ids'] == event['expected_graph_ids'])
    return checks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    source_manifest = json.loads((a.source/'manifest.json').read_text())
    source_trace = a.source/'traces/worker_seed100.jsonl'
    baseline = indexed(rows(source_trace))
    graph = indexed(rows(a.source/'graph_options/graph_options.jsonl'))
    source_metrics = metrics(source_manifest['exp_name'])
    source_hash = digest(source_trace)
    pairs, audits, run_audits = [], [], []
    for sp in sorted((a.root/'schedules').glob('schedule_rank*.json')):
        rank = sp.stem.replace('schedule_', '')
        schedule = json.loads(sp.read_text())
        run = {}
        for mode in ('disabled', 'enabled'):
            rd = a.root/'runs'/(rank+'_'+mode)
            mf = json.loads((rd/'manifest.json').read_text())
            sm = json.loads((rd/'hook/summary.json').read_text())
            tp = rd/'traces/worker_seed100.jsonl'
            hp = rd/'hook/decisions.jsonl'
            t, h = indexed(rows(tp)), indexed(rows(hp))
            m = metrics(mf['exp_name'])
            cfg = dict(source_manifest['overrides'], TRAINER_NAME='SS-ETP-NativeFullReturn')
            checks = {
                'manifest_schedule': mf['schedule'] == schedule,
                'manifest_source': mf['source_manifest_sha256'] == digest(a.source/'manifest.json'),
                'manifest_config': mf['config'] == source_manifest['config'] and mf['overrides'] == cfg,
                'enabled_identity': mf['enabled'] == (mode == 'enabled') == sm['enabled'],
                'hook_coverage': sm['hook_completed'] and sm['decisions'] == len(t) == len(h) and set(t) == set(h),
                'all_episodes_terminated': all(max((r for r in t.values() if str(r['episode_id']) == ep), key=lambda r:r['high_level_step'])['done'] for ep in m),
                'metric_episode_set': set(m) == set(source_metrics),
                'executed_expected_actions': all(t[k]['action'] == v['expected_action'] and v['native_action_identity_verified'] for k,v in h.items()),
                'native_continuation': all(v['intervened'] or (v['output_index'] == v['effective_native_index'] and t[k]['action'] == v['native_action']) for k,v in h.items()),
                'intervention_coverage': {k for k,v in h.items() if v['intervened']} == ({key(e) for e in schedule['events']} if mode == 'enabled' else set()),
            }
            if mode == 'disabled':
                checks.update(source_trajectory=digest(tp) == source_hash, source_metrics=m == source_metrics,
                              source_graph_identity=all(v['baseline_prefix_verified'] and v['instruction']==graph[k]['instruction'] and v['graph_ids']==graph[k]['graph_ids'] and v['native_index']==graph[k]['policy_index'] for k,v in h.items()))
            run_audits.append({'run': rank+'_'+mode, 'checks': checks, 'pass': all(checks.values()),
                'trace_sha256': digest(tp), 'hook_sha256': digest(hp), 'manifest_sha256': digest(rd/'manifest.json'),
                'schedule_sha256': digest(sp), 'metrics_sha256': digest(next((Path('data/logs/eval_results')/mf['exp_name']).glob('stats_ep_*.json')))})
            run[mode] = t,h,m,all(checks.values())
        dt,dh,dm,dp = run['disabled']; at,ah,am,apass = run['enabled']
        for e in schedule['events']:
            k=key(e); checks=prefix_checks(e,dt,at,dh,ah)
            checks['run_integrity'] = dp and apass
            checks['source_graph_event'] = graph[k]['graph_ids']==e['expected_graph_ids'] and graph[k]['graph_position']==e['expected_graph_position'] and next(o for o in graph[k]['options'] if o['index']==e['alternative_index'])['action']==e['alternative_action']
            pid=rank+'_'+str(e['episode_id'])+'_'+str(e['high_level_step'])
            audits.append({'pair_id':pid,'state_key':list(k),'checks':checks,'pass':all(checks.values())})
            if not all(checks.values()): continue
            ep=str(e['episode_id'])
            no=outcome(dm[ep],[r for q,r in dt.items() if q[:2]==k[:2]])
            ao=outcome(am[ep],[r for q,r in at.items() if q[:2]==k[:2]])
            pairs.append({'pair_id':pid,'scene_id':k[0],'episode_id':ep,'high_level_step':k[2],
                'trajectory_id':e['trajectory_id'],'native_index':e['baseline_index'],'alternative_index':e['alternative_index'],
                'native_outcome':no,'alternative_outcome':ao,'label':label(no,ao),
                'delta':{m:float(ao[m])-float(no[m]) for m in ('ndtw','sdtw','spl','goal_distance','path_length','primitive_action_count','collision_events')},
                'native_run':rank+'_disabled','alternative_run':rank+'_enabled'})
        if int(rank[4:]) % 25 == 0: print(rank, 'audited',len(audits), 'valid',len(pairs),flush=True)
    # A failed sentinel invalidates the entire state, including its other branches.
    bad_states={tuple(x['state_key']) for x in audits if not x['pass']}
    pairs=[p for p in pairs if key(p) not in bad_states]
    summary={'status':'PASS' if not bad_states and all(r['pass'] for r in run_audits) else 'FAIL',
        'runs':len(run_audits),'attempted_pairs':len(audits),'valid_pairs':len(pairs),'invalid_states':len(bad_states),
        'states':len({key(p) for p in pairs}),'episodes':len({p['episode_id'] for p in pairs}),
        'routes':len({p['trajectory_id'] for p in pairs}),'scenes':len({p['scene_id'] for p in pairs}),
        'labels':dict(Counter(p['label'] for p in pairs)),
        'source_trace_sha256':source_hash,'source_graph_sha256':digest(a.source/'graph_options/graph_options.jsonl'),
        'checkpoint_sha256_current':digest(source_manifest['config']['checkpoint']),
        'checkpoint_note':'Manifests record same checkpoint path, not a contemporaneous file hash. Current hash is provenance, not proof of historical byte identity.',
        'failure_counts':dict(Counter(f for x in audits for f,ok in x['checks'].items() if not ok))}
    for name,obj in [('summary.json',summary),('runs.json',run_audits),('pairs_audit.json',audits)]:
        (a.output/name).write_text(json.dumps(obj,indent=2)+'\n')
    (a.output/'pairs.jsonl').write_text(''.join(json.dumps(p,sort_keys=True)+'\n' for p in pairs))
    print(json.dumps(summary,indent=2))
    if summary['status']!='PASS': raise SystemExit(1)

if __name__=='__main__': main()
