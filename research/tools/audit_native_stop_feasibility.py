#!/usr/bin/env python3
"""Audit existing exact native STOP branches; no rollout or fitted trigger."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
from audit_single_intervention import key, read_rows, group_rows, one
from run_option_capture import digest


def success(distance):
    if not np.isfinite(distance):
        raise ValueError('Nonfinite goal distance')
    return distance <= 3.


def failure_flags(route):
    return {'missed_native_stop': not route['success'] and bool(route['successful_counterfactual_stop_steps']),
            'oracle_arrival_without_boundary': not route['success'] and route['oracle_success'] and not route['boundary_arrival_le_3m'],
            'boundary_arrival_without_native_stop': not route['success'] and route['boundary_arrival_le_3m'] and not route['successful_counterfactual_stop_steps']}


def cohort(c):
    root = Path(c['root']); probe = root / c['probe']; ni = root / c['noninterference']
    gate = json.loads((ni/'summary.json').read_text())
    if not gate['baseline_noninterference_gate_passed']:
        raise ValueError('Noninterference gate failed: '+str(root))
    integrity_path = root/'integrity_001/summary.json'
    integrity = json.loads(integrity_path.read_text())
    if not integrity['integrity_gate_passed']:
        raise ValueError('Integrity gate failed: '+str(root))
    probe_summary = json.loads((probe/'summary.json').read_text())
    if not probe_summary.get('all_required_gates_passed', False):
        raise ValueError('Probe gate failed: '+str(root))
    metrics = json.loads((ni/'capture_episodes.json').read_text())
    capture = root/'capture_001'; trace_path = one(capture/'traces','*.jsonl')
    traces = group_rows(read_rows(trace_path))
    graph_path = capture/'graph_options/graph_options.jsonl'
    graphs = {key(r): r for r in (json.loads(line) for line in graph_path.open())}
    decisions = {key(r): r for r in (json.loads(line) for line in (probe/'decisions.jsonl').open())}
    branch_path = probe/'branch_traces.jsonl'; stops={}
    raw_hash=digest(branch_path)
    if raw_hash != integrity['raw_trace_sha256']:
        raise ValueError('Raw branch hash differs from independent integrity audit')
    for line in branch_path.open():
        row=json.loads(line)
        if row['order']=='forward' and row['index']==0:
            k=tuple(row['decision_key'])
            if k in stops: raise ValueError('Duplicate forward STOP')
            stops[k]=row['trace']
    if set(stops) != set(decisions): raise ValueError('STOP branch coverage mismatch')
    if set(metrics)!=set(traces) or len(stops)!=sum(len(t) for t in traces.values()):
        raise ValueError('Incomplete cohort coverage')
    rows=[]
    for k, decision in decisions.items():
        ep=str(k[1]); graph=graphs[k]; baseline=traces[ep][k[2]]; stop=stops[k]
        if stop['action']['act'] != 0 or not stop['done']:
            raise ValueError('Index-0 branch is not native terminal STOP')
        if (np.linalg.norm(np.asarray(stop['pre_pose']['position'])-baseline['pre_pose']['position'])>1e-5
                or stop['pre_rng'] != baseline['pre_rng'] or abs(stop['distance_before']-baseline['distance_before'])>1e-5):
            raise ValueError('STOP branch does not start at exact baseline state')
        option=next(o for o in graph['options'] if o['index']==0)
        if stop['action']!=option['action'] or not option['admissible']:
            raise ValueError('Native STOP action dictionary mismatch')
        if not stop['primitives'] or stop['primitives'][-1]['action']!=0:
            raise ValueError('Missing primitive STOP')
        if graph['effective_index']==0:
            if stop['action']!=baseline['action'] or stop['primitives']!=baseline['primitives'] or abs(stop['distance_after']-baseline['distance_after'])>1e-6:
                raise ValueError('Selected STOP does not reproduce baseline execution')
        if stop['post_metrics']['steps_taken'] != len(stop['primitives']):
            raise ValueError('STOP primitive count mismatch')
        pre=float(baseline['distance_before']); terminal=float(stop['distance_after'])
        rows.append({'scene_id':k[0],'episode_id':ep,'high_level_step':k[2],
                     'pre_distance_m':pre,'pre_near_le_3m':success(pre),
                     'native_index':graph['effective_index'],'native_was_stop':graph['effective_index']==0,
                     'native_action_type':baseline['action']['act'],
                     'counterfactual_stop_success':success(terminal),'stop_terminal_distance_m':terminal,
                     'stop_primitive_count':len(stop['primitives']),'stop_back_path_nodes':len(stop['action'].get('back_path') or []),
                     'baseline_boundary_after_m':float(baseline['distance_after']),
                     'baseline_boundary_near_le_3m':float(baseline['distance_after'])<=3.,
                     'baseline_episode_success':bool(metrics[ep]['success']),
                     'baseline_episode_oracle_success':bool(metrics[ep]['oracle_success'])})
    routes={}
    for ep, rr in traces.items():
        m=metrics[ep]; endpoint_dist=[float(x['distance_after']) for x in rr]
        relevant=[r for r in rows if r['episode_id']==ep]
        success_stops=[r for r in relevant if r['counterfactual_stop_success']]
        if bool(m['success'])!=success(endpoint_dist[-1]) or abs(m['distance_to_goal']-endpoint_dist[-1])>1e-6:
            raise ValueError('Native final success/goal distance mismatch')
        if sum(len(x['primitives']) for x in rr)!=m['steps_taken'] or len(rr)!=m['high_level_steps']:
            raise ValueError('Baseline action counts differ')
        boundary_dist=[float(rr[0]['distance_before'])]+endpoint_dist
        baseline_boundaries=[x for x in boundary_dist if success(x)]
        routes[ep]={'scene_id':rr[0]['scene_id'],'success':bool(m['success']),'oracle_success':bool(m['oracle_success']),
                    'final_distance_m':float(m['distance_to_goal']),'min_decision_boundary_m':min(boundary_dist),
                    'boundary_arrival_le_3m':bool(baseline_boundaries),'successful_counterfactual_stop_steps':[r['high_level_step'] for r in success_stops],
                    'native_stop_steps':[r['high_level_step'] for r in relevant if r['native_was_stop']],
                    'failed_route':not bool(m['success'])}
        routes[ep].update(failure_flags(routes[ep]))
    failures=[x for x in routes.values() if x['failed_route']]
    cross={}
    for near in [False,True]:
        for succ in [False,True]:
            vals=[r for r in rows if r['pre_near_le_3m']==near and r['counterfactual_stop_success']==succ]
            cross['pre_near_{}_stop_success_{}'.format(near,succ)]={'decisions':len(vals),'routes':len(set(v['episode_id'] for v in vals))}
    return {'cohort':c['name'],'episodes':len(routes),'decisions':len(rows),'successes':sum(not r['failed_route'] for r in routes.values()),
            'failures':len(failures),'decision_cross_tab':cross,'failed_routes':failures,'routes':routes,'decisions_rows':rows,
            'exact_3m_boundary_count':sum(r['pre_distance_m']==3. or r['baseline_boundary_after_m']==3. or r['stop_terminal_distance_m']==3. for r in rows),
            'missed_native_stop_failures':sum(r['missed_native_stop'] for r in failures),
            'oracle_arrival_without_boundary_failures':sum(r['oracle_arrival_without_boundary'] for r in failures),
            'boundary_arrival_without_native_stop_failures':sum(r['boundary_arrival_without_native_stop'] for r in failures),
            'all_required_gates_passed':True,
            'source_hashes':dict({str(x):digest(x) for x in [root/c['probe']/ 'summary.json',integrity_path,ni/'summary.json',ni/'capture_episodes.json',trace_path,graph_path,probe/'decisions.jsonl']}, **{str(branch_path):raw_hash})}


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);args=p.parse_args();cfg=json.loads(args.config.read_text())
    result={'experiment_id':cfg['experiment_id'],'status':'complete','post_hoc':True,'no_model_trained':True,'no_new_rollout':True,
            'success_operator':'<= 3 m','cohorts':{c['name']:cohort(c) for c in cfg['cohorts']},
            'limits':cfg['limits'],'git_commit':subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),'python':sys.version}
    out=Path(cfg['output_dir']);out.mkdir(parents=True,exist_ok=False);(out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    md=['# Native STOP feasibility diagnostic','', '| Cohort | Episodes | Successes | Failures | Counterfactual successful STOP decisions | Failed routes with one |','| --- | ---: | ---: | ---: | ---: | ---: |']
    for name,x in result['cohorts'].items():
        md.append('| {} | {} | {} | {} | {} | {} |'.format(name,x['episodes'],x['successes'],x['failures'],sum(x['decision_cross_tab'][k]['decisions'] for k in x['decision_cross_tab'] if k.endswith('True')),sum(bool(v['successful_counterfactual_stop_steps']) for v in x['failed_routes'])))
    (out/'summary.md').write_text('\n'.join(md)+'\n')
    print(json.dumps({k:{a:v[a] for a in ['episodes','successes','failures','decision_cross_tab']} for k,v in result['cohorts'].items()},indent=2))


if __name__=='__main__':main()
