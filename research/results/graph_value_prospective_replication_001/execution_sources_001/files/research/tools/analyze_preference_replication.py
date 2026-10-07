#!/usr/bin/env python3
"""All-route paired analysis for the prospective frozen-model replication."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from analyze_frozen_preference_cycle import METRICS
from audit_single_intervention import read_rows, group_rows, paired_report


def point_gate(arm, random_sr):
    m=arm['metrics']
    return {'positive_sr':m['success']['delta']>0,
            'nondecreasing_spl':m['spl']['delta']>=-1e-6,
            'nondecreasing_ndtw':m['ndtw']['delta']>=-1e-6,
            'no_primitive_increase':m['primitive_action_count']['delta']<=0,
            'quality_rescue_in_two_scenes':len(arm['quality_rescue_scenes'])>=2,
            'sr_beats_mean_random':m['success']['intervention']>random_sr}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    cfg=json.loads(a.config.read_text());root=Path(cfg['replication']['root']);cycle=root/'replication_cycle_001'
    if not json.loads((cycle/'status.json').read_text()).get('all_arms_complete'):raise ValueError('Incomplete arms')
    sample=json.loads(Path(cfg['sampling_manifest']).read_text());scene_of={str(e['episode_id']):e['scene_id'] for e in sample['episodes']}
    scenes=sorted(set(scene_of.values()))
    if len({sum(s==v for v in scene_of.values()) for s in scenes})!=1:raise ValueError('Expected equal scene sample sizes')
    rng=np.random.RandomState(cfg['analysis']['bootstrap_seed']);boot=rng.randint(0,len(scenes),(cfg['analysis']['bootstrap_samples'],len(scenes)))
    def ci(values):
        means=np.asarray([np.mean([v for ep,v in values.items() if scene_of[ep]==s]) for s in scenes])
        return np.percentile(means[boot].mean(axis=1),[2.5,97.5]).tolist()
    arms={};rows_by={};files=[a.config,Path(__file__),root/'registration_001.json',root/'sampling_001/route_audit_001.json']
    for name in cfg['replication']['arms']:
        schedule=json.loads((root/'schedules_001'/(name+'.json')).read_text())
        if not schedule['events']:
            marker=json.loads((cycle/(name+'_noop.json')).read_text())
            if marker['schedule_sha256']!=hashlib.sha256((root/'schedules_001'/(name+'.json')).read_bytes()).hexdigest():raise ValueError('Noop schedule changed')
            original=group_rows(read_rows(root/'capture_001/traces/worker_seed100.jsonl'))
            baseline=json.loads((root/'noninterference_001/capture_episodes.json').read_text())
            rows,_=paired_report(baseline,baseline,set(),original,original,1e-6)
        else:
            audit=cycle/(name+'_enabled_audit');metric=cycle/(name+'_enabled_metrics')/'summary.json'
            if not json.loads((audit/'summary.json').read_text())['all_required_gates_passed'] or not json.loads(metric.read_text())['all_required_gates_passed']:raise ValueError('Invalid arm '+name)
            path=audit/'paired_episodes.json';rows=json.loads(path.read_text());files += [path,audit/'summary.json',metric]
        rows_by[name]={r['episode_id']:r for r in rows}
        if set(rows_by[name])!=set(scene_of):raise ValueError('Population changed')
        metrics={}
        for k in METRICS:
            values={r['episode_id']:r['delta'][k] for r in rows}
            metrics[k]={'baseline':float(np.mean([r['baseline'][k] for r in rows])),
                        'intervention':float(np.mean([r['intervention'][k] for r in rows])),
                        'delta':float(np.mean(list(values.values()))),'scene_cluster_ci95':ci(values)}
        rescued=[r['episode_id'] for r in rows if r['delta']['success']>0];lost=[r['episode_id'] for r in rows if r['delta']['success']<0]
        quality=[ep for ep in rescued if rows_by[name][ep]['delta']['ndtw']>=-1e-6]
        arms[name]={'episodes':len(rows),'interventions':len(schedule['events']),'eligible_states':len(schedule['eligible_states']),
                    'metrics':metrics,'success_rescued':rescued,'success_lost':lost,'quality_rescued':quality,
                    'quality_rescue_scenes':sorted({scene_of[ep] for ep in quality}),
                    'ndtw_harmed':[r['episode_id'] for r in rows if r['delta']['ndtw'] < -1e-6],
                    'spl_harmed':[r['episode_id'] for r in rows if r['delta']['spl'] < -1e-6],
                    'by_scene':{s:{k:float(np.mean([r['delta'][k] for r in rows if scene_of[r['episode_id']]==s])) for k in METRICS} for s in scenes}}
    randoms=[n for n in arms if n.startswith('random_')];comparisons={}
    for label,names in [('cost_matched',['cost_matched']),('cost_own',['cost_own']),('mean_random',randoms)]:
        comparisons['full_vs_'+label]={}
        for k in METRICS:
            values={ep:rows_by['full'][ep]['intervention'][k]-float(np.mean([rows_by[n][ep]['intervention'][k] for n in names])) for ep in scene_of}
            comparisons['full_vs_'+label][k]={'delta':float(np.mean(list(values.values()))),'scene_cluster_ci95':ci(values)}
    random_sr=float(np.mean([arms[n]['metrics']['success']['intervention'] for n in randoms]))
    gates=point_gate(arms['full'],random_sr)
    contrast=comparisons['full_vs_cost_matched']
    mechanism=contrast['success']['delta']>0 and contrast['success']['scene_cluster_ci95'][0]>0 and all(contrast[k]['delta']>=-1e-6 for k in ['spl','ndtw']) and contrast['primitive_action_count']['delta']<=0
    all_point=all(gates.values())
    out={'experiment_id':cfg['experiment_id'],'arms':arms,'comparisons':comparisons,'replication_point_gates':gates,
         'replication_point_gate_passed':all_point,'added_feature_value_gate_passed':bool(mechanism),
         'decision':'CONDITIONAL GO for broader independent evidence' if all_point else 'NO-GO for scaling this frozen first-disagreement recipe',
         'scene_clusters':len(scenes),'bootstrap_seed':cfg['analysis']['bootstrap_seed'],'bootstrap_samples':cfg['analysis']['bootstrap_samples'],
         'limits':['Prospective research-route holdout with overlapping scenes; not baseline-checkpoint holdout or benchmark.',
                   'One simulator seed; random choices are not training replications.','COST also uses old full-return labels; this tests feature sufficiency, not label-horizon necessity.',
                   'Failure of this policy does not prove proposal coverage failure or invalidate every graph value method.'],
         'hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    a.output.mkdir(exist_ok=False)
    with (a.output/'summary.json').open('x') as stream:json.dump(out,stream,indent=2);stream.write('\n')
    print(json.dumps({'decision':out['decision'],'gates':gates,'added_feature_value_gate_passed':bool(mechanism),'arms':{n:{'interventions':v['interventions'],'rescued':v['success_rescued'],'lost':v['success_lost'],'deltas':{k:v['metrics'][k]['delta'] for k in ['success','spl','ndtw','primitive_action_count']}} for n,v in arms.items()}},indent=2))

if __name__=='__main__':main()
