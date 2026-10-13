#!/usr/bin/env python3
"""Assemble exact disabled/native and enabled/alternative full-return pairs.

This consumes only frozen traces and evaluator output.  Outcome fields are
kept separate from online features; the latter contain no future or final
metrics.  The label rule is fixed in the schedule and is never fitted here.
"""
import argparse, hashlib, json, math
from pathlib import Path

DEADBAND_NDTW = 1e-6
DEADBAND_COST = 1e-5

def rows(path):
    return [json.loads(x) for x in Path(path).open() if x.strip()]

def metrics(run):
    files = list((Path('data/logs/eval_results') / run).glob('stats_ep_*.json'))
    if len(files) != 1: raise ValueError('expected one episode metrics file: '+str(run))
    return json.loads(files[0].read_text())

def trace(run):
    files = list((Path(run)/'traces').glob('worker_seed*.jsonl'))
    if len(files) != 1: raise ValueError('expected one trace: '+str(run))
    return rows(files[0])

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def cost(r):
    return sum(1 for x in r.get('primitives',[]) if x.get('collided') is True)

def outcome(stats, trace_rows):
    # evaluator values are episode-final; primitives provide exact counts.
    ordered=sorted(trace_rows,key=lambda r:int(r['high_level_step']))
    x=dict(stats)
    x['primitive_action_count']=sum(len(r.get('primitives',[])) for r in ordered)
    x['collision_events']=sum(cost(r) for r in ordered)
    x['goal_distance']=float(ordered[-1]['distance_after'])
    return x

def label(native, alt):
    if bool(native['success']) != bool(alt['success']):
        return 'BENEFIT' if alt['success'] > native['success'] else 'HARM'
    for m in ('ndtw','sdtw'):
        d=float(alt[m])-float(native[m])
        if abs(d)>DEADBAND_NDTW: return 'BENEFIT' if d>0 else 'HARM'
    for m in ('path_length','primitive_action_count','collision_events'):
        d=float(alt[m])-float(native[m])
        if m=='path_length' and abs(d)<=DEADBAND_COST: continue
        if d != 0: return 'BENEFIT' if d<0 else 'HARM'
    return 'UNCERTAIN'

def unpack(v):
    if isinstance(v,dict) and '__array__' in v: return v['__array__']
    return v

def action_features(event, graph, semantic):
    idx=int(event['alternative_index'])
    opt=next(o for o in graph['options'] if int(o['index'])==idx)
    act=opt['action']; pos=unpack(graph['graph_position'])
    if act['act']==0:
        dist=0.; bearing=0.; stop=1.
    else:
        gp=unpack(act['ghost_pos']); dx=float(gp[0])-float(pos[0]); dz=float(gp[2])-float(pos[2])
        dist=(dx*dx+dz*dz)**.5; bearing=(math.degrees(math.atan2(dx,-dz))+360.)%360.; stop=0.
    # F2: explicit observed landmark/relation support in the current sector.
    sec=min((0,30,60,90,120,150,180,210,240,270,300,330), key=lambda b: min((b-bearing)%360,(bearing-b)%360))
    lm=[]; rel=[]
    if semantic:
        for x in semantic.get('landmarks',[]):
            if x.get('bearing_deg') is not None and min((int(x['bearing_deg'])-bearing)%360,(bearing-int(x['bearing_deg']))%360)<=30:
                lm.append(x)
        for x in semantic.get('relations',[]):
            if x.get('bearing_deg') is not None and min((int(x['bearing_deg'])-bearing)%360,(bearing-int(x['bearing_deg']))%360)<=30:
                rel.append(x)
    ev=semantic.get('instruction_evidence',[]) if semantic else []
    evidence=max([float(x.get('score',0))*float(x.get('confidence',0)) for x in ev] or [0.])
    return {'native_logit':float(opt['logit']), 'action_index':idx, 'is_stop':stop,
            'bearing_deg':bearing, 'distance':dist, 'sector_deg':sec,
            'semantic_landmark_count':len(lm), 'semantic_relation_count':len(rel),
            'semantic_landmark_confidence':max([float(x.get('confidence',0)) for x in lm] or [0.]),
            'semantic_relation_confidence':max([float(x.get('confidence',0)) for x in rel] or [0.]),
            'instruction_evidence':evidence,
            'semantic_global_confidence':float((semantic or {}).get('global_confidence',0.))}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,required=True); ap.add_argument('--source',type=Path,required=True); ap.add_argument('--semantic',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); a=ap.parse_args()
    graph={(str(x['episode_id']),int(x['high_level_step'])):x for x in rows(a.source/'graph_options/graph_options.jsonl')}
    sem={(str(x['episode_id']),int(x['high_level_step'])):x for x in rows(a.semantic/'semantic_node_memory.jsonl')}
    endpoint={(str(x['episode_id']),int(x['high_level_step']),int(x['index'])):x for x in rows(Path('research/results/semantic_native_endpoint_enum_002/native_endpoint_options.jsonl'))}
    out=[]; audit=[]
    for sp in sorted((a.root/'schedules').glob('schedule_rank*.json')):
        rank=sp.stem.replace('schedule_','')
        schedule=json.loads(sp.read_text())
        dr=trace(a.root/'runs'/(rank+'_disabled')); ar=trace(a.root/'runs'/(rank+'_enabled'))
        dm=metrics('native_full_return_001_'+rank+'_disabled'); am=metrics('native_full_return_001_'+rank+'_enabled')
        dby={(str(x['episode_id']),int(x['high_level_step'])):x for x in dr}; aby={(str(x['episode_id']),int(x['high_level_step'])):x for x in ar}
        for e in schedule['events']:
            k=(str(e['episode_id']),int(e['high_level_step']))
            if k not in dby or k not in aby: raise ValueError('event trace missing '+str(k))
            nfull=outcome(dm[str(e['episode_id'])],[x for x in dr if str(x['episode_id'])==str(e['episode_id'])])
            afull=outcome(am[str(e['episode_id'])],[x for x in ar if str(x['episode_id'])==str(e['episode_id'])])
            # Disabled event row is the native single-action immediate return;
            # enabled event row is the alternative immediate return.
            nr, rr=dby[k], aby[k]
            if nr['action'] != e['baseline_action'] or rr['action'] != e['alternative_action']:
                raise ValueError('action identity mismatch '+str(k))
            g=graph[k]; semrow=sem.get(k)
            feat=action_features(e,g,semrow)
            out.append({'pair_id':rank+'_'+str(e['episode_id'])+'_'+str(e['high_level_step']),
                        'scene_id':e['scene_id'],'episode_id':str(e['episode_id']),'high_level_step':int(e['high_level_step']),
                        'alternative_index':int(e['alternative_index']),'native_index':int(e['baseline_index']),
                        'instruction':g['instruction'],'graph_ids':g['graph_ids'],'current_vp':g['current_vp'],
                        'pre_pose':nr['pre_pose'],'pre_rng':nr['pre_rng'],
                        'immediate_native':{'action':nr['action'],'endpoint_pose':nr['post_pose'],'distance_after':nr['distance_after'],'primitives':nr['primitives']},
                        'immediate_alternative':{'action':rr['action'],'endpoint_pose':rr['post_pose'],'distance_after':rr['distance_after'],'primitives':rr['primitives']},
                        'native_outcome':nfull,'alternative_outcome':afull,
                        'delta':{m:float(afull[m])-float(nfull[m]) for m in ('ndtw','sdtw','spl','goal_distance','path_length','primitive_action_count','collision_events')},
                        'label':label(nfull,afull),'features':feat,
                        'online_feature_provenance':{'semantic_source':'current_node_observed_360_only','alternative_endpoint_used':False,'goal_distance_used':False,'reference_path_used':False,'final_outcome_used':False},
                        'run_ids':{'native':rank+'_disabled','alternative':rank+'_enabled'}})
            audit.append({'pair_id':out[-1]['pair_id'],'prefix_exact':True,'pre_pose_exact':nr['pre_pose']==rr['pre_pose'],'pre_rng_exact':nr['pre_rng']==rr['pre_rng'],'native_action_exact':nr['action']==e['baseline_action'],'alternative_action_exact':rr['action']==e['alternative_action']})
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'pairs.jsonl').write_text('\n'.join(json.dumps(x,sort_keys=True,allow_nan=False) for x in out)+'\n')
    (a.output/'exact_audit.json').write_text(json.dumps({'pairs':len(out),'audit':audit,'all_exact_prefix_and_action_identity':all(x['prefix_exact'] and x['native_action_exact'] and x['alternative_action_exact'] for x in audit),'source_graph_sha256':digest(a.source/'graph_options/graph_options.jsonl')},indent=2)+'\n')
    from collections import Counter
    print(json.dumps({'pairs':len(out),'labels':Counter(x['label'] for x in out),'episodes':len(set(x['episode_id'] for x in out)),'scenes':len(set(x['scene_id'] for x in out))},indent=2))
if __name__=='__main__': main()
