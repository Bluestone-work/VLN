#!/usr/bin/env python3
"""Freeze one state and a metadata-only alternative for every sampled route."""
import argparse, hashlib, json, random
from pathlib import Path
from audit_single_intervention import read_rows, key
from run_option_capture import digest


def valid_options(row, selected):
    return [o for o in row['options'] if o['index']>0 and o['index']!=selected and o['admissible']
            and o['action']['act']==4 and o['graph_id'] is not None]


def choose_state(rows,seed):
    eligible=[r for r in rows.values() if not r['budget_stop'] and not r['no_vp_left']
              and r['effective_index']>0 and valid_options(r,r['effective_index'])]
    if not eligible:return None
    return min(eligible,key=lambda r:hashlib.sha256('{}|{}|{}|{}|state'.format(seed,r['scene_id'],r['episode_id'],r['high_level_step']).encode()).hexdigest())


def choose_alternative(state,kind,seed):
    opts=valid_options(state,state['effective_index'])
    top=max(opts,key=lambda o:(float(o['logit']),-int(o['index'])))
    if kind=='top_logit':return top
    if kind!='seeded_graph_id':raise ValueError('Unknown alternative kind')
    others=[o for o in opts if o['index']!=top['index']]
    if not others:return None
    return min(others,key=lambda o:hashlib.sha256('{}|{}|{}|{}|{}'.format(
        seed,state['scene_id'],state['episode_id'],state['high_level_step'],o['graph_id']).encode()).hexdigest())


def schedule(root,config,kind,seed):
    graphs=read_rows(root/'capture_001/graph_options/graph_options.jsonl')
    manifest=json.loads((root/'capture_001/manifest.json').read_text())
    episodes=json.loads((root/'sampling_001/manifest.json').read_text())['route_episode_ids']
    grouped={str(e['episode_id']):[] for e in episodes}
    for k,row in graphs.items():
        if k[1] in grouped:grouped[k[1]].append(row)
    events=[];omitted=[]
    for episode,rows in grouped.items():
        state=choose_state({key(r):r for r in rows},seed)
        if state is None:
            omitted.append({'episode_id':episode,'reason':'no_nonstop_state_with_alternative'});continue
        selected=state['effective_index']
        alt=choose_alternative(state,kind,config['outcome_blind_schedule']['random_alternative_seed'])
        if alt is None:
            omitted.append({'episode_id':episode,'reason':'no_second_distinct_alternative','high_level_step':state['high_level_step']});continue
        action=next(o for o in state['options'] if o['index']==selected)
        event={'scene_id':state['scene_id'],'episode_id':episode,'high_level_step':state['high_level_step'],
               'trajectory_id':str(state['trajectory_id']),'baseline_index':selected,'alternative_index':alt['index'],
               'baseline_action':action['action'],'alternative_action':alt['action'],
               'expected_graph_ids':state['graph_ids'],'expected_graph_position':state['graph_position'],
               'selection_kind':kind,'native_logit':state['logits'][selected],
               'alternative_logit':alt['logit']}
        events.append(event)
    return {'experiment_id':config['experiment_id']+'-'+kind.upper(),'status':'frozen_before_continuation',
            'analysis_only_oracle':True,'expected_event_count':len(events),
            'base_experiment_config':str(config['_path']),'protocol':config['continuation_protocol'],
            'selection_rule':'One non-STOP state per route and candidate selected from graph IDs/masks/logits only; no outcomes or goal/reference data',
            'events':events,'omitted_routes_retained_as_controls':omitted,'hashes':{str(f):digest(f) for f in [config['_path'],Path(config['continuation_protocol']),Path('research/FULL_RETURN_TRAIN16_PROTOCOL.md'),Path(__file__),
               root/'capture_001/graph_options/graph_options.jsonl',root/'capture_001/manifest.json',root/'sampling_001/manifest.json']}}


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--kind',required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    config=json.loads(args.config.read_text());config['_path']=str(args.config)
    result=schedule(args.root,config,args.kind,config['outcome_blind_schedule']['state_seed'])
    with args.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({'kind':args.kind,'events':len(result['events']),'scenes':len({e['scene_id'] for e in result['events']})},indent=2))

if __name__=='__main__':main()
