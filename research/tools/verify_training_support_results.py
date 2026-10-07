#!/usr/bin/env python3
"""Recompute pair labels and masked top choices without the audit's selectors."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from graph_value_feasibility import action_features, METRICS
from audit_preference_training_support import digest


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',type=Path,required=True); ap.add_argument('--root',type=Path,required=True)
    args=ap.parse_args(); cfg=json.loads(args.config.read_text()); plan=json.loads(Path(cfg['training_plan']).read_text())
    gp=Path(plan['config']['source_root'])/'capture_001/graph_options/graph_options.jsonl'
    graphs={}
    for line in gp.open():
        row=json.loads(line); graphs[(str(row['episode_id']),row['high_level_step'])]=row
    returns={}
    for line in Path(cfg['training_returns']).open():
        row=json.loads(line); case=row['case']
        if case['mode']!='action':continue
        values=dict(row['metrics']); values['primitive_action_count']=int(round(values.get('primitive_action_count',values['steps_taken'])))
        returns[(str(case['episode_id']),case['high_level_step'],case['action_index'])]=values
    model=json.loads(Path(cfg['frozen_model']).read_text())
    w=np.asarray(model['weights_standardized']); mu=np.asarray(model['mean']); scale=np.asarray(model['scale'])
    def score(g,o):return float(np.dot(w,(action_features(g,o)-mu)/scale))
    directions=np.asarray([1 if k in ['success','spl','ndtw'] else -1 for k in METRICS])
    eps=np.asarray([0 if k=='primitive_action_count' else cfg['metric_tolerance'] for k in METRICS])
    def label(a,b):
        diff=(np.asarray([a[k] for k in METRICS])-np.asarray([b[k] for k in METRICS]))*directions
        if np.all(diff>=-eps) and np.any(diff>eps):return 'strict',True
        if np.all(diff<=eps) and np.any(diff<-eps):return 'strict',False
        return ('tie' if np.all(np.abs(diff)<=eps) else 'mixed'),None
    pair_checks=0; choice_checks=0; return_checks=0
    pp=args.root/'analysis_001/pairs.csv'; sp=args.root/'analysis_001/states.json'
    with pp.open(newline='') as stream:
        for p in csv.DictReader(stream):
            key=(p['episode_id'],int(p['step'])); left=int(p['left_index']); right=int(p['right_index'])
            kind,left_wins=label(returns[key+(left,)],returns[key+(right,)])
            if kind!=p['relation']:raise ValueError('Pair label differs')
            if kind=='strict':
                winner,loser=(left,right) if left_wins else (right,left)
                if winner!=int(p['winner_index']):raise ValueError('Winner identity differs')
                graph=graphs[key]; options={o['index']:o for o in graph['options']}
                if str(score(graph,options[winner])>score(graph,options[loser]))!=p['frozen_correct']:raise ValueError('Frozen accuracy differs')
                if str(options[winner]['logit']>options[loser]['logit'])!=p['logit_correct']:raise ValueError('Native logit accuracy differs')
            pair_checks+=1
    for state in json.loads(sp.read_text()):
        key=(state['episode_id'],state['step']); graph=graphs[key]; native=graph['effective_index']
        moves=[o for o in graph['options'] if o['admissible'] and o['action']['act']==4 and o['index']>0 and graph['mask'][o['index']] and not graph['visited'][o['index']]]
        chosen=native
        if native>0 and not graph['budget_stop'] and not graph['no_vp_left'] and len(moves)>=2:
            chosen=sorted(moves,key=lambda o:(-score(graph,o),o['index']))[0]['index']
        if chosen!=state['chosen_index'] or native!=state['native_index']:raise ValueError('Masked choice differs')
        if not state['supported']:raise ValueError('Unexpected missing return in complete archive')
        for field,index in [('native_metrics',native),('chosen_metrics',chosen)]:
            for metric in METRICS:
                if state[field][metric]!=returns[key+(index,)][metric]:raise ValueError('Archived return differs')
                return_checks+=1
        choice_checks+=1
    output=args.root/'verification_001'; output.mkdir(exist_ok=False)
    result={'all_checks_passed':True,'pair_label_checks':pair_checks,'masked_choice_checks':choice_checks,'return_component_checks':return_checks,
            'independence':'Independent vector inequality labels and masked sorting; shared frozen feature extraction. No separate simulator reproduction.',
            'hashes':{str(p):digest(p) for p in [args.config,Path(__file__),pp,sp,gp,Path(cfg['training_returns']),Path(cfg['frozen_model'])]}}
    with (output/'summary.json').open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
