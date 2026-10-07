#!/usr/bin/env python3
"""Fit and freeze a tiny pairwise graph preference model on one cohort."""
import argparse, collections, hashlib, json
from pathlib import Path
import numpy as np
from graph_value_feasibility import action_features, dominates, fit_pairwise, FEATURE_NAMES


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--plan',type=Path,required=True); ap.add_argument('--run',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); a=ap.parse_args()
    plan=json.loads(a.plan.read_text()); root=Path(plan['config']['source_root'])
    graphs={}
    for line in (root/'capture_001/graph_options/graph_options.jsonl').open():
        row=json.loads(line); graphs[(str(row['episode_id']),int(row['high_level_step']))]=row
    by=collections.defaultdict(list)
    for line in (a.run/'results.jsonl').open():
        result=json.loads(line); c=result['case']
        if c.get('mode')!='action': continue
        key=(str(c['episode_id']),int(c['high_level_step'])); row=graphs[key]
        option=next(o for o in row['options'] if int(o['index'])==int(c['action_index']))
        metrics=dict(result['metrics']); metrics['primitive_action_count']=int(round(metrics.get('primitive_action_count',metrics['steps_taken'])))
        by[key].append({'features':action_features(row,option),'metrics':metrics,'index':int(c['action_index'])})
    items=[x for xs in by.values() for x in xs]; mean=np.mean([x['features'] for x in items],axis=0); scale=np.std([x['features'] for x in items],axis=0); scale[scale<1e-8]=1.0
    for x in items: x['z']=(x['features']-mean)/scale
    pairs=[]
    for xs in by.values():
        for i in range(len(xs)):
            for j in range(i+1,len(xs)):
                if dominates(xs[i]['metrics'],xs[j]['metrics']): pairs.append((xs[i],xs[j]))
                elif dominates(xs[j]['metrics'],xs[i]['metrics']): pairs.append((xs[j],xs[i]))
    w=fit_pairwise(pairs)
    if w is None: raise ValueError('No strict pairs in training cohort')
    result={'schema_version':1,'feature_names':list(FEATURE_NAMES),'weights_standardized':w.tolist(),'mean':mean.tolist(),'scale':scale.tolist(),'train_states':len(by),'train_actions':len(items),'strict_pairs':len(pairs),'training_plan':str(a.plan),'training_run':str(a.run),'privileged_labels_used_only_for_fit':True,'deployable_features_only':True,'hashes':{str(a.plan):hashlib.sha256(a.plan.read_bytes()).hexdigest(),str(a.run/'results.jsonl'):hashlib.sha256((a.run/'results.jsonl').read_bytes()).hexdigest(),str(Path(__file__)):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps({k:result[k] for k in ['train_states','train_actions','strict_pairs']}))
if __name__=='__main__': main()
