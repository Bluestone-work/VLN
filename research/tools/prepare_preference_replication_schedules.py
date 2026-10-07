#!/usr/bin/env python3
"""Generate all preregistered schedules from current graph features only."""
import argparse
import collections
import hashlib
import json
import random
from pathlib import Path
import numpy as np
from prepare_ranker_intervention_schedule import admissible_moves, first_disagreement, digest
from graph_value_feasibility import action_features


def cost_at_full_state(full_choice, model):
    if full_choice is None: return None
    row, native, _ = full_choice
    weights=np.asarray(model['weights_standardized']); mean=np.asarray(model['mean']); scale=np.asarray(model['scale'])
    best=max(admissible_moves(row),key=lambda o:(float(weights.dot((action_features(row,o)-mean)/scale)),-o['index']))
    return row, native, best


def random_at_full_state(full_choice, seed):
    if full_choice is None: return None
    row,native,_=full_choice
    salt='{}|{}|{}|{}'.format(seed,row['scene_id'],row['episode_id'],row['high_level_step'])
    rng=random.Random(int(hashlib.sha256(salt.encode()).hexdigest(),16))
    best=rng.choice(sorted(admissible_moves(row),key=lambda o:o['index']))
    return row,native,best


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);a=ap.parse_args()
    cfg=json.loads(a.config.read_text());rep=cfg['replication'];root=Path(rep['root'])
    registration=root/'registration_001.json'
    reg=json.loads(registration.read_text())
    for path,sha in reg['hashes'].items():
        if digest(path)!=sha: raise ValueError('Registration changed: '+path)
    full=json.loads(Path(rep['full_model']).read_text());cost=json.loads(Path(rep['cost_model']).read_text())
    sample=json.loads(Path(cfg['sampling_manifest']).read_text())
    graph_path=root/'capture_001/graph_options/graph_options.jsonl'; graphs=collections.defaultdict(list)
    for line in graph_path.open():
        row=json.loads(line);graphs[str(row['episode_id'])].append(row)
    if set(graphs)!={str(e['episode_id']) for e in sample['episodes']}: raise ValueError('Route population mismatch')
    schedules={name:{'events':[],'omitted_routes_retained_as_controls':[],'eligible_states':[]} for name in rep['arms']}
    for ep in sample['episodes']:
        key=str(ep['episode_id']); rows=graphs[key]
        f=first_disagreement(rows,full)
        choices={'full':f,'cost_own':first_disagreement(rows,cost),'cost_matched':cost_at_full_state(f,cost)}
        choices.update({'random_'+str(seed):random_at_full_state(f,seed) for seed in rep['random_seeds']})
        for name,chosen in choices.items():
            schedule=schedules[name]
            if chosen is None:
                schedule['omitted_routes_retained_as_controls'].append({'episode_id':key,'reason':'no_disagreement'})
                continue
            row,native,best=chosen
            schedule['eligible_states'].append({'episode_id':key,'high_level_step':row['high_level_step']})
            if native['index']==best['index']:
                schedule['omitted_routes_retained_as_controls'].append({'episode_id':key,'reason':'native_selected_at_matched_state','high_level_step':row['high_level_step']})
                continue
            schedule['events'].append({'scene_id':row['scene_id'],'episode_id':key,'high_level_step':row['high_level_step'],
                'trajectory_id':str(row['trajectory_id']),'baseline_index':native['index'],'alternative_index':best['index'],
                'baseline_action':native['action'],'alternative_action':best['action'],'expected_graph_ids':row['graph_ids'],
                'expected_graph_position':row['graph_position'],'selection_kind':name,'native_logit':native['logit'],'alternative_logit':best['logit']})
    paths=[a.config,registration,Path(rep['full_model']),Path(rep['cost_model']),Path(cfg['sampling_manifest']),
           graph_path,root/'capture_001/manifest.json',Path(cfg['protocol']),Path(__file__),
           Path('research/tools/prepare_ranker_intervention_schedule.py'),Path('research/tools/graph_value_feasibility.py')]
    hashes={str(p):digest(p) for p in paths}
    directory=root/'schedules_001';directory.mkdir(exist_ok=False)
    for name,s in schedules.items():
        s.update(experiment_id=cfg['experiment_id']+'-'+name,status='frozen_before_intervention_outcomes',analysis_only_oracle=False,
                 learned_model_evaluated=not name.startswith('random'),expected_event_count=len(s['events']),
                 base_experiment_config=str(a.config),protocol=cfg['protocol'],hashes=hashes,
                 selection_rule='Prospectively registered one-disagreement rule; FULL timing for matched controls; no goal/reference/outcome inputs.',
                 limits='Prospective route-disjoint training replication, overlapping scenes, baseline trained on train; one simulator seed; offline intervention harness, not benchmark.')
        with (directory/(name+'.json')).open('x') as stream:json.dump(s,stream,indent=2);stream.write('\n')
    print(json.dumps({name:{'events':len(s['events']),'eligible':len(s['eligible_states'])} for name,s in schedules.items()},indent=2))

if __name__=='__main__':main()
