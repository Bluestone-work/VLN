#!/usr/bin/env python3
"""Compare one-step full-option branches with the same state's full returns."""
import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
import numpy as np
from audit_single_intervention import key
from analyze_full_returns import classify
from run_option_capture import digest


def main():
    p = argparse.ArgumentParser(); p.add_argument('--config', type=Path, required=True)
    args = p.parse_args(); cfg = json.loads(args.config.read_text())
    root, probe = Path(cfg['source_root']), Path(cfg['probe_dir'])
    analysis = json.loads(Path(cfg['analysis_path']).read_text())
    graph = {}
    for line in (root/'capture_001/graph_options/graph_options.jsonl').open():
        row = json.loads(line); graph[key(row)] = row
    branches = {}
    raw = probe/'branch_traces.jsonl'
    for line in raw.open():
        row = json.loads(line)
        if row['order'] != 'forward': continue
        branches.setdefault(tuple(row['decision_key']), {})[row['index']] = row['trace']
    events=[]
    for event in analysis['events']:
        k=(event['scene_id'],event['episode_id'],event['step'])
        if k not in branches or event['alternative_index'] not in branches[k]: raise ValueError('Missing branch')
        state=graph[k]; native=state['effective_index']; opts=branches[k]
        candidate=[]
        for idx, trace in opts.items():
            if idx == 0 or trace['action']['act'] == 0: continue
            progress=float(trace['distance_before']-trace['distance_after'])
            primitives=len(trace['primitives'])
            candidate.append({'index':idx,'progress_m':progress,'primitive_count':primitives,
                              'distance_after':float(trace['distance_after']),
                              'current_proposal':next(o['current_proposal'] for o in state['options'] if o['index']==idx),
                              'logit':next(o['logit'] for o in state['options'] if o['index']==idx)})
        if not candidate: raise ValueError('No non-STOP branches')
        native_row=next(x for x in candidate if x['index']==native)
        selected=next(x for x in candidate if x['index']==event['alternative_index'])
        sorted_progress=sorted(candidate,key=lambda x:(-x['progress_m'],x['index']))
        sorted_efficiency=sorted(candidate,key=lambda x:(-(x['progress_m']/(max(x['primitive_count'],1))),x['index']))
        event_out={'kind':event['kind'],'scene_id':event['scene_id'],'episode_id':event['episode_id'],'step':event['step'],
                   'alternative_index':event['alternative_index'],'category':event['category'],
                   'alternative_local':selected,'native_local':native_row,
                   'best_local':sorted_progress[0],'best_progress_rank':1+next(i for i,x in enumerate(sorted_progress) if x['index']==event['alternative_index']),
                   'best_efficiency_rank':1+next(i for i,x in enumerate(sorted_efficiency) if x['index']==event['alternative_index']),
                   'candidate_count':len(candidate),'extra_local_progress_m':selected['progress_m']-native_row['progress_m'],
                   'extra_local_primitive_count':selected['primitive_count']-native_row['primitive_count'],
                   'final_delta':event['delta'],'horizons':event['horizons']}
        events.append(event_out)
    def finite_mean(rows, key): return float(np.mean([r[key] for r in rows])) if rows else None
    summary={}
    for kind in ['top_logit','seeded_graph_id']:
        rows=[e for e in events if e['kind']==kind]
        summary[kind]={'events':len(rows),'mean_local_progress_delta_m':finite_mean(rows,'extra_local_progress_m'),
                       'mean_local_primitive_delta':finite_mean(rows,'extra_local_primitive_count'),
                       'alternative_local_progress_positive_final_dominated':sum(e['extra_local_progress_m']>1e-6 and e['category']=='dominated' for e in rows),
                       'alternative_local_progress_negative_final_dominates':sum(e['extra_local_progress_m']<-1e-6 and e['category']=='dominates' for e in rows),
                       'local_progress_positive':sum(e['extra_local_progress_m']>1e-6 for e in rows),
                       'local_progress_negative':sum(e['extra_local_progress_m']<-1e-6 for e in rows),
                       'mean_progress_rank':finite_mean(rows,'best_progress_rank'),
                       'mean_efficiency_rank':finite_mean(rows,'best_efficiency_rank')}
    output=Path(cfg['output_dir']); output.mkdir(parents=True,exist_ok=False)
    result={'experiment_id':cfg['experiment_id'],'status':'complete','post_hoc':True,'no_model_trained':True,
            'no_new_rollout':True,'events':events,'summary':summary,
            'limits':cfg['limits'],'source_hashes':{str(x):digest(x) for x in [args.config,Path(__file__),Path(cfg['analysis_path']),raw,root/'capture_001/graph_options/graph_options.jsonl']},
            'git_commit':subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),'python':sys.version}
    (output/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    (output/'events.jsonl').write_text('\n'.join(json.dumps(e,sort_keys=True) for e in events)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
