#!/usr/bin/env python3
"""Freeze one exact single-action schedule per alternative rank.

The input is the already validated 44-episode native graph capture.  Each
episode contributes at most one event to a run, so every event has an intact
native prefix and an unambiguous native continuation after its alternative.
No outcome, goal, reference path, or semantic endpoint is read here.
"""
import argparse, collections, hashlib, json
from pathlib import Path

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1<<20), b''): h.update(block)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--source-capture',type=Path,required=True); ap.add_argument('--output-dir',type=Path,required=True); a=ap.parse_args()
    path=a.source_capture/'graph_options/graph_options.jsonl'
    rows=[json.loads(x) for x in path.open() if x.strip()]
    by=collections.defaultdict(list)
    for r in rows:
        for o in r['options']:
            if o.get('admissible') and int(o['index']) != int(r['effective_index']):
                by[str(r['episode_id'])].append((int(r['high_level_step']),int(o['index']),r,o))
    for e in by: by[e].sort(key=lambda x:(x[0],x[1]))
    episodes=sorted(by); max_rank=max(len(x) for x in by.values())
    a.output_dir.mkdir(parents=True,exist_ok=False)
    schedules=[]
    for rank in range(max_rank):
        events=[]
        for episode in episodes:
            if rank >= len(by[episode]): continue
            step,index,r,o=by[episode][rank]
            native=next(x for x in r['options'] if int(x['index'])==int(r['effective_index']))
            events.append({'scene_id':r['scene_id'],'episode_id':str(r['episode_id']),
                'high_level_step':step,'trajectory_id':str(r['trajectory_id']),
                'baseline_index':int(r['effective_index']),'alternative_index':index,
                'target_index':index,'target_is_native':False,
                'expected_graph_ids':r['graph_ids'],'expected_graph_position':r['graph_position'],
                'baseline_action':native['action'],'alternative_action':o['action'],
                'target_action':o['action'],'target_graph_id':o.get('graph_id')})
        schedule={'experiment_id':'NATIVE-SINGLE-ACTION-FULL-RETURN-44-RANK{:03d}'.format(rank),
            'status':'frozen_before_outcomes','analysis_only_oracle':True,
            'expected_event_count':len(events),'source_capture':str(a.source_capture),
            'events':events,
            'label_rule':'success, then nDTW, then SDTW, then path/primitive/collision cost; fixed deadband 1e-6/1e-5; no tuning',
            'hashes':{'source_graph':digest(path)}}
        p=a.output_dir/'schedule_rank{:03d}.json'.format(rank); p.write_text(json.dumps(schedule,indent=2)+'\n'); schedules.append(p)
    summary={'source_capture':str(a.source_capture),'source_graph_sha256':digest(path),
        'episodes':len(episodes),'states':len(rows),'scenes':len(set(r['scene_id'] for r in rows)),
        'admissible_alternatives':sum(len(v) for v in by.values()),'schedules':len(schedules),
        'alternatives_per_episode':{e:len(by[e]) for e in episodes}}
    (a.output_dir/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
