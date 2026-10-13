#!/usr/bin/env python3
"""Build and execute exact full-return pairs for the 10-state endpoint cohort."""
import argparse
import collections
import json
import subprocess
import sys
from pathlib import Path


def digest(path):
    import hashlib
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()


def key(row):
    return (str(row['scene_id']), str(row['episode_id']), int(row['high_level_step']))


def metrics_path(exp):
    paths = list(Path('data/logs/eval_results', exp).glob('stats_ep_*.json'))
    if len(paths) != 1: raise ValueError('Expected one episode metrics file for {}'.format(exp))
    return paths[0]


def trace_path(run):
    paths = list((run / 'traces').glob('worker_seed*.jsonl'))
    if len(paths) != 1: raise ValueError('Expected one trace file in {}'.format(run))
    return paths[0]


def read_jsonl(path):
    return [json.loads(x) for x in Path(path).open() if x.strip()]


def build_schedules(source, endpoint, out):
    graphs = {key(r): r for r in read_jsonl(source / 'graph_options/graph_options.jsonl')}
    endpoints = read_jsonl(endpoint / 'native_endpoint_options.jsonl')
    states = collections.defaultdict(list)
    for row in endpoints:
        k = (str(row['episode_id']), int(row['high_level_step']))
        graph = next((g for g in graphs.values() if str(g['episode_id']) == k[0] and int(g['high_level_step']) == k[1]), None)
        if graph is None: raise ValueError('Endpoint state absent from source graph: {}'.format(k))
        option = next(o for o in graph['options'] if o['index'] == row['index'])
        if option['action'] != row['replay_trace']['action']:
            raise ValueError('Endpoint action does not match graph option: {}'.format(k))
        row = dict(row, scene_id=graph['scene_id'], trajectory_id=graph['trajectory_id'],
                   graph_position=graph['graph_position'], graph_ids=graph['graph_ids'],
                   native_index=graph['effective_index'], native_action=next(o for o in graph['options'] if o['index'] == graph['effective_index'])['action'])
        states[k].append(row)
    targets_by_episode = collections.defaultdict(list)
    for (episode, step), rows in sorted(states.items()):
        rows = sorted(rows, key=lambda r: r['index'])
        native = next(r for r in rows if r['selected_native'])
        targets_by_episode[episode].extend(r for r in rows if not r['selected_native'])
    schedules = []
    out.mkdir(parents=True, exist_ok=False)
    max_len=max(len(v) for v in targets_by_episode.values())
    episodes=sorted(targets_by_episode)
    for rank in range(max_len):
            events=[]
            for episode in episodes:
                targets=targets_by_episode[episode]
                if rank >= len(targets): continue
                target=targets[rank]
                graph=graphs[(str(target['scene_id']),str(episode),int(target['high_level_step']))]
                native=next(r for r in states[(episode,int(target['high_level_step']))] if r['selected_native'])
                events.append({'scene_id':target['scene_id'],'episode_id':episode,'high_level_step':target['high_level_step'],
                               'trajectory_id':target['trajectory_id'],'baseline_index':native['index'],
                               'alternative_index':target['index'],'target_index':target['index'],
                               'target_is_native':False,'expected_graph_ids':target['graph_ids'],
                               'expected_graph_position':target['graph_position'],
                               'baseline_action':native['replay_trace']['action'],
                               'alternative_action':target['replay_trace']['action'],
                               'target_action':target['replay_trace']['action'],
                               'target_graph_id':target.get('graph_id'),
                               'endpoint_row_hash':digest(endpoint/'native_endpoint_options.jsonl')})
            if not events: continue
            schedule={'experiment_id':'NATIVE-SINGLE-ACTION-FULL-RETURN-001-RANK{}'.format(rank),
                      'status':'frozen_before_outcomes','analysis_only_oracle':True,
                      'expected_event_count':len(events),'source_capture':str(source),
                      'endpoint_enum':str(endpoint),'events':events,
                      'label_rule':'success, then nDTW, then SDTW, then path/primitive/collision cost; fixed deadband 1e-6/1e-5; no tuning',
                      'hashes':{'source_graph':digest(source/'graph_options/graph_options.jsonl'),
                                'endpoint_options':digest(endpoint/'native_endpoint_options.jsonl')}}
            path=out/'schedule_rank{:02d}.json'.format(rank);path.write_text(json.dumps(schedule,indent=2)+'\n');schedules.append(path)
    return schedules, states


def compare_trace_prefix(source_rows, run_rows, event):
    src={(str(r['scene_id']),str(r['episode_id']),int(r['high_level_step'])):r for r in source_rows}
    cur={(str(r['scene_id']),str(r['episode_id']),int(r['high_level_step'])):r for r in run_rows}
    target=(event['scene_id'],str(event['episode_id']),int(event['high_level_step']))
    for k, row in src.items():
        if k not in cur: raise ValueError('Missing trace key {}'.format(k))
        if k[1] not in {str(e['episode_id']) for e in [event]} or k[2] < event['high_level_step']:
            if row != cur[k]: raise ValueError('Prefix trace differs {}'.format(k))
    return src[target], cur[target]


def labels(native, alternative):
    eps=1e-6
    if native['success'] != alternative['success']:
        return 'BENEFIT' if alternative['success'] > native['success'] else 'HARM'
    for metric in ['ndtw','sdtw']:
        delta=alternative[metric]-native[metric]
        if abs(delta)>eps: return 'BENEFIT' if delta>0 else 'HARM'
    # Fixed cost tie-breakers. Integer primitives/collisions are exact.
    for metric in ['path_length','primitive_action_count','collision_events']:
        delta=alternative[metric]-native[metric]
        if metric == 'path_length' and abs(delta)<=1e-5: continue
        if delta != 0: return 'BENEFIT' if delta<0 else 'HARM'
    return 'UNCERTAIN'


def add_cost(metrics, trace_rows):
    rows=sorted(trace_rows,key=lambda r:int(r['high_level_step']))
    out=dict(metrics)
    out['primitive_action_count']=sum(len(r['primitives']) for r in rows)
    out['collision_events']=sum(p.get('collided') is True for r in rows for p in r['primitives'])
    out['goal_distance']=float(rows[-1]['distance_after'])
    return out


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source-capture',type=Path,required=True);ap.add_argument('--endpoint-enum',type=Path,required=True);ap.add_argument('--output-root',type=Path,required=True);ap.add_argument('--execute',action='store_true');args=ap.parse_args()
    source=args.source_capture; endpoint=args.endpoint_enum; root=args.output_root;root.mkdir(parents=True,exist_ok=False)
    schedules, states=build_schedules(source,endpoint,root/'schedules')
    if args.execute:
        for i,schedule in enumerate(schedules):
            run=root/'runs'/'rank{:02d}_enabled'.format(i);run.parent.mkdir(parents=True,exist_ok=True)
            cmd=[sys.executable,'research/tools/run_native_full_return.py','--schedule',str(schedule),
                 '--source-capture',str(source),'--output-dir',str(run),
                 '--exp-name','native_full_return_001_rank{:02d}_enabled'.format(i),'--enabled']
            log=root/'logs';log.mkdir(exist_ok=True)
            code=subprocess.call(cmd,stdout=(log/('rank{:02d}.stdout'.format(i))).open('w'),
                                 stderr=(log/('rank{:02d}.stderr'.format(i))).open('w'))
            if code: raise RuntimeError('full-return run failed: {}'.format(cmd))
    print(json.dumps({'schedules':len(schedules),'states':len(states),'actions':sum(len(v) for v in states.values()),'execute':args.execute},indent=2))


if __name__=='__main__':main()
