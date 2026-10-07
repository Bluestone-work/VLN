#!/usr/bin/env python3
"""Independent cross-arm check: equal interventions must yield equal returns."""
import argparse
import itertools
import json
from pathlib import Path


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    cfg=json.loads(a.config.read_text());root=Path(cfg['replication']['root']);cycle=root/'replication_cycle_001'
    if json.loads((cycle/'status.json').read_text())['status']!='complete':raise ValueError('Incomplete cycle')
    episodes=[str(e['episode_id']) for e in json.loads(Path(cfg['sampling_manifest']).read_text())['episodes']]
    events={};metrics={};native=json.loads((root/'noninterference_001/capture_episodes.json').read_text())
    for name in cfg['replication']['arms']:
        schedule=json.loads((root/'schedules_001'/(name+'.json')).read_text());events[name]={e['episode_id']:e for e in schedule['events']}
        if schedule['events']:
            rows=json.loads((cycle/(name+'_enabled_audit')/'paired_episodes.json').read_text());metrics[name]={r['episode_id']:r['intervention'] for r in rows}
        else:metrics[name]=native
    checks=[];issues=[]
    identity=lambda e:None if e is None else (e['high_level_step'],e['alternative_action'])
    for left,right in itertools.combinations(cfg['replication']['arms'],2):
        count=0
        for ep in episodes:
            if identity(events[left].get(ep))!=identity(events[right].get(ep)):continue
            count+=1
            left_metrics=metrics[left][ep];right_metrics=metrics[right][ep]
            required={'success','spl','ndtw','primitive_action_count','path_length'}
            if set(left_metrics)!=set(right_metrics) or not required.issubset(left_metrics):
                raise ValueError('Missing or inconsistent metric keys for '+ep)
            diff=[k for k in left_metrics if left_metrics[k]!=right_metrics[k]]
            if diff:issues.append({'left':left,'right':right,'episode_id':ep,'different_metrics':diff})
        checks.append({'left':left,'right':right,'identical_intervention_routes':count})
    reg=json.loads((root/'registration_001.json').read_text());cap=json.loads((root/'cycle_001/status.json').read_text())
    if not reg['registered_utc']<cap['created_utc']:raise ValueError('Registration not before capture')
    out={'all_checks_passed':not issues,'comparisons':checks,'route_comparisons':sum(c['identical_intervention_routes'] for c in checks),'issues':issues,'registration_precedes_target_capture':True}
    a.output.mkdir(exist_ok=False)
    with (a.output/'summary.json').open('x') as stream:json.dump(out,stream,indent=2);stream.write('\n')
    print(json.dumps(out,indent=2))
    if issues:raise ValueError('Cross-arm equality failed')

if __name__=='__main__':main()
