#!/usr/bin/env python3
"""Exploratory equal-primitive windows, after exact native geodesic calibration."""
import argparse
import gzip
import json
import math
from pathlib import Path

import habitat_sim
import numpy as np

from audit_single_intervention import read_rows, group_rows, one
from analyze_continuation_horizon import window
from continuation_labels import confirmation
from run_option_capture import digest


def path_distance(finder, point, goal):
    path=habitat_sim.MultiGoalShortestPath()
    path.requested_start=np.asarray(point,dtype=np.float32)
    path.requested_ends=np.asarray([goal],dtype=np.float32)
    if not finder.find_path(path) or not math.isfinite(path.geodesic_distance):
        raise ValueError('No finite geodesic for a retained actual pose')
    return float(path.geodesic_distance)


def budget_window(records,step,budget,finder,goal):
    if budget<1:raise ValueError('Nonpositive primitive budget')
    positions=[records[step]['pre_pose']['position']]
    primitive_count=0;full_decisions=0;terminal=False
    for row in records[step:]:
        consumed=0
        for primitive in row['primitives']:
            if primitive_count==budget:break
            positions.append(primitive['pose']['position'])
            primitive_count+=1;consumed+=1
        if consumed==len(row['primitives']):
            full_decisions+=1
            terminal=bool(row['done'])
        if primitive_count==budget or terminal:break
    array=np.asarray(positions,dtype=np.float64)
    return {'distance_to_goal':path_distance(finder,positions[-1],goal),
            'primitive_count':primitive_count,
            'movement_m':float(np.linalg.norm(np.diff(array,axis=0),axis=1).sum()),
            'fully_completed_high_level_decisions':full_decisions,
            'terminal':terminal,'last_pose':positions[-1]}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args()
    cfg=json.loads(args.config.read_text())
    args.output_dir.mkdir(parents=True,exist_ok=False)
    hashes={str(args.config):digest(args.config),str(Path(__file__)):digest(Path(__file__))}
    jobs=[];checks=[];finders={}
    for batch in cfg['batches']:
        root=Path(batch['root']);run=root/'enabled_001';audit=root/'enabled_audit_001'
        if not json.loads((audit/'summary.json').read_text())['all_required_gates_passed']:
            raise ValueError('Actual continuation fidelity missing')
        m=json.loads((run/'manifest.json').read_text())
        before_path=one(Path(m['source_capture'])/'traces','*.jsonl')
        after_path=one(run/'traces','*.jsonl')
        before,after=group_rows(read_rows(before_path)),group_rows(read_rows(after_path))
        dataset_path=Path(m['config']['dataset_path'])
        with gzip.open(str(dataset_path),'rt') as f:episodes={str(e['episode_id']):e for e in json.load(f)['episodes']}
        pairs={p['episode_id']:p for p in json.loads((audit/'paired_episodes.json').read_text())}
        for source in [run/'manifest.json',audit/'summary.json',audit/'paired_episodes.json',dataset_path,before_path,after_path]:
            hashes[str(source)]=digest(source)
        for event in m['schedule']['events']:
            ep=event['episode_id'];scene=event['scene_id']
            navmesh=Path(scene).with_suffix('.navmesh')
            if scene not in finders:
                finder=habitat_sim.PathFinder()
                if not finder.load_nav_mesh(str(navmesh)):raise ValueError('Cannot load recorded scene navmesh')
                finders[scene]=finder;hashes[str(navmesh)]=digest(navmesh)
            finder=finders[scene];goal=episodes[ep]['goals'][0]['position']
            for label,rows in [('baseline',before[ep]),('intervention',after[ep])]:
                for row in rows:
                    for pose,field in [('pre_pose','distance_before'),('post_pose','distance_after')]:
                        measured=path_distance(finder,row[pose]['position'],goal)
                        delta=abs(measured-row[field])
                        checks.append({'batch':batch['name'],'episode_id':ep,'run':label,
                                       'step':row['high_level_step'],'endpoint':pose,'absolute_error_m':delta,
                                       'passed':delta<=cfg['geodesic_tolerance_m']})
            jobs.append((batch,event,before[ep],after[ep],finder,goal,pairs[ep]))
    calibration={'passed':all(c['passed'] for c in checks),'endpoint_comparisons':len(checks),
                 'max_absolute_error_m':max(c['absolute_error_m'] for c in checks),'checks':checks}
    (args.output_dir/'calibration.json').write_text(json.dumps(calibration,indent=2)+'\n')
    if not calibration['passed']:raise ValueError('Native geodesic calibration failed; do not interpret windows')
    events=[]
    for batch,event,old,new,finder,goal,pair in jobs:
        step=event['high_level_step']
        native_a=window(old,step,cfg['baseline_high_level_horizon'])
        native_b=window(new,step,cfg['baseline_high_level_horizon'])
        budget=native_a['primitive_count']
        a=budget_window(old,step,budget,finder,goal)
        b=budget_window(new,step,budget,finder,goal)
        if (abs(a['distance_to_goal']-native_a['distance_to_goal'])>cfg['geodesic_tolerance_m']
                or abs(a['movement_m']-native_a['movement_m'])>cfg['geodesic_tolerance_m']):
            raise ValueError('Equal-budget baseline must reproduce its H=2 boundary')
        events.append({'batch':batch['name'],'episode_id':event['episode_id'],'step':step,
                       'primitive_budget':budget,'baseline':a,'intervention':b,
                       'equal_primitive_rule':confirmation(a,b,cfg['rule']),
                       'original_h2_rule':confirmation(native_a,native_b,cfg['rule']),
                       'final_delta':pair['delta']})
    batches={}
    for batch in cfg['batches']:
        rows=[e for e in events if e['batch']==batch['name']]
        changed=[e['episode_id'] for e in rows if e['equal_primitive_rule']['accepted']!=e['original_h2_rule']['accepted']]
        accepted=[e for e in rows if e['equal_primitive_rule']['accepted']]
        batches[batch['name']]={'events':len(rows),'changed_acceptance_episode_ids':changed,
            'accepted_episode_ids':[e['episode_id'] for e in accepted],
            'accepted_final_primitive_delta':sum(e['final_delta']['primitive_action_count'] for e in accepted),
            'accepted_ndtw_worse':sum(e['final_delta']['ndtw']<-1e-6 for e in accepted),
            'accepted_success_lost':sum(e['final_delta']['success']<0 for e in accepted)}
    result={'experiment_id':cfg['experiment_id'],'analysis_type':cfg['analysis_type'],
            'native_geodesic_calibration_passed':True,'calibration_endpoint_comparisons':len(checks),
            'max_geodesic_error_m':calibration['max_absolute_error_m'],'events':events,'batches':batches,
            'hashes':hashes,'no_model_trained':True,
            'limits':'Post-hoc, privileged offline read of actual primitive poses. Baseline H=2 determines a different budget per event. Equal primitive count is not equal wall-clock time. STOP is absorbing; partial options are observations only, not a new executable action. No new rollout or independent validation.'}
    (args.output_dir/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'calibration_endpoints':len(checks),'max_error_m':calibration['max_absolute_error_m'],'batches':batches},indent=2))


if __name__=='__main__':main()
