#!/usr/bin/env python3
"""Locate oracle arrival within previously recorded controller execution."""
import argparse
import gzip
import json
from pathlib import Path
import habitat_sim
from analyze_equal_primitive_budget import path_distance
from audit_single_intervention import one, read_rows, group_rows
from run_option_capture import digest


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);args=p.parse_args()
    cfg=json.loads(args.config.read_text());source=json.loads(Path(cfg['source_result']).read_text())
    cohorts=json.loads(Path(cfg['source_config']).read_text())['cohorts']
    output=Path(cfg['output_dir']);output.mkdir(parents=True,exist_ok=False)
    paths=[args.config,Path(__file__),Path(cfg['source_config']),Path(cfg['source_result']),Path('research/tools/analyze_equal_primitive_budget.py')]
    jobs=[];checks=[];finders={}
    for c in cohorts:
        cresult=source['cohorts'][c['name']]
        if not cresult['all_required_gates_passed']:raise ValueError('STOP audit incomplete')
        chosen=[ep for ep,r in cresult['routes'].items() if not r['success'] and r['oracle_success']]
        if not chosen:continue
        root=Path(c['root']);manifest_path=root/'capture_001/manifest.json'
        manifest=json.loads(manifest_path.read_text());dataset=Path(manifest['config']['dataset_path'])
        with gzip.open(str(dataset),'rt') as f: episodes={str(e['episode_id']):e for e in json.load(f)['episodes']}
        trace_path=one(root/'capture_001/traces','*.jsonl');traces=group_rows(read_rows(trace_path))
        paths.extend([manifest_path,dataset,trace_path])
        for ep in chosen:
            rr=traces[ep];scene=rr[0]['scene_id'];mesh=Path(scene).with_suffix('.navmesh')
            if scene not in finders:
                finder=habitat_sim.PathFinder()
                if not finder.load_nav_mesh(str(mesh)):raise ValueError('Failed navmesh load')
                finders[scene]=finder;paths.append(mesh)
            finder=finders[scene];goal=episodes[ep]['goals'][0]['position']
            for r in rr:
                for pose,field in [('pre_pose','distance_before'),('post_pose','distance_after')]:
                    error=abs(path_distance(finder,r[pose]['position'],goal)-r[field])
                    checks.append({'cohort':c['name'],'episode_id':ep,'step':r['high_level_step'],'pose':pose,'absolute_error_m':error})
            jobs.append((c['name'],ep,rr,finder,goal,cresult['routes'][ep]))
    maximum=max(x['absolute_error_m'] for x in checks)
    calibration={'checks':checks,'comparisons':len(checks),'max_error_m':maximum,'passed':maximum<=cfg['geodesic_tolerance_m']}
    (output/'calibration.json').write_text(json.dumps(calibration,indent=2)+'\n')
    if not calibration['passed']:raise ValueError('Geodesic calibration failed; no interior interpretation')
    results=[]
    for cohort,ep,rr,finder,goal,route in jobs:
        primitives=[];count=0
        for r in rr:
            for i,primitive in enumerate(r['primitives']):
                count+=1;distance=path_distance(finder,primitive['pose']['position'],goal)
                primitives.append({'high_level_step':r['high_level_step'],'within_option_primitive':i+1,'episode_primitive':count,
                    'primitive_action':primitive['action'],'macro_action':r['action']['act'],'distance_m':distance,
                    'is_option_end':i==len(r['primitives'])-1,'within_3m':distance<=cfg['success_distance_m']})
        arrivals=[r for r in primitives if r['within_3m']]
        if not arrivals:raise ValueError('Native oracle_success cannot be reconstructed')
        results.append({'cohort':cohort,'episode_id':ep,'scene_id':rr[0]['scene_id'],
            'native_oracle_success_reconstructed':True,'first_arrival':arrivals[0],
            'minimum_distance_m':min(r['distance_m'] for r in primitives),
            'arrival_primitive_count':len(arrivals),'route_audit':route,'primitives':primitives})
    result={'experiment_id':cfg['experiment_id'],'status':'complete','post_hoc':True,'no_new_rollout':True,'no_model_trained':True,
        'calibration_comparisons':len(checks),'max_calibration_error_m':maximum,'episodes':results,
        'source_hashes':{str(p):digest(p) for p in paths},'limits':cfg['limits']}
    (output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps([{k:v for k,v in r.items() if k not in ['primitives','route_audit']} for r in results],indent=2))


if __name__=='__main__':main()
