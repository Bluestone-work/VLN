#!/usr/bin/env python3
"""All-route paired metrics and scene-cluster intervals for frozen interventions."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

METRICS = ['success','spl','ndtw','sdtw','distance_to_goal','path_length','primitive_action_count','high_level_steps','collision_events']


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,required=True); ap.add_argument('--cycle',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); a=ap.parse_args()
    if json.loads((a.cycle/'status.json').read_text())['status']!='complete': raise ValueError('Incomplete experiment')
    cfg=json.loads((a.root/'capture_001/manifest.json').read_text())['config']
    sample=json.loads(Path(cfg['sampling_manifest']).read_text())
    mapping={str(e['episode_id']):e['scene_id'] for e in sample['episodes']}
    scenes=sorted(set(mapping.values())); rng=np.random.RandomState(20261012)
    bootstrap=rng.randint(0,len(scenes),size=(5000,len(scenes)))
    arms={}; files=[]; route_maps={}
    for arm in ['learned']+['random_'+str(s) for s in [20261021,20261022,20261023]]:
        audit=a.cycle/(arm+'_enabled_audit'); gate=json.loads((audit/'summary.json').read_text())
        met=json.loads((a.cycle/(arm+'_enabled_metrics')/'summary.json').read_text())
        if not gate['all_required_gates_passed'] or not met['all_required_gates_passed']: raise ValueError('Failed fidelity gate')
        path=audit/'paired_episodes.json'; files.append(path)
        rows=json.loads(path.read_text()); by={r['episode_id']:r for r in rows}
        if set(by)!=set(mapping): raise ValueError('Route population mismatch')
        route_maps[arm]=by
        metrics={}
        for metric in METRICS:
            scene_means=np.array([np.mean([r['delta'][metric] for r in rows if mapping[r['episode_id']]==s]) for s in scenes])
            intervals=np.percentile(scene_means[bootstrap].mean(axis=1),[2.5,97.5]).tolist()
            metrics[metric]={'baseline':float(np.mean([r['baseline'][metric] for r in rows])), 'intervention':float(np.mean([r['intervention'][metric] for r in rows])), 'delta':float(np.mean([r['delta'][metric] for r in rows])), 'scene_cluster_ci95':intervals}
        quality=[r['episode_id'] for r in rows if r['delta']['success']>0 and r['delta']['ndtw']>=-1e-6]
        arms[arm]={'episodes':len(rows),'interventions':gate['interventions'],'metrics':metrics,
                   'success_rescued':[r['episode_id'] for r in rows if r['delta']['success']>0],
                   'success_lost':[r['episode_id'] for r in rows if r['delta']['success']<0],
                   'quality_rescued':quality,'quality_rescue_scenes':sorted({mapping[e] for e in quality}),
                   'treated_metrics':next(g for name,g in gate['paired_groups'].items() if name.startswith('targeted_')),
                   'audit_passed':True}
    learned=arms['learned']; mm=learned['metrics']; random_names=[k for k in arms if k.startswith('random')]
    comparisons={}
    for metric in METRICS:
        diff={ep:route_maps['learned'][ep]['intervention'][metric]-np.mean([route_maps[k][ep]['intervention'][metric] for k in random_names]) for ep in mapping}
        sm=np.array([np.mean([v for ep,v in diff.items() if mapping[ep]==s]) for s in scenes])
        comparisons[metric]={'delta_vs_mean_random':float(np.mean(list(diff.values()))),'scene_cluster_ci95':np.percentile(sm[bootstrap].mean(axis=1),[2.5,97.5]).tolist()}
    gates={'quality_rescue_in_two_scenes':len(learned['quality_rescue_scenes'])>=2,'positive_sr':mm['success']['delta']>0,'nondecreasing_spl':mm['spl']['delta']>=-1e-6,'nondecreasing_ndtw':mm['ndtw']['delta']>=-1e-6,'no_primitive_increase':mm['primitive_action_count']['delta']<=0,'sr_beats_mean_random':comparisons['success']['delta_vs_mean_random']>0}
    out={'experiment':'GRAPH-VALUE-ROUTE-EXECUTION-002','arms':arms,'learned_vs_mean_random':comparisons,'gates':gates,'decision':'GO for a new prospective study' if all(gates.values()) else 'NO-GO for scaling this linear first-disagreement recipe','bootstrap_seed':20261012,'bootstrap_samples':5000,'scene_clusters':len(scenes),'limits':['Exploratory target follow-up after census inspection; not a pristine prospective confirmation.','Route-disjoint relative to listed diagnostic sources, scene overlap allowed; not baseline training holdout.','One simulator seed; random seeds are control choices, not training replications.','No claim that failure of this ranker rejects all graph value methods or proves proposal failure.'],'hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files+[Path(__file__)]}}
    a.output.mkdir(parents=True,exist_ok=False); (a.output/'summary.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({'decision':out['decision'],'gates':gates,'arms':{k:{'interventions':v['interventions'],'rescued':v['success_rescued'],'lost':v['success_lost'],'deltas':{m:v['metrics'][m]['delta'] for m in METRICS}} for k,v in arms.items()}},indent=2))

if __name__=='__main__': main()
