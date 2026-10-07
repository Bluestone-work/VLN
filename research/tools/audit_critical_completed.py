#!/usr/bin/env python3
"""Independent prefix/cut coverage audit of the completed critical-state run."""
import argparse,gzip,json
from pathlib import Path
from run_option_capture import digest

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 m=json.loads((a.run/'manifest.json').read_text());root=Path(m['plan']['config']['source_root']);rows=[json.loads(x) for x in (a.run/'results.jsonl').open()];base={};branches={}
 fail=set(m['plan']['failed_episodes'])
 for line in (root/'capture_001/traces/worker_seed100.jsonl').open():
  r=json.loads(line)
  if str(r['episode_id']) in fail:base[(str(r['episode_id']),r['high_level_step'])]=r
 needed={(c['episode_id'],c['high_level_step'],c['action_index']) for c in m['plan']['cases']}
 for line in (root/m['plan']['config']['probe']/'branch_traces.jsonl').open():
  r=json.loads(line);k=(str(r['decision_key'][1]),r['decision_key'][2],r['index'])
  if r['order']=='forward' and k in needed:branches[k]=r['trace']
 # Filtered dataset is exactly the original failed episode records, same vocab.
 source=json.load(gzip.open(m['source']['config']['dataset_path'],'rt'));subset=json.load(gzip.open(m['overrides']['TASK_CONFIG.DATASET.DATA_PATH'],'rt'))
 original={str(e['episode_id']):e for e in source['episodes']};sub={str(e['episode_id']):e for e in subset['episodes']}
 assert set(sub)==fail and all(sub[k]==original[k] for k in sub)
 assert {k:v for k,v in source.items() if k!='episodes'}=={k:v for k,v in subset.items() if k!='episodes'}
 count=primitives=cuts=0;sensing={};cut_rows=[]
 for result in rows:
  c=result['case'];ep=c['episode_id'];actual=[json.loads(x) for x in (a.run/'cases'/c['case_id']/'trace.jsonl').open()];step=c.get('high_level_step',len(actual))
  neutral=c['mode']=='action' and c['action_index']==c['effective_index']
  for i,r in enumerate(actual):
   expected=None
   if c['mode'] in ['control','sense_only'] or neutral or i<step:expected=base[(ep,i)]
   elif c['mode']=='action' and i==step:expected=branches[(ep,i,c['action_index'])]
   if expected:
    for name in ['action','pre_pose','pre_rng','post_pose','post_rng','primitives','observation_hashes','distance_before','distance_after','done']:
     assert r[name]==expected[name],(c['case_id'],i,name)
    count+=1;primitives+=len(r['primitives'])
  if 'cut_primitive' in c:
   r=actual[step];original=base[(ep,step)];cut=c['cut_primitive'];assert 0<cut<len(original['primitives'])
   assert r['phases'][cut-1]=='ghost'
   key=(ep,step,cut)
   if c['mode']=='sense_only':sensing[key]=r['cut_sensor_hashes']
   else:
    assert r['cut_sensor_hashes']==sensing[key]
    assert len(r['primitives'])==cut and r['primitives']==original['primitives'][:cut]
    assert r['pre_metrics']==original['pre_metrics']
    assert r['post_metrics']['steps_taken']-r['pre_metrics']['steps_taken']==cut
    before=(r['pre_metrics'].get('collisions') or {}).get('count',0);after=(r['post_metrics'].get('collisions') or {}).get('count',0)
    assert after-before==sum(p['collided'] is True for p in r['primitives'])
    cuts+=1
   cut_rows.append({'case_id':c['case_id'],'cut':cut,'sensors_match':True,'partial_cost_retained':True})
 a.output.mkdir(parents=True,exist_ok=False)
 out={'all_checks_passed':True,'subset_episode_and_vocab_identity':True,'exact_full_option_or_prefix_checks':count,'exact_primitive_records':primitives,'interrupt_prefix_and_cost_checks':cuts,'cut_rows':cut_rows,'result_sha256':digest(a.run/'results.jsonl'),'script_sha256':digest(__file__)}
 (a.output/'summary.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='cut_rows'},indent=2))

if __name__=='__main__':main()
