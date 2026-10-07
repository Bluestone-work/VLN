#!/usr/bin/env python3
"""Advance through registered continuation gates; no fitting or user prompts."""
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from run_option_capture import digest


def now():return datetime.now(timezone.utc).isoformat()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--route-config',type=Path,required=True)
    p.add_argument('--root',type=Path,required=True)
    args=p.parse_args()
    cfg=json.loads(args.config.read_text())
    foundation=json.loads((args.root/'cycle_001/status.json').read_text())
    if foundation.get('status')!='complete':raise ValueError('Foundation cycle not complete')
    directory=args.root/'continuation_cycle_001'
    directory.mkdir(parents=True,exist_ok=False)
    files=[args.config,args.route_config,Path(cfg['continuation_protocol']),Path(__file__),
           Path('research/tools/run_single_intervention.py'),Path('research/tools/audit_single_intervention.py'),
           Path('research/tools/analyze_prospective_continuation.py'),Path('research/tools/continuation_labels.py'),
           Path('research/tools/prepare_continuation_schedule.py'),Path('research/tools/verify_intervention_metrics.py'),
           Path('vlnce_baselines/adaptive_action/single_intervention.py')]
    manifest={'created_utc':now(),'config':cfg,'stages':[],'status':'running',
              'hashes':{str(f):digest(f) for f in files},'argv':sys.argv,
              'git_commit':subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),
              'environment':{k:os.environ.get(k) for k in ['EGL_PLATFORM','CUDA_VISIBLE_DEVICES']}}
    for f in files:
        dest=directory/'sources'/f;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(f.read_bytes())
    def save():(directory/'status.json').write_text(json.dumps(manifest,indent=2)+'\n')
    def stage(name,script,arguments):
        command=[sys.executable,'research/tools/'+script]+[str(x) for x in arguments]
        record={'name':name,'command':command,'started_utc':now(),'status':'running'}
        manifest['stages'].append(record);save();print('START '+name,flush=True)
        with (directory/(name+'.stdout.log')).open('xb') as out,(directory/(name+'.stderr.log')).open('xb') as err:
            code=subprocess.call(command,stdout=out,stderr=err)
        record.update(status='passed' if code==0 else 'failed',returncode=code,finished_utc=now());save()
        if code:raise RuntimeError('Failed stage '+name)
        print('PASS '+name,flush=True)
    source=args.root/'capture_001';control=args.root/'noninterference_001';probe=args.root/'probe_full001'
    stage('route','analyze_route_options.py',['--config',args.route_config,'--probe-dir',probe,
        '--integrity-dir',args.root/'integrity_001','--noninterference-dir',control,'--output-dir',args.root/'route_001'])
    schedule_path=args.root/'schedule_h1_001.json'
    stage('schedule','prepare_continuation_schedule.py',['--config',args.config,'--root',args.root,'--output',schedule_path])
    schedule=json.loads(schedule_path.read_text())
    if not schedule['events']:
        manifest.update(status='complete_no_eligible_events',finished_utc=now(),no_model_trained=True);save()
        print('COMPLETE: no eligible events; no rule tuning or fitting',flush=True);return
    def run_pair(label,schedule_file):
        dirs={m:args.root/(label+m+'_001') for m in ['disabled','enabled']}
        audits={m:args.root/(label+m+'_audit_001') for m in ['disabled','enabled']}
        for mode in ['disabled','enabled']:
            options=['--schedule',schedule_file,'--source-capture',source,'--output-dir',dirs[mode],
                     '--exp-name','aaa_continuation_label_train64_001_'+label+mode]
            if mode=='enabled':options+=['--enabled','--control-gate',audits['disabled']/'summary.json']
            stage(label+mode,'run_single_intervention.py',options)
            stage(label+mode+'_audit','audit_single_intervention.py',['--run-dir',dirs[mode],
                  '--source-noninterference',control,'--source-probe',probe,'--output-dir',audits[mode]])
        stage(label+'native_reconstruction','verify_intervention_metrics.py',['--run-dir',dirs['enabled'],
              '--output-dir',args.root/(label+'native_reconstruction_001')])
    run_pair('',schedule_path)
    stage('h2_analysis','analyze_prospective_continuation.py',['--config',args.config,'--root',args.root,
          '--output-dir',args.root/'h2_analysis_001'])
    analysis=json.loads((args.root/'h2_analysis_001/summary.json').read_text())
    accepted={e['episode_id'] for e in analysis['events'] if e['h2_accepted']}
    if analysis['mixed_schedule_rollout_gate'] and len(accepted)<len(schedule['events']):
        mixed=dict(schedule)
        mixed.update(experiment_id=cfg['experiment_id']+'-H2-CONFIRMED',
                     status='frozen_after_registered_h2_rule_before_mixed_rollout',
                     events=[e for e in schedule['events'] if e['episode_id'] in accepted],
                     expected_event_count=len(accepted),
                     selection_rule='Registered H=2 confirmation on already frozen H=1 candidates; privileged future information')
        mixed['hashes']=dict(schedule['hashes'])
        mixed['hashes'][str(args.root/'h2_analysis_001/summary.json')]=digest(args.root/'h2_analysis_001/summary.json')
        mixed_path=args.root/'schedule_h2_001.json'
        with mixed_path.open('x') as f:json.dump(mixed,f,indent=2);f.write('\n')
        run_pair('h2_',mixed_path)
        manifest['h2_execution']='separate_mixed_schedule_completed'
    elif accepted=={e['episode_id'] for e in schedule['events']}:
        manifest['h2_execution']='identical_to_already_executed_h1_schedule'
    else:
        manifest['h2_execution']='not_run_registered_gate_failed'
    if manifest['h2_execution']!='not_run_registered_gate_failed':
        from audit_single_intervention import read_rows,one
        baseline=read_rows(one(source/'traces','*.jsonl'))
        h1=read_rows(one(args.root/'enabled_001/traces','*.jsonl'))
        actual_dir=args.root/('h2_enabled_001' if manifest['h2_execution']=='separate_mixed_schedule_completed' else 'enabled_001')
        actual=read_rows(one(actual_dir/'traces','*.jsonl'))
        expected={k:r for k,r in baseline.items() if k[1] not in accepted}
        expected.update({k:r for k,r in h1.items() if k[1] in accepted})
        if actual!=expected:raise ValueError('Mixed rollout differs from composed full traces')
        check={'passed':True,'exact_full_trace_comparisons':len(actual),'accepted_routes':len(accepted),
               'mode':manifest['h2_execution'],'actual_run_dir':str(actual_dir),
               'actual_trace_sha256':digest(one(actual_dir/'traces','*.jsonl'))}
        (args.root/'h2_composition_verification_001.json').write_text(json.dumps(check,indent=2)+'\n')
        manifest['composition_verification']=check
    manifest.update(status='complete',finished_utc=now(),no_model_trained=True);save()
    print('COMPLETE '+cfg['experiment_id'],flush=True)


if __name__=='__main__':main()
