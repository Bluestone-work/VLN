#!/usr/bin/env python3
"""Frozen replication with preregistration checks and gated sequential stages."""
import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime,timezone
from pathlib import Path


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda:stream.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);ap.add_argument('--register',action='store_true');a=ap.parse_args()
    cfg=json.loads(a.config.read_text());root=Path(cfg['replication']['root']);regpath=root/'registration_001.json'
    if a.register:
        if (root/'capture_001').exists():raise ValueError('Cannot preregister after target capture')
        audit=root/'sampling_001/route_audit_001.json'
        if not json.loads(audit.read_text())['valid_for_confirmation']:raise ValueError('Sample audit failed')
        paths=[a.config,Path(cfg['protocol']),Path(cfg['base_config']),Path(cfg['checkpoint']),Path(cfg['sampling_manifest']),
               Path(cfg['dataset_path']),Path(cfg['replication']['full_model']),Path(cfg['replication']['cost_model']),audit]
        paths += [Path('research/tools')/name for name in ['run_preference_replication.py','prepare_preference_replication_schedules.py','analyze_preference_replication.py','freeze_reduced_graph_preference.py','graph_value_feasibility.py','prepare_ranker_intervention_schedule.py','run_graph_option_cycle.py','run_single_intervention.py','audit_single_intervention.py','verify_intervention_metrics.py']]
        paths += [Path('vlnce_baselines')/name for name in ['ss_trainer_ETP.py','common/environments.py','adaptive_action/single_intervention.py']]
        result={'registered_utc':datetime.now(timezone.utc).isoformat(),'source_git_commit':subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),
                'no_target_capture_exists':True,'config':cfg,'hashes':{str(p):digest(p) for p in paths}}
        with regpath.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
        print('Registered '+str(regpath));return
    reg=json.loads(regpath.read_text())
    for path,sha in reg['hashes'].items():
        if digest(path)!=sha:raise ValueError('Registered source changed: '+path)
    directory=root/'replication_cycle_001';directory.mkdir(exist_ok=False)
    status={'status':'running','git_commit':subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),'registration_sha256':digest(regpath),'stages':[]}
    def save():(directory/'status.json').write_text(json.dumps(status,indent=2)+'\n')
    def stage(name,script,options):
        cmd=[sys.executable,'research/tools/'+script]+[str(x) for x in options]
        step={'name':name,'command':cmd,'started_utc':datetime.now(timezone.utc).isoformat()};status['stages'].append(step);save()
        print('START '+name,flush=True)
        with (directory/(name+'.stdout.log')).open('xb') as out,(directory/(name+'.stderr.log')).open('xb') as err:
            code=subprocess.call(cmd,stdout=out,stderr=err)
        step.update(returncode=code,finished_utc=datetime.now(timezone.utc).isoformat());save()
        if code:status['status']='failed';save();raise RuntimeError(name+' failed; original attempt retained')
        print('PASS '+name,flush=True)
    stage('foundation','run_graph_option_cycle.py',['--config',a.config,'--root',root,'--exp-prefix','aaa_gv_prospective001'])
    stage('schedules','prepare_preference_replication_schedules.py',['--config',a.config])
    for arm in cfg['replication']['arms']:
        schedule=root/'schedules_001'/(arm+'.json')
        if not json.loads(schedule.read_text())['events']:
            with (directory/(arm+'_noop.json')).open('x') as stream:json.dump({'arm':arm,'no_changed_actions':True,'schedule_sha256':digest(schedule),'native_metrics_source':str(root/'noninterference_001/capture_episodes.json')},stream,indent=2)
            continue
        for mode in ['disabled','enabled']:
            name=arm+'_'+mode; run=directory/name
            opts=['--schedule',schedule,'--source-capture',root/'capture_001','--output-dir',run,'--exp-name','aaa_gv_prospective001_'+name]
            if mode=='enabled':opts+=['--enabled','--control-gate',directory/(arm+'_disabled_audit')/'summary.json']
            stage(name,'run_single_intervention.py',opts)
            stage(name+'_audit','audit_single_intervention.py',['--run-dir',run,'--source-noninterference',root/'noninterference_001','--source-probe',root/'probe_full001','--output-dir',directory/(name+'_audit')])
            if mode=='enabled':stage(name+'_metrics','verify_intervention_metrics.py',['--run-dir',run,'--output-dir',directory/(name+'_metrics')])
    status['all_arms_complete']=True;save()
    stage('analysis','analyze_preference_replication.py',['--config',a.config,'--output',root/'paired_analysis_001'])
    status['status']='complete';save()

if __name__=='__main__':main()
