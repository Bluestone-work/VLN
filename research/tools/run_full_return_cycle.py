#!/usr/bin/env python3
"""Run two outcome-blind alternatives through native continuation and audit both."""
import argparse,json,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
from run_option_capture import digest

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--tag',default='001');args=p.parse_args()
    cfg=json.loads(args.config.read_text());root=args.root
    if json.loads((root/'cycle_001/status.json').read_text()).get('status')!='complete':raise ValueError('Foundation incomplete')
    if not json.loads((root/'integrity_001/summary.json').read_text())['integrity_gate_passed']:raise ValueError('Integrity incomplete')
    schedules=[('top_logit',root/'schedule_top_logit_001.json'),('seeded_graph_id',root/'schedule_seeded_graph_id_001.json')]
    source=root/'capture_001';probe=root/'probe_full001';control=root/'control_001'
    manifest={'experiment_id':cfg['experiment_id'],'status':'running','started_utc':datetime.now(timezone.utc).isoformat(),'stages':[]}
    out=root/('full_return_cycle_'+args.tag);out.mkdir(exist_ok=False)
    def stage(name,script,argv):
        cmd=[sys.executable,'research/tools/'+script]+[str(x) for x in argv]
        rec={'name':name,'command':cmd,'status':'running','started_utc':datetime.now(timezone.utc).isoformat()};manifest['stages'].append(rec)
        (out/'status.json').write_text(json.dumps(manifest,indent=2)+'\n')
        with (out/(name+'.stdout.log')).open('x') as so,(out/(name+'.stderr.log')).open('x') as se:code=subprocess.call(cmd,stdout=so,stderr=se)
        rec.update(status='passed' if code==0 else 'failed',returncode=code,finished_utc=datetime.now(timezone.utc).isoformat());(out/'status.json').write_text(json.dumps(manifest,indent=2)+'\n')
        if code:raise RuntimeError(name+' failed')
    for kind,schedule in schedules:
        disabled=root/(kind+'_disabled_'+args.tag);daudit=root/(kind+'_disabled_audit_'+args.tag)
        enabled=root/(kind+'_enabled_'+args.tag);eaudit=root/(kind+'_enabled_audit_'+args.tag);recon=root/(kind+'_native_reconstruction_'+args.tag)
        expbase='aaa_full_return_label_train16_'+args.tag+'_'+kind+'_'
        stage(kind+'_disabled','run_single_intervention.py',['--schedule',schedule,'--source-capture',source,'--output-dir',disabled,'--exp-name',expbase+'disabled'])
        stage(kind+'_disabled_audit','audit_single_intervention.py',['--run-dir',disabled,'--source-noninterference',root/'noninterference_001','--source-probe',probe,'--output-dir',daudit])
        stage(kind+'_enabled','run_single_intervention.py',['--schedule',schedule,'--source-capture',source,'--output-dir',enabled,'--exp-name',expbase+'enabled','--enabled','--control-gate',daudit/'summary.json'])
        stage(kind+'_enabled_audit','audit_single_intervention.py',['--run-dir',enabled,'--source-noninterference',root/'noninterference_001','--source-probe',probe,'--output-dir',eaudit])
        stage(kind+'_reconstruction','verify_intervention_metrics.py',['--run-dir',enabled,'--output-dir',recon])
    manifest.update(status='complete',finished_utc=datetime.now(timezone.utc).isoformat(),no_model_trained=True,tag=args.tag)
    (out/'status.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps({'status':manifest['status'],'stages':len(manifest['stages'])},indent=2))

if __name__=='__main__':main()
