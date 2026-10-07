#!/usr/bin/env python3
"""Execute and audit registered one-action arms, retaining exclusive logs."""
import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--exp-prefix', required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    state = {'git_commit': subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),
             'argv': sys.argv, 'stages': [], 'status': 'running'}
    def save():
        (args.output/'status.json').write_text(json.dumps(state, indent=2)+'\n')
    def run(name, script, options):
        cmd = [sys.executable, 'research/tools/'+script] + [str(x) for x in options]
        step = {'name': name, 'command': cmd, 'started_utc': datetime.now(timezone.utc).isoformat()}
        state['stages'].append(step); save()
        print('START '+name, flush=True)
        with (args.output/(name+'.stdout.log')).open('xb') as out, (args.output/(name+'.stderr.log')).open('xb') as err:
            code = subprocess.call(cmd, stdout=out, stderr=err)
        step.update(returncode=code, finished_utc=datetime.now(timezone.utc).isoformat()); save()
        if code:
            state['status']='failed'; save(); raise RuntimeError(name+' failed')
        print('PASS '+name, flush=True)
    arms = [('learned', 'schedule_ranker_002.json')] + [('random_'+str(seed), 'schedule_random_'+str(seed)+'.json') for seed in [20261021,20261022,20261023]]
    for arm, schedule in arms:
        for mode in ['disabled', 'enabled']:
            name = arm+'_'+mode
            directory = args.output/name
            opts = ['--schedule',args.root/schedule,'--source-capture',args.root/'capture_001','--output-dir',directory,'--exp-name',args.exp_prefix+'_'+name]
            if mode == 'enabled':
                opts += ['--enabled','--control-gate',args.output/(arm+'_disabled_audit')/'summary.json']
            run(name, 'run_single_intervention.py', opts)
            run(name+'_audit','audit_single_intervention.py',['--run-dir',directory,'--source-noninterference',args.root/'noninterference_001','--source-probe',args.root/'probe_full001','--output-dir',args.output/(name+'_audit')])
            if mode == 'enabled':
                run(name+'_metrics','verify_intervention_metrics.py',['--run-dir',directory,'--output-dir',args.output/(name+'_metrics')])
    state['status']='complete'; save()


if __name__ == '__main__':
    main()
