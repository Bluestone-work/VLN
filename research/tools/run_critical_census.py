#!/usr/bin/env python3
"""Run frozen native-action full returns and the small interrupt oracle."""
import argparse,json,os,subprocess,sys,shutil
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));os.chdir(str(ROOT))
from run_option_capture import digest

def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--exp-name',required=True);p.add_argument('--smoke',action='store_true');p.add_argument('--smoke-gate',type=Path);a=p.parse_args()
    plan=json.loads(a.plan.read_text());cfg=plan['config'];root=Path(cfg['source_root']);source=json.loads((root/'capture_001/manifest.json').read_text())
    for name,h in list(plan.get('hashes', {}).items())+list(source['hashes'].items()):
        if digest(name)!=h:raise ValueError('Frozen input changed: '+name)
    if not a.smoke:
        if not a.smoke_gate:raise ValueError('Smoke gate required')
        gate=json.loads((a.smoke_gate/'summary.json').read_text())
        gm=json.loads((a.smoke_gate/'manifest.json').read_text())
        if gate['status']!='smoke_complete' or not gate['all_validity_checks_passed']:raise ValueError('Smoke gate failed')
        if gm['plan']!=plan:raise ValueError('Smoke used a different plan')
        for name in ['research/tools/critical_case_driver.py','vlnce_baselines/adaptive_action/critical_census.py']:
            if digest(name)!=gm['hashes'][name]:raise ValueError('Execution code changed after smoke')
    if (Path('data/logs/eval_results')/a.exp_name).exists():raise ValueError('Native output exists')
    a.output.mkdir(parents=True,exist_ok=False)
    opts=dict(source['overrides']);opts.update(TRAINER_NAME='SS-ETP-CriticalCensus',ENV_NAME='VLNCECriticalEnv')
    # The custom driver evaluates explicitly scheduled routes; outer native eval
    # aggregates one baseline control on return, never oracle cases as a benchmark.
    opts['EVAL.EPISODE_COUNT']='1'
    opts['TASK_CONFIG.DATASET.DATA_PATH']=str(Path(cfg.get('failure_subset_dataset',a.plan.parent/('failure_routes_{}.json.gz'.format(len(plan['failed_episodes']))))))
    paths=[a.plan,Path(__file__),Path('research/tools/critical_case_driver.py'),Path('vlnce_baselines/adaptive_action/critical_census.py'),Path(cfg['protocol'])]
    paths.append(Path(opts['TASK_CONFIG.DATASET.DATA_PATH']))
    paths.append(root/'capture_001/traces/worker_seed100.jsonl')
    branch_path = root/cfg['probe']/'branch_traces.jsonl'
    if branch_path.exists():
        paths.append(branch_path)
    manifest={'created_utc':datetime.now(timezone.utc).isoformat(),'plan':plan,'plan_path':str(a.plan),'source':source,'overrides':opts,'smoke':a.smoke,'output':str(a.output.resolve()),'exp_name':a.exp_name,'git_commit':subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),'hashes':{str(p):digest(p) for p in paths},'hardware':subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version','--format=csv,noheader']).decode(),'python':sys.version,'argv':sys.argv,'oracle_analysis_only':True}
    manifest_path=a.output/'manifest.json';manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    for name in list(source['hashes'])+[str(p) for p in paths]:
        p=Path(name)
        if p.suffix in ['.py','.yaml','.md','.json'] and p.stat().st_size<5000000:
            dst=a.output/'source_snapshot'/p;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(str(p),str(dst))
    (a.output/'git_diff.patch').write_bytes(subprocess.check_output(['git','diff','--binary']))
    os.environ['ETPNAV_CRITICAL_MANIFEST']=str(manifest_path.resolve())
    os.environ.pop('ETPNAV_OPTION_TRACE_DIR',None);os.environ.pop('ETPNAV_GRAPH_OPTION_DIR',None)
    from vlnce_baselines.adaptive_action.critical_census import CriticalEnv,CriticalTrainer
    from run import run_exp
    run_exp(a.exp_name,source['config']['base_config'],'eval',opts=[x for pair in opts.items() for x in pair],local_rank=0)

if __name__=='__main__':main()
