#!/usr/bin/env python3
"""Freeze all native graph actions at registered failure-critical states."""
import argparse
import json
from pathlib import Path
from run_option_capture import digest


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    cfg=json.loads(args.config.read_text());root=Path(cfg['source_root']);capture=root/'capture_001';probe=root/cfg['probe'];ni=root/cfg['noninterference']
    for path,field in [(probe/'summary.json','all_required_gates_passed'),(ni/'summary.json','baseline_noninterference_gate_passed'),(root/'integrity_001/summary.json','integrity_gate_passed')]:
        if not json.loads(path.read_text())[field]:raise ValueError('Failed source gate: '+str(path))
    metrics=json.loads((ni/'capture_episodes.json').read_text());fail=sorted(ep for ep,m in metrics.items() if not m['success'])
    graphs={}
    for line in (capture/'graph_options/graph_options.jsonl').open():
        g=json.loads(line)
        if str(g['episode_id']) in fail:
            for o in g['options']:o.pop('embedding',None)
            graphs[(str(g['episode_id']),g['high_level_step'])]=g
    decisions={}
    for line in (probe/'decisions.jsonl').open():
        d=json.loads(line)
        if str(d['episode_id']) in fail:decisions[(str(d['episode_id']),d['high_level_step'])]=d
    states=[];cases=[]
    for ep in fail:
        rows=sorted((d for (e,s),d in decisions.items() if e==ep),key=lambda d:d['high_level_step'])
        nonstop=[d['high_level_step'] for d in rows if d['effective_index']>0]
        for d in rows:
            step=d['high_level_step'];selected=next(o for o in d['options'] if o['index']==d['effective_index']);reasons=[]
            if selected['action_type']==4:
                if selected['collision_events']>=cfg['collision_min']:reasons.append('collision')
                if selected['target_error_horizontal_m']>=cfg['endpoint_deviation_m']:reasons.append('deviation')
                if selected['primitive_events']>=cfg['stall_primitives_min'] and selected['motion_path_m']<=cfg['stall_path_m_max']:reasons.append('stall')
                if step==nonstop[-1]:reasons.append('last_nonstop')
            else:reasons.append('terminal')
            if not reasons:continue
            g=graphs[(ep,step)];opts=[o for o in g['options'] if o['admissible']]
            assert [o['index'] for o in opts]==[o['index'] for o in d['options']]
            state={'episode_id':ep,'scene_id':g['scene_id'],'high_level_step':step,'reasons':reasons,'effective_index':g['effective_index'],'actions':len(opts)};states.append(state)
            for o in opts:
                cases.append(dict(state,case_id='e{}_s{:02d}_a{:03d}'.format(ep,step,o['index']),mode='action',action_index=o['index'],action=o['action']))
    hashes={str(p):digest(p) for p in [args.config,Path(cfg['protocol']),capture/'manifest.json',capture/'graph_options/graph_options.jsonl',probe/'decisions.jsonl',ni/'capture_episodes.json',root/'integrity_001/summary.json']}
    result={'config':cfg,'failed_episodes':fail,'states':states,'cases':cases,'hashes':hashes,'scope':'all registered critical states, exhaustive admissible native actions; single replacement then full native continuation'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({'failures':len(fail),'critical_states':len(states),'full_return_cases':len(cases),'scenes':len({s['scene_id'] for s in states})}))

if __name__=='__main__':main()
