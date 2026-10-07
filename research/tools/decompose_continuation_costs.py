#!/usr/bin/env python3
"""Descriptive complete-cost accounting; never label mechanisms as causal fixes."""
import argparse
import collections
import json
from pathlib import Path

import numpy as np

from audit_single_intervention import read_rows, group_rows, one, key
from analyze_continuation_horizon import movement, window
from run_option_capture import digest


def breakdown(records, step):
    groups={'immediate':[records[step]],'later_navigation':[], 'later_stop':[], 'later_other':[]}
    for row in records[step+1:]:
        name={4:'later_navigation',0:'later_stop'}.get(row['action']['act'],'later_other')
        groups[name].append(row)
    return {name:{'primitives':sum(len(r['primitives']) for r in rows),
                  'movement_m':sum(movement(r) for r in rows),
                  'collision_events':sum(p['collided'] is True for r in rows for p in r['primitives']),
                  'decisions':len(rows)} for name,rows in groups.items()}


def vector(value):
    return np.asarray(value['__array__'] if isinstance(value,dict) else value,dtype=np.float64)


def stop_details(records, decision_rows):
    row=records[-1]
    decision=decision_rows[key(row)]
    action=row['action']
    return {'action_type':action['act'],'step':row['high_level_step'],
            'primitive_count':len(row['primitives']),'movement_m':movement(row),
            'budget_stop':bool(decision['budget_stop']),
            'back_path_nodes':len(action.get('back_path') or []),
            'stop_vp':action.get('stop_vp'),'current_vp':action.get('cur_vp'),
            'distance_after':row['distance_after']}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args()
    cfg=json.loads(args.config.read_text())
    hashes={str(args.config):digest(args.config),str(Path(__file__)):digest(Path(__file__)),
            'research/tools/analyze_continuation_horizon.py':digest('research/tools/analyze_continuation_horizon.py')}
    events=[];batches={}
    for batch in cfg['batches']:
        root=Path(batch['root']);run=root/'enabled_001';audit_dir=root/'enabled_audit_001'
        audit=json.loads((audit_dir/'summary.json').read_text())
        if not audit['all_required_gates_passed']:raise ValueError('Unverified actual trajectories')
        manifest=json.loads((run/'manifest.json').read_text())
        source=Path(manifest['source_capture'])
        before_path=one(source/'traces','*.jsonl');after_path=one(run/'traces','*.jsonl')
        before,after=group_rows(read_rows(before_path)),group_rows(read_rows(after_path))
        original_choices=read_rows(source/'graph_options/graph_options.jsonl')
        new_choices=read_rows(run/'hook/decisions.jsonl')
        pairs={r['episode_id']:r for r in json.loads((audit_dir/'paired_episodes.json').read_text())}
        for f in [audit_dir/'summary.json',audit_dir/'paired_episodes.json',run/'manifest.json',before_path,
                  after_path,source/'graph_options/graph_options.jsonl',run/'hook/decisions.jsonl']:
            hashes[str(f)]=digest(f)
        batch_rows=[]
        for event in manifest['schedule']['events']:
            ep,step=event['episode_id'],event['high_level_step']
            a,b=breakdown(before[ep],step),breakdown(after[ep],step)
            delta={group:{m:b[group][m]-a[group][m] for m in a[group]} for group in a}
            primitive_total=sum(v['primitives'] for v in delta.values())
            if primitive_total!=pairs[ep]['delta']['primitive_action_count']:
                raise ValueError('Primitive decomposition does not tie to native metrics')
            if sum(v['collision_events'] for v in delta.values())!=pairs[ep]['delta']['collision_events']:
                raise ValueError('Collision decomposition does not tie')
            old_id=event['baseline_action']['ghost_vp']
            reselected=[]
            for row in after[ep][step+1:]:
                if row['action'].get('ghost_vp')==old_id:
                    reselected.append({'step':row['high_level_step'],
                       'requested_target_shift_m':float(np.linalg.norm(vector(row['action']['ghost_pos'])-vector(event['baseline_action']['ghost_pos']))),
                       'primitive_count':len(row['primitives']),
                       'back_path_nodes':len(row['action'].get('back_path') or [])})
            horizons=[]
            for h in range(1,max(len(before[ep]),len(after[ep]))-step+1):
                x,y=window(before[ep],step,h),window(after[ep],step,h)
                horizons.append({'horizon':h,'extra_goal_progress_m':x['distance_to_goal']-y['distance_to_goal'],
                                 'primitive_delta':y['primitive_count']-x['primitive_count'],
                                 'baseline_terminal':x['terminal'],'intervention_terminal':y['terminal']})
            item={'batch':batch['name'],'episode_id':ep,'scene_id':event['scene_id'],'step':step,
                  'cost_breakdown_baseline':a,'cost_breakdown_intervention':b,'cost_deltas':delta,
                  'final_delta':pairs[ep]['delta'],'primitive_tieout_passed':True,
                  'collision_tieout_passed':True,'same_ghost_id_reselected':reselected,
                  'baseline_stop':stop_details(before[ep],original_choices),
                  'intervention_stop':stop_details(after[ep],new_choices),
                  'descriptive_horizon_profile':horizons}
            events.append(item);batch_rows.append(item)
        total=sum(r['final_delta']['primitive_action_count'] for r in batch_rows)
        batches[batch['name']]={'routes':len(batch_rows),'scenes':len({r['scene_id'] for r in batch_rows}),
            'total_primitive_delta':total,
            'group_primitive_deltas':{k:sum(r['cost_deltas'][k]['primitives'] for r in batch_rows) for k in a},
            'same_displaced_ghost_id_reselected_routes':sum(bool(r['same_ghost_id_reselected']) for r in batch_rows),
            'leave_one_route_out_primitive_delta':{r['episode_id']:total-r['final_delta']['primitive_action_count'] for r in batch_rows}}
    result={'experiment_id':cfg['experiment_id'],'analysis_type':cfg['analysis_type'],
            'all_accounting_checks_passed':True,'events':events,'batches':batches,
            'limits':'Post-hoc descriptive accounting on two differently selected training batches. Do not pool as benchmark performance. Same ghost ID is not necessarily the same full action; target shift is retained. High-level horizon profiles have unequal primitive costs; no horizon tuning or causal claim.',
            'no_model_trained':True,'hashes':hashes}
    args.output_dir.mkdir(parents=True,exist_ok=False)
    (args.output_dir/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['| Batch / episode | Immediate | Later navigation | Later STOP | Total primitives | nDTW delta pp |',
           '| --- | ---: | ---: | ---: | ---: | ---: |']
    for r in events:
        d=r['cost_deltas'];f=r['final_delta']
        lines.append('| {} / {} | {:+d} | {:+d} | {:+d} | {:+d} | {:+.3f} |'.format(r['batch'],r['episode_id'],
            d['immediate']['primitives'],d['later_navigation']['primitives'],d['later_stop']['primitives'],
            f['primitive_action_count'],100*f['ndtw']))
    (args.output_dir/'paired_table.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(batches,indent=2))


if __name__=='__main__':main()
