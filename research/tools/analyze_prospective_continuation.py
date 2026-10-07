#!/usr/bin/env python3
"""Apply the registered H=2 filter; preserve all adverse and rejected outcomes."""
import argparse
import collections
import json
from pathlib import Path

from audit_single_intervention import read_rows, group_rows, one
from analyze_continuation_horizon import window
from continuation_labels import confirmation
from run_option_capture import digest


METRICS = ['success','spl','ndtw','sdtw','distance_to_goal','path_length',
           'primitive_action_count','high_level_steps','collision_events']


def aggregate(pairs, ids, use_alternative):
    rows = [p for p in pairs if p['episode_id'] in ids]
    result = {'episodes':len(rows),'metrics':{},'success_rescued':[],'success_lost':[]}
    for p in rows:
        before=p['baseline']
        after=p['intervention'] if p['episode_id'] in use_alternative else before
        if after['success']>before['success']:result['success_rescued'].append(p['episode_id'])
        if after['success']<before['success']:result['success_lost'].append(p['episode_id'])
    for m in METRICS:
        before=[p['baseline'][m] for p in rows]
        after=[(p['intervention'] if p['episode_id'] in use_alternative else p['baseline'])[m] for p in rows]
        result['metrics'][m]={'baseline_mean':sum(before)/len(rows) if rows else None,
                              'composed_mean':sum(after)/len(rows) if rows else None,
                              'sum_delta':sum(after)-sum(before),
                              'mean_delta':(sum(after)-sum(before))/len(rows) if rows else None}
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args()
    cfg=json.loads(args.config.read_text())
    run_dir=args.root/'enabled_001';audit_dir=args.root/'enabled_audit_001'
    audit=json.loads((audit_dir/'summary.json').read_text())
    if not audit['all_required_gates_passed'] or audit['mode']!='enabled':
        raise ValueError('Accepted actual continuations required')
    manifest=json.loads((run_dir/'manifest.json').read_text())
    if manifest['config']!=cfg:raise ValueError('Different registered configuration')
    schedule=manifest['schedule']
    for source,expected in schedule['hashes'].items():
        if digest(source)!=expected:raise ValueError('Frozen source changed: '+source)
    old_path=one(Path(manifest['source_capture'])/'traces','*.jsonl')
    new_path=one(run_dir/'traces','*.jsonl')
    old,new=group_rows(read_rows(old_path)),group_rows(read_rows(new_path))
    pairs=json.loads((audit_dir/'paired_episodes.json').read_text())
    lookup={p['episode_id']:p for p in pairs}
    events=[]
    for event in schedule['events']:
        ep,step=event['episode_id'],event['high_level_step']
        observations={}
        for h in [1,cfg['continuation']['primary_horizon'],cfg['continuation']['descriptive_horizon']]:
            a,b=window(old[ep],step,h),window(new[ep],step,h)
            observations[str(h)]={'baseline':a,'intervention':b,
                                   'comparison':confirmation(a,b,cfg['continuation'])}
        if not observations['1']['comparison']['accepted']:
            raise ValueError('Registered H=1 candidate no longer passes original gain/cost gate')
        h2=observations[str(cfg['continuation']['primary_horizon'])]
        accepted=h2['comparison']['accepted']
        delta=lookup[ep]['delta']
        post=step+cfg['continuation']['primary_horizon']
        later_primitive_delta=sum(len(r['primitives']) for r in new[ep][post:])-sum(len(r['primitives']) for r in old[ep][post:])
        final_extra_progress=old[ep][-1]['distance_after']-new[ep][-1]['distance_after']
        events.append({'episode_id':ep,'scene_id':event['scene_id'],'high_level_step':step,
                       'horizons':observations,'h2_accepted':accepted,'final_delta':delta,
                       'final_extra_goal_progress_m':final_extra_progress,
                       'both_terminal_by_h2':h2['baseline']['terminal'] and h2['intervention']['terminal'],
                       'either_terminal_by_h2':h2['baseline']['terminal'] or h2['intervention']['terminal'],
                       'extra_goal_progress_strictly_after_h2_m':final_extra_progress-h2['comparison']['extra_goal_progress_m'],
                       'primitive_delta_strictly_after_h2':later_primitive_delta,
                       'success_lost':delta['success']<0,
                       'ndtw_worse':delta['ndtw']<-1e-6,
                       'episode_primitive_cost_increased':delta['primitive_action_count']>0})
    treated={r['episode_id'] for r in events}
    accepted={r['episode_id'] for r in events if r['h2_accepted']}
    rejected=treated-accepted
    population=set(lookup)
    groups={}
    for name,ids in [('treated',treated),('accepted',accepted),('rejected',rejected),('all_routes',population)]:
        groups[name]={'h1_actual':aggregate(pairs,ids,treated),
                      'h2_offline_composition':aggregate(pairs,ids,accepted)}
    scene_count=len({r['scene_id'] for r in events})
    accepted_rows=[r for r in events if r['h2_accepted']]
    risk_counts={name:sum(r[name] for r in accepted_rows) for name in
                 ['success_lost','ndtw_worse','episode_primitive_cost_increased','both_terminal_by_h2','either_terminal_by_h2']}
    cm=groups['treated']['h2_offline_composition']['metrics']
    efficiency=bool(accepted) and (cm['spl']['sum_delta']>1e-6 or cm['ndtw']['sum_delta']>1e-6)
    # Registered diagnostic gate only; favorable means never erase individual harm.
    should_verify=bool(accepted) and efficiency and not risk_counts['success_lost'] and cm['primitive_action_count']['sum_delta']<=0
    any_cost_harm=risk_counts['episode_primitive_cost_increased']>0
    bootstrap={'status':'not_run_insufficient_treated_scenes','minimum_scenes':8}
    if scene_count>=8:
        from audit_ranker_validity import cluster_ci
        bootstrap={'status':'descriptive_paired_scene_bootstrap','samples':5000,'seed':20261007,'metrics':{}}
        for metric in METRICS:
            values=[r['final_delta'][metric] if r['h2_accepted'] else 0. for r in events]
            bootstrap['metrics'][metric]=cluster_ci(values,[r['scene_id'] for r in events],20261007,5000)
    result={'experiment_id':cfg['experiment_id'],'analysis_type':'prospectively registered H=2 confirmation of H=1 candidates',
            'total_routes':len(population),'treated_routes':len(treated),'treated_scenes':scene_count,
            'accepted_routes':len(accepted),'rejected_routes':len(rejected),'events':events,'groups':groups,
            'accepted_risk_counts':risk_counts,'paired_scene_bootstrap':bootstrap,
            'mixed_schedule_rollout_gate':should_verify,
            'current_filter_insufficient_due_to_later_harm':bool(risk_counts['success_lost'] or any_cost_harm),
            'learning_support_gate_passed':False,
            'next_action':'verify_mixed_schedule' if should_verify else 'diagnose_or_stop_label_rule',
            'privileged_analysis_only':True,'h2_result_is_offline_composition':True,'learned_model_trained':False,
            'limits':'One candidate per eligible route, one seed, new training scenes; H=2 is future oracle information. This is not best-of-all-H=2 search. SR/path metrics overlap windows when terminal; report suffix diagnostics. No learned, held-out or novelty claim.',
            'hashes':{str(f):digest(f) for f in [args.config,Path(cfg['continuation_protocol']),Path(__file__),
                        Path(__file__).with_name('continuation_labels.py'),Path(__file__).with_name('analyze_continuation_horizon.py'),
                        audit_dir/'summary.json',audit_dir/'paired_episodes.json',old_path,new_path]}}
    args.output_dir.mkdir(parents=True,exist_ok=False)
    with (args.output_dir/'summary.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    columns=['Episode / step','H2 accepted','H2 gain m','H2 primitive delta','Final nDTW delta pp','Final primitive delta','Success delta','Both stop by H2']
    lines=['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']
    for r in events:
        h=r['horizons']['2']['comparison'];d=r['final_delta']
        lines.append('| {} / {} | {} | {:.4f} | {:+d} | {:+.3f} | {:+d} | {:+.0f} | {} |'.format(
            r['episode_id'],r['high_level_step'],r['h2_accepted'],h['extra_goal_progress_m'],h['primitive_delta'],
            d['ndtw']*100,d['primitive_action_count'],d['success'],r['both_terminal_by_h2']))
    (args.output_dir/'paired_table.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['events','groups','hashes','paired_scene_bootstrap']},indent=2))


if __name__=='__main__':main()
