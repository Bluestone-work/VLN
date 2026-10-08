"""Confirmation driver for event versus outcome-blind timing arms."""
import json
import time
from pathlib import Path
import numpy as np
from fastdtw import fastdtw
from replay_option_calibration import compare
from route_alignment import trace_path
from vlnce_baselines.adaptive_action.critical_census import CriticalHook


def reconstruction(records,reference,native):
    points=[]
    for r in records:
        segment=trace_path(r)
        if points and not np.array_equal(points[-1],segment[0]):raise ValueError('Discontinuous physical path')
        points.extend(segment if not points else segment[1:])
    path=np.asarray(points,dtype=np.float32);ref=np.asarray(reference,dtype=np.float64)
    length=float(np.linalg.norm(path[1:]-path[:-1],axis=1).sum());error=records[-1]['distance_after'];success=float(error<=3)
    ndtw=float(np.exp(-fastdtw(path,ref,dist=lambda a,b:np.linalg.norm(b-a))[0]/(len(ref)*3)))
    shortest=records[0]['distance_before']
    values={'distance_to_goal':error,'success':success,'spl':success*shortest/max(shortest,length),'ndtw':ndtw,'sdtw':ndtw*success,'path_length':length,'steps_taken':sum(len(r['primitives']) for r in records),'high_level_steps':len(records)}
    deltas={k:abs(values[k]-native[k]) for k in values}
    if max(deltas.values())>1e-6:raise ValueError('Native final metrics mismatch: '+str(deltas))
    if len(records)>15 or not records[-1]['done'] or records[-1]['action']['act']!=0:raise ValueError('Invalid native horizon or termination')
    return {'comparisons':len(values),'max_delta':max(deltas.values()),'passed':True}


def find_interrupts(controls,states,max_routes):
    critical={(s['episode_id'],s['high_level_step']):s for s in states if any(r in s['reasons'] for r in ['collision','stall','deviation'])}
    events=[]
    for ep in sorted(controls):
        for rec in controls[ep]:
            k=(ep,rec['high_level_step'])
            if k not in critical or rec['action']['act']!=4:continue
            seq=rec['primitives'];phases=rec['phases'];positions=[rec['pre_pose']['position']]+[p['pose']['position'] for p in seq]
            streak=0;cut=None;reason=None
            for i,(p,phase) in enumerate(zip(seq,phases)):
                if phase!='ghost':streak=0;continue
                if p['action']==1 and np.linalg.norm(np.asarray(positions[i+1])-positions[i])<=0.01:streak+=1
                else:streak=0
                if i+1>=len(seq):continue
                if p['collided'] and p['action']==1:cut=i+1;reason='collision';break
                if streak>=3:cut=i+1;reason='stall';break
            if cut is None and 'deviation' in critical[k]['reasons']:
                indices=[i for i,p in enumerate(seq) if phases[i]=='ghost' and p['action']==1]
                if len(indices)>=2:
                    i=indices[(len(indices)-1)//2];target=np.asarray(rec['action']['ghost_pos']['__array__'])
                    residual=np.linalg.norm((target-np.asarray(positions[i+1]))[[0,2]])
                    if residual>(len(indices)-indices.index(i)-1)*0.25+0.5 and i+1<len(seq):cut=i+1;reason='midpoint_deviation_oracle'
            if cut:
                events.append({'episode_id':ep,'scene_id':rec['scene_id'],'high_level_step':rec['high_level_step'],'cut_primitive':cut,'cut_reason':reason,'baseline_primitive_count':len(seq)})
                break
        if len(events)>=max_routes:break
    return events


def drive_cases(trainer,native_rollout):
    manifest=trainer.manifest;plan=manifest['plan'];cfg=plan['config'];root=Path(cfg['source_root']);out=Path(manifest['output']);gates=manifest['source']['config'].get('gates', {'endpoint_tolerance_m':1e-5,'rotation_tolerance_rad':1e-5,'progress_tolerance_m':1e-5,'primitive_action_sequence_exact':True,'per_primitive_collision_sequence_exact':True,'rng_post_exact':True,'terminal_flag_exact':True,'baseline_episode_metrics_tolerance':1e-6})
    fail=set(plan.get('replay_episodes', plan.get('failed_episodes', [])))
    baseline={};traces={};branches={}
    for line in (root/'capture_001/graph_options/graph_options.jsonl').open():
        r=json.loads(line);ep=str(r['episode_id'])
        if ep in fail:
            for o in r['options']:o.pop('embedding',None)
            baseline[(ep,r['high_level_step'])]=r
    for line in (root/'capture_001/traces/worker_seed100.jsonl').open():
        r=json.loads(line);ep=str(r['episode_id'])
        if ep in fail:traces.setdefault(ep,[]).append(r)
    required={(c['episode_id'],c['high_level_step'],c['action_index']) for c in plan['cases']}
    if required:
        for line in (root/cfg['probe']/'branch_traces.jsonl').open():
            r=json.loads(line);ep=str(r['decision_key'][1]);k=(ep,r['decision_key'][2],r['index'])
            if r['order']=='forward' and k in required:branches[k]=r['trace']
    if set(branches)!=required:raise ValueError('Missing isolated branch evidence')
    baseline_metrics=json.loads((root/cfg['noninterference']/'capture_episodes.json').read_text())
    controls={};results=[];started=time.monotonic()
    class Progress:
        def update(self,*args,**kwargs):pass
    trainer.pbar=Progress()
    def execute(case):
        ep=case['episode_id'];caseout=out/'cases'/case['case_id'];trainer.envs.resume_all()
        trainer.envs.call_at(0,'configure_case',{'case':case,'output':str(caseout),'initial_rng':traces[ep][0]['pre_rng']})
        trainer._graph_option_capture=CriticalHook(case,baseline,caseout/'graph')
        trainer.stat_eps={};t=time.monotonic();native_rollout('eval')
        normalized_metrics={str(k):v for k,v in trainer.stat_eps.items()}
        m=normalized_metrics[ep];rows=[json.loads(l) for l in (caseout/'trace.jsonl').open()]
        control=case['mode']=='control';step=case.get('high_level_step',len(rows));checks=[]
        neutral=(case['mode']=='action' and case['action_index']==case['effective_index'])
        if not control and not trainer._graph_option_capture.encountered:raise ValueError('Intervention state missed')
        for i,r in enumerate(rows):
            if control or neutral or case['mode']=='sense_only' or i<step:
                expected=traces[ep][i]
            elif case['mode']=='action' and i==step:
                expected=branches[(ep,step,case['action_index'])]
            else:continue
            c=compare(expected,r,gates)
            if not c['passed'] or not c['sensor_hashes_exact']:raise ValueError('Physical/RNG/sensor replay failed: '+str(c))
            checks.append(c)
        if control or neutral or case['mode']=='sense_only':
            if set(m)!=set(baseline_metrics[ep]) or any(abs(m[k]-baseline_metrics[ep][k])>1e-6 for k in m):raise ValueError('Control metrics differ')
        interrupt=case['mode'].startswith('interrupt')
        if interrupt:
            r=rows[step];old=traces[ep][step];cut=case['cut_primitive']
            if (not r['interrupted'] or r['primitives']!=old['primitives'][:cut]
                    or r['pre_rng']!=old['pre_rng'] or r['post_rng']!=old['post_rng']
                    or r['action']!=old['action']):raise ValueError('Interrupted physical prefix or RNG changed')
            if not rows[step]['sensed_at_cut']:raise ValueError('Missing intermediate sensing')
        recon=reconstruction(rows,trainer.gt_data[ep]['locations'],m)
        row={'case':case,'metrics':m,'baseline_metrics':baseline_metrics[ep],
             'delta':{k:m[k]-baseline_metrics[ep][k] for k in m},'prefix_and_branch_checks':len(checks),
             'metric_reconstruction':recon,'validity_passed':True,'runtime_seconds':time.monotonic()-t,
             'primitive_collision_events':sum(p['collided'] is True for r in rows for p in r['primitives']),
             'observation_at_calls':sum(r['observation_at_calls'] for r in rows),
             'cut_sensor_hashes':rows[step]['cut_sensor_hashes'] if 'cut_primitive' in case else None,
             'extra_sensing_calls':sum(r['sensed_at_cut'] for r in rows),'policy_calls':len(rows)}
        (caseout/'result.json').write_text(json.dumps(row,indent=2)+'\n')
        with (out/'results.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        results.append(row)
        status={'status':'running','completed':len(results),'native_cases_completed':sum(x['case']['mode']=='action' for x in results),'native_cases_planned':len(plan['cases']),'elapsed_seconds':time.monotonic()-started,'last_case':case['case_id']}
        (out/'status.json').write_text(json.dumps(status,indent=2)+'\n')
        print('case={} sr={} ndtw={:.4f} seconds={:.2f}'.format(case['case_id'],m['success'],m['ndtw'],row['runtime_seconds']),flush=True)
        return rows
    try:
        for ep in sorted(fail):controls[ep]=execute({'case_id':'control_'+ep,'episode_id':ep,'mode':'control'})
        if 'interrupt_events' in plan:
            # Frozen timing-oracle schedules may include successful native routes.
            # Validate every cut against the newly reproduced native controls.
            events=plan['interrupt_events'];seen=set()
            for event in events:
                ep=event['episode_id'];step=event['high_level_step'];cut=event['cut_primitive']
                rec=controls[ep][step];key=(ep,step,cut)
                identity=(ep,step,cut,event.get('cut_kind'),event.get('timing_seed'))
                if identity in seen or rec['action']['act']!=4 or not 0<cut<len(rec['primitives']):
                    raise ValueError('Invalid or duplicate scheduled interrupt: '+str(key))
                if rec['phases'][cut-1]!='ghost' or len(rec['primitives'])!=event['baseline_primitive_count']:
                    raise ValueError('Scheduled ghost phase/count mismatch: '+str(key))
                seen.add(identity)
            if manifest['smoke']:
                smoke_keys={tuple(k) for k in plan['smoke_event_keys']}
                events=[e for e in events if (e['episode_id'],e['high_level_step'],e['cut_primitive'],e.get('cut_kind'),e.get('timing_seed')) in smoke_keys]
                if len(events)!=len(smoke_keys):raise ValueError('Missing scheduled smoke event')
        else:
            events=find_interrupts(controls,plan['states'],cfg['max_interrupt_routes'])
        (out/'interrupt_plan.json').write_text(json.dumps({'events':events,'frozen_before_intervention':True,'rule':cfg['protocol']},indent=2)+'\n')
        # Confirmation timing arms run after all baseline controls, while the same
        # checkpoint remains loaded; full-return census follows unchanged plan.
        for event in events:
            sensing=None
            for mode in ['sense_only','interrupt_consume']:
                suffix='_{}_p{}'.format(event.get('timing_seed','event'),event['cut_primitive']) if 'interrupt_events' in plan else ''
                case=dict(event,mode=mode,case_id='{}_{}_s{}{}'.format(mode,event['episode_id'],event['high_level_step'],suffix))
                execute(case)
                hashes=results[-1]['cut_sensor_hashes']
                if mode=='sense_only':sensing=hashes
                elif hashes!=sensing:raise ValueError('Intermediate observations differ across arms')
        cases=plan['cases']
        if manifest['smoke']:
            # Exercise native selected, STOP override, non-STOP replacement,
            # and overriding a learned STOP without changing forced-stop masks.
            take=[]
            for predicate in [lambda c:c['action_index']==c['effective_index'],lambda c:c['action_index']==0 and c['effective_index']>0,lambda c:c['action_index']>0 and c['effective_index']>0 and c['action_index']!=c['effective_index'],lambda c:c['action_index']>0 and c['effective_index']==0]:
                matching=[c for c in cases if predicate(c)]
                if matching:take.append(matching[0])
            cases=take
        for case in cases:execute(case)
        summary={'status':'smoke_complete' if manifest['smoke'] else 'complete','controls':len(controls),'interrupt_events':events,'native_cases_planned':len(plan['cases']),'native_cases_completed':len(cases),'all_validity_checks_passed':True,'no_model_trained':True,'oracle_analysis_only':True,'elapsed_seconds':time.monotonic()-started,'scope':plan['scope']}
        (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');(out/'status.json').write_text(json.dumps(summary,indent=2)+'\n')
        # Outer evaluator reports a baseline control only. All counterfactuals
        # are stored in results.jsonl with explicit intervention mode labels.
        ep=sorted(fail)[0];trainer.stat_eps={ep:baseline_metrics[ep]}
    except Exception as exc:
        (out/'FAILED.json').write_text(json.dumps({'error':repr(exc),'completed_cases':len(results)},indent=2)+'\n')
        trainer.envs.close()
        raise
