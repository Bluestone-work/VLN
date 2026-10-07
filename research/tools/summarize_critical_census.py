#!/usr/bin/env python3
"""Tie complete critical-state coverage to overlapping full-return rescue sets."""
import argparse,collections,json
from pathlib import Path
from run_option_capture import digest


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    summary=json.loads((a.run/'summary.json').read_text());manifest=json.loads((a.run/'manifest.json').read_text());plan=manifest['plan']
    if summary['status']!='complete' or not summary['all_validity_checks_passed']:raise ValueError('Full run not accepted')
    rows=[json.loads(x) for x in (a.run/'results.jsonl').open()];byid={r['case']['case_id']:r for r in rows}
    if len(byid)!=len(rows):raise ValueError('Duplicate case')
    actions=[r for r in rows if r['case']['mode']=='action'];expected={c['case_id'] for c in plan['cases']}
    if {r['case']['case_id'] for r in actions}!=expected:raise ValueError('Not exhaustive over frozen cases')
    if any(not r['validity_passed'] or not r['metric_reconstruction']['passed'] for r in rows):raise ValueError('Validity failed')
    routes={};states=[]
    for state in plan['states']:
        selected=[r for r in actions if r['case']['episode_id']==state['episode_id'] and r['case']['high_level_step']==state['high_level_step']]
        if len(selected)!=state['actions']:raise ValueError('State coverage incomplete')
        rescue=[r for r in selected if r['metrics']['success']]
        states.append(dict(state,successful_actions=[r['case']['action_index'] for r in rescue],
                           ndtw_nondegrading_successful_actions=[r['case']['action_index'] for r in rescue if r['delta']['ndtw']>=-1e-6],
                           best_ndtw=max(r['metrics']['ndtw'] for r in selected)))
    for ep in plan['failed_episodes']:
        rr=[r for r in actions if r['case']['episode_id']==ep];rescue=[r for r in rr if r['metrics']['success']]
        ranking=[r for r in rescue if r['case']['effective_index']>0 and r['case']['action_index']>0]
        stopping=[r for r in rescue if r['case']['effective_index']==0 or r['case']['action_index']==0]
        inter=[r for r in rows if r['case']['episode_id']==ep and r['case']['mode'].startswith('interrupt')]
        routes[ep]={'scene_id':rr[0]['case']['scene_id'],'critical_states':len({r['case']['high_level_step'] for r in rr}),
                    'native_actions':len(rr),'native_rescue_count':len(rescue),
                    'ranking_rescue_cases':[r['case']['case_id'] for r in ranking],
                    'termination_rescue_cases':[r['case']['case_id'] for r in stopping],
                    'ranking_ndtw_nondegrading_cases':[r['case']['case_id'] for r in ranking if r['delta']['ndtw']>=-1e-6],
                    'interrupt_tested':bool(inter),
                    'interrupt_rescue_cases':[r['case']['case_id'] for r in inter if r['metrics']['success']],
                    'unresolved_after_tested_interventions':not rescue and not any(r['metrics']['success'] for r in inter)}
    rank={ep for ep,r in routes.items() if r['ranking_rescue_cases']};term={ep for ep,r in routes.items() if r['termination_rescue_cases']};interrupt={ep for ep,r in routes.items() if r['interrupt_rescue_cases']};unknown={ep for ep,r in routes.items() if r['unresolved_after_tested_interventions']}
    robust_rank={ep for ep,r in routes.items() if r['ranking_ndtw_nondegrading_cases']}
    def describe(ids):return {'episodes':sorted(ids),'count':len(ids),'scenes':len({routes[ep]['scene_id'] for ep in ids}),'denominator':len(routes)}
    ir=[]
    for r in rows:
        if r['case']['mode'] in ['sense_only','interrupt_consume','interrupt_retain']:
            ir.append({'case':r['case'],'metrics':r['metrics'],'delta':r['delta'],'observation_at_calls':r['observation_at_calls'],'extra_sensing_calls':r['extra_sensing_calls'],'policy_calls':r['policy_calls']})
    tested_interrupt={ep for ep,r in routes.items() if r['interrupt_tested']}
    rank_evidence=describe(robust_rank);inter_evidence=describe(interrupt)
    inter_evidence.update(denominator=len(tested_interrupt),tested_episodes=sorted(tested_interrupt),
                          cohort_routes=len(routes),denominator_scope='routes with an executed interrupt arm')
    if len(robust_rank)>=2 and rank_evidence['scenes']>=2:
        decision='CONDITIONAL GO for a separately registered long-horizon graph-value/preference diagnostic; not permission to train on validation or claim a learned gain'
    elif len(interrupt)>=2 and inter_evidence['scenes']>=2:
        decision='CONDITIONAL GO for a larger independent interrupt confirmation; no deployable gain established'
    else:decision='NO-GO for learning from current evidence; unresolved failures must not be called proposal failures'
    result={'experiment_id':plan['config']['experiment_id'],'status':'complete','cohort':plan['config']['source_root'],
            'critical_states':len(states),'native_full_returns':len(actions),'control_routes':summary['controls'],
            'all_coverage_and_validity_gates_passed':True,'metric_reconstruction_comparisons':sum(r['metric_reconstruction']['comparisons'] for r in rows),
            'ranking_opportunity':describe(rank),'ranking_sr_and_ndtw_opportunity':rank_evidence,'termination_opportunity':describe(term),'interrupt_opportunity':inter_evidence,'unresolved':describe(unknown),
            'ranking_majority_supported':len(robust_rank)>len(routes)/2,'sr_only_ranking_majority_supported':len(rank)>len(routes)/2,'interrupt_majority_supported':len(interrupt)>len(tested_interrupt)/2,
            'routes':routes,'states':states,'interrupt_results':ir,'decision':decision,
            'limits':['Outcome-conditioned retrospective census; not unbiased benchmark performance or deployable oracle. Validation data must never become fitting labels.',
                      'One-step native choice replaced; future policy is native, not an exhaustive policy tree.',
                      'Ranking, termination and execution are coupled; rescue sets overlap, no forced causal partition.',
                      'No native rescue does not prove proposal coverage failure.',
                      'Interrupt oracle uses four frozen first-event cuts, not an optimal cut-time upper bound.',
                      'Fixed seed 100; no multi-seed learned claim.'],
            'hashes':{str(p):digest(p) for p in [a.run/'manifest.json',a.run/'summary.json',a.run/'results.jsonl',Path(__file__)]}}
    a.output.mkdir(parents=True,exist_ok=False);(a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Critical-state full-return census','', '| Route | States | Native actions | Ranking rescue | Termination rescue | Interrupt rescue | Unresolved |','| --- | ---: | ---: | --- | --- | --- | --- |']
    for ep,r in routes.items():lines.append('| {} | {} | {} | {} | {} | {} | {} |'.format(ep,r['critical_states'],r['native_actions'],bool(r['ranking_rescue_cases']),bool(r['termination_rescue_cases']),bool(r['interrupt_rescue_cases']) if r['interrupt_tested'] else 'not tested',r['unresolved_after_tested_interventions']))
    lines+=['',decision,'','These are overlapping opportunities, not mutually exclusive causal counts.']
    (a.output/'table.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['routes','states','interrupt_results','hashes']},indent=2))

if __name__=='__main__':main()
