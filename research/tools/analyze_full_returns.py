#!/usr/bin/env python3
"""Full-return vectors for both frozen alternatives, with no fitted reward."""
import argparse,json,collections,math
from pathlib import Path
from audit_single_intervention import read_rows,group_rows,one
from analyze_continuation_horizon import window
from run_option_capture import digest

DIRECTIONS={'success':1,'spl':1,'ndtw':1,'distance_to_goal':-1,'path_length':-1,'primitive_action_count':-1}

def classify(delta):
    if not set(DIRECTIONS).issubset(delta):
        raise ValueError('Incomplete primary return vector')
    values=[delta[k]*direction for k,direction in DIRECTIONS.items()]
    if not all(math.isfinite(v) for v in values):
        raise ValueError('Nonfinite primary return vector')
    if int(delta['primitive_action_count']) != delta['primitive_action_count']:
        raise ValueError('Primitive cost must be integer')
    better=any(v>1e-6 for v in values);worse=any(v< -1e-6 for v in values)
    return 'mixed' if better and worse else 'dominates' if better else 'dominated' if worse else 'tied'

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--tag',required=True);p.add_argument('--output-dir',type=Path,required=True);args=p.parse_args()
    root=args.root;cfg=json.loads((root/'capture_001/manifest.json').read_text())['config']
    source_trace=one(root/'capture_001/traces','*.jsonl');base=group_rows(read_rows(source_trace))
    rows=[];groups={};hashes={str(Path(__file__)):digest(Path(__file__)),str(source_trace):digest(source_trace)}
    for kind in ['top_logit','seeded_graph_id']:
        ad=root/(kind+'_enabled_audit_'+args.tag);run=root/(kind+'_enabled_'+args.tag)
        audit=json.loads((ad/'summary.json').read_text())
        if not audit['all_required_gates_passed']:raise ValueError('Failed audit')
        recon=root/(kind+'_native_reconstruction_'+args.tag)/'summary.json'
        if not json.loads(recon.read_text())['all_required_gates_passed']:raise ValueError('Failed reconstruction')
        schedule_path=root/('schedule_'+kind+'_001.json');schedule=json.loads(schedule_path.read_text())
        current_path=one(run/'traces','*.jsonl');current=group_rows(read_rows(current_path))
        pairs={r['episode_id']:r for r in json.loads((ad/'paired_episodes.json').read_text())}
        for event in schedule['events']:
            ep,step=event['episode_id'],event['high_level_step'];pair=pairs[ep]
            a,b=base[ep],current[ep]
            horizons={}
            for h in [1,2]:
                old,new=window(a,step,h),window(b,step,h)
                horizons[str(h)]={'extra_goal_progress_m':old['distance_to_goal']-new['distance_to_goal'],
                  'primitive_delta':new['primitive_count']-old['primitive_count'],
                  'path_delta_m':new['movement_m']-old['movement_m'],
                  'both_terminal':old['terminal'] and new['terminal']}
            rows.append({'kind':kind,'scene_id':event['scene_id'],'episode_id':ep,'step':step,
                         'alternative_index':event['alternative_index'],'category':classify(pair['delta']),
                         'baseline':pair['baseline'],'alternative':pair['intervention'],'delta':pair['delta'],
                         'horizons':horizons,
                         'baseline_stop_primitives':len(a[-1]['primitives']),
                         'alternative_stop_primitives':len(b[-1]['primitives'])})
        groups[kind]=audit['paired_groups']
        for f in [ad/'summary.json',ad/'paired_episodes.json',recon,schedule_path,current_path]:hashes[str(f)]=digest(f)
    state_groups=collections.defaultdict(list)
    for r in rows:state_groups[(r['scene_id'],r['episode_id'],r['step'])].append(r)
    # Schedules must compare the same native state; candidates must be distinct.
    top={r['episode_id']:r for r in rows if r['kind']=='top_logit'}
    for r in rows:
        if r['kind']=='seeded_graph_id':
            if r['step']!=top[r['episode_id']]['step'] or r['alternative_index']==top[r['episode_id']]['alternative_index']:
                raise ValueError('Schedules differ in state or duplicate alternatives')
    counts={kind:dict(collections.Counter(r['category'] for r in rows if r['kind']==kind)) for kind in groups}
    dominating=[r for r in rows if r['category']=='dominates']
    scenes={r['scene_id'] for r in dominating};routes={r['episode_id'] for r in dominating}
    disagreements={'h1_positive_but_final_dominated':[], 'h1_negative_but_final_dominates':[],
                   'h2_positive_but_final_dominated':[], 'h2_negative_but_final_dominates':[]}
    for r in rows:
        for h in ['1','2']:
            gain=r['horizons'][h]['extra_goal_progress_m']
            if gain>1e-6 and r['category']=='dominated':disagreements['h'+h+'_positive_but_final_dominated'].append([r['kind'],r['episode_id']])
            if gain< -1e-6 and r['category']=='dominates':disagreements['h'+h+'_negative_but_final_dominates'].append([r['kind'],r['episode_id']])
    result={'experiment_id':cfg['experiment_id'],'status':'completed','analysis':'registered full-return dominance, no scalar reward',
      'population_routes':cfg['episodes'],'alternatives':len(rows),'sampled_states':len(state_groups),
      'category_counts':counts,'dominating_routes':len(routes),'dominating_scenes':len(scenes),
      'dominating_episode_ids':sorted(routes),'groups':groups,'events':rows,'short_window_disagreements':disagreements,
      'learning_support_gate_passed':len(scenes)>=6,'no_model_trained':True,
      'decision':'CONDITIONAL GO for a separate frozen-feature feasibility study' if len(scenes)>=6 else 'NO-GO for fitting on this pilot; support is too sparse',
      'limits':'Two outcome-blind candidates per selected baseline state; limited candidate coverage, one seed, training scenes. Dominance is an analysis label, not a deployable policy or exhaustive oracle upper bound.',
      'hashes':hashes}
    args.output_dir.mkdir(parents=True,exist_ok=False)
    (args.output_dir/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    table=['| Kind | Episode / step | Category | SR delta | SPL delta pp | nDTW delta pp | Primitive delta | H1 progress | H2 progress |',
           '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in rows:
        d=r['delta'];table.append('| {} | {} / {} | {} | {:+.0f} | {:+.3f} | {:+.3f} | {:+.0f} | {:+.3f} | {:+.3f} |'.format(
          r['kind'],r['episode_id'],r['step'],r['category'],d['success'],d['spl']*100,d['ndtw']*100,d['primitive_action_count'],
          r['horizons']['1']['extra_goal_progress_m'],r['horizons']['2']['extra_goal_progress_m']))
    (args.output_dir/'paired_table.md').write_text('\n'.join(table)+'\n')
    print(json.dumps({k:result[k] for k in ['alternatives','sampled_states','category_counts','dominating_routes','dominating_scenes','decision']},indent=2))

if __name__=='__main__':main()
