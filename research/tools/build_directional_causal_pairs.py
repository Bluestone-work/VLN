#!/usr/bin/env python3
"""Join the fail-closed causal audit with current-node semantic observations.

The join is deliberately separate from rollout collection.  It accepts a
semantic row only when its node, pose, and all 24 pre-action observation hashes
match the frozen source capture.  Alternative endpoint observations are never
read here.
"""
import argparse, json, math, hashlib, re
from pathlib import Path


def rows(p):
    return [json.loads(x) for x in Path(p).open() if x.strip()]


def arr(x):
    return x.get('__array__', x) if isinstance(x, dict) else x


def same(a, b):
    return a == b


def qmul(a, b):
    ax, ay, az, aw = a; bx, by, bz, bw = b
    return [aw*bx + ax*bw + ay*bz - az*by,
            aw*by - ax*bz + ay*bw + az*bx,
            aw*bz + ax*by - ay*bx + az*bw,
            aw*bw - ax*bx - ay*by - az*bz]


def inverse_rotate(q, v):
    # Habitat coefficients are [x,y,z,w].  graph_utils rotates with q^-1.
    qc = [-float(q[0]), -float(q[1]), -float(q[2]), float(q[3])]
    return qmul(qmul(qc, [float(v[0]), float(v[1]), float(v[2]), 0.0]),
                [float(q[0]), float(q[1]), float(q[2]), float(q[3])])[:3]


def relative_bearing(pre_pose, target, origin):
    d = [float(target[i]) - float(origin[i]) for i in range(3)]
    local = inverse_rotate(pre_pose['rotation'], d)
    # Positive degrees are local LEFT, matching the 12-view capture prompt.
    return math.degrees(math.atan2(-local[0], -local[2])) % 360.0


def nearest_sector(b):
    sectors = list(range(0, 360, 30))
    return min(sectors, key=lambda s: min((s-b) % 360., (b-s) % 360.))


def in_sector(b, x, width=30.):
    if x is None: return False
    d = abs((float(x)-float(b)+180.) % 360.-180.)
    return d <= width


def semantic_features(sem, bearing):
    # STOP has no endpoint direction and must not borrow the front sector.
    if bearing is None:
        return {'sector_deg': None, 'semantic_landmark_count': 0,
                'semantic_relation_count': 0, 'semantic_landmark_confidence': 0.,
                'semantic_relation_confidence': 0., 'instruction_evidence': 0.,
                'instruction_evidence_count': 0, 'semantic_global_confidence': 0.,
                'semantic_abstained': True, 'sector_landmarks':[], 'sector_relations':[], 'sector_evidence':[]}
    sec=nearest_sector(bearing)
    lm = [x for x in sem.get('landmarks', []) if in_sector(sec, x.get('bearing_deg'))]
    rel = [x for x in sem.get('relations', []) if in_sector(sec, x.get('bearing_deg'))]
    ev = [x for x in sem.get('instruction_evidence', []) if in_sector(sec, x.get('bearing_deg'))]
    return {'sector_deg': nearest_sector(bearing),
            'semantic_landmark_count': len(lm), 'semantic_relation_count': len(rel),
            'semantic_landmark_confidence': max([float(x.get('confidence', 0.)) for x in lm] or [0.]),
            'semantic_relation_confidence': max([float(x.get('confidence', 0.)) for x in rel] or [0.]),
            'instruction_evidence': max([float(x.get('score', 0.))*float(x.get('confidence', 0.)) for x in ev] or [0.]),
            'instruction_evidence_count': len(ev),
            'semantic_global_confidence': float(sem.get('global_confidence', 0.)),
            'semantic_abstained': not any(float(x.get('confidence',0))>0 for x in lm+rel+ev),
            'sector_landmarks':lm,'sector_relations':rel,'sector_evidence':ev}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--audit', type=Path, required=True)
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--semantic', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    a=ap.parse_args()
    audit = [x for x in json.loads((a.audit/'pairs_audit.json').read_text()) if x['pass']]
    raw_pairs = {x['pair_id']: x for x in rows(a.audit/'pairs.jsonl')}
    graph = {(str(x['episode_id']),int(x['high_level_step'])):x for x in rows(a.source/'graph_options/graph_options.jsonl')}
    trace = {(str(x['episode_id']),int(x['high_level_step'])):x for x in rows(a.source/'traces/worker_seed100.jsonl')}
    semrows = rows(a.semantic/'semantic_node_memory.jsonl')
    sem = {(str(x['episode_id']),int(x['high_level_step'])):x for x in semrows}
    if len(sem) != len(semrows): raise ValueError('Duplicate semantic identity')
    phrases=json.loads((a.semantic/'ordered_phrases.json').read_text())
    from build_native_full_return_pairs import digest
    if json.loads((a.audit/'summary.json').read_text())['status']!='PASS': raise ValueError('Causal audit failed')
    if json.loads((a.semantic/'manifest.json').read_text())['graph_sha256']!=digest(a.source/'graph_options/graph_options.jsonl'): raise ValueError('Wrong semantic source graph')
    out=[]; join_audit=[]
    for audit_row in audit:
        p=raw_pairs[audit_row['pair_id']]; k=(str(p['episode_id']),int(p['high_level_step']))
        g=graph[k]; t=trace[k]; s=sem.get(k)
        checks={'semantic_present': s is not None,
                'node_id_exact': s is not None and str(s.get('node_id')) == str(g['current_vp']),
                'position_exact': s is not None and arr(s.get('position')) == arr(g['graph_position']),
                'observation_hashes_exact': s is not None and len(s.get('source_observation_hashes',{}))==24 and s.get('source_observation_hashes') == g['semantic_visual']['observation_hashes'],
                'no_privileged_input': s is not None and s.get('privileged_future_input_used') is False,
                'instruction_exact': phrases[k[0]]['instruction']==g['instruction']==g['semantic_visual']['instruction'],
                'phrases_exact': s is not None and [x['phrase'] for x in s['instruction_evidence']]==phrases[k[0]]['phrases']}
        if not all(checks.values()):
            join_audit.append({'pair_id':p['pair_id'],'checks':checks,'pass':False}); continue
        e=next(x for x in json.loads((a.audit.parent/'schedules'/('schedule_'+p['native_run'].replace('_disabled','')+'.json')).read_text())['events'] if str(x['episode_id'])==k[0] and int(x['high_level_step'])==k[1])
        # The event stores the exact alternative action/ghost position.
        act=e['alternative_action']; bearing=None
        if int(act['act']) != 0:
            bearing=relative_bearing(t['pre_pose'], arr(act['ghost_pos']), arr(g['graph_position']))
        opt=next(x for x in g['options'] if int(x['index'])==int(p['alternative_index']))
        native_opt=next(x for x in g['options'] if int(x['index'])==int(p['native_index']))
        sf=semantic_features(s,bearing)
        feat={'native_logit':float(opt['logit']), 'native_selected_logit':float(native_opt['logit']),
              'native_logit_margin':float(opt['logit'])-float(native_opt['logit']),
              'action_index':int(p['alternative_index']), 'is_stop':int(act['act'])==0,
              'bearing_deg':bearing, 'distance':0. if bearing is None else math.sqrt(sum((float(arr(act['ghost_pos'])[i])-float(arr(g['graph_position'])[i]))**2 for i in range(3))),
              'native_embedding':arr(opt.get('embedding', [])),
              'current_proposal':bool(opt['current_proposal']), 'graph_size':len(g['graph_ids']),
              'back_path_length':len(act['back_path']), **sf}
        out.append({**p, 'instruction':g['instruction'], 'current_vp':g['current_vp'],
                    'graph_ids':g['graph_ids'], 'pre_pose':t['pre_pose'],
                    'pre_observation_hashes':g['semantic_visual']['observation_hashes'], 'features':feat,
                    'semantic_provenance':{'source':str(a.semantic),'current_node_only':True,'alternative_endpoint_used':False}})
        join_audit.append({'pair_id':p['pair_id'],'checks':checks,'pass':True})
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'pairs.jsonl').write_text(''.join(json.dumps(x,sort_keys=True,separators=(',',':'))+'\n' for x in out))
    (a.output/'join_audit.json').write_text(json.dumps({'rows':len(join_audit),'passed':sum(x['pass'] for x in join_audit),'checks':join_audit},indent=2)+'\n')
    print(json.dumps({'pairs':len(out),'join_rows':len(join_audit),'passed':sum(x['pass'] for x in join_audit)},indent=2))
    if len(out)!=len(audit): raise SystemExit(1)

if __name__=='__main__': main()
