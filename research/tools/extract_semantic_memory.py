#!/usr/bin/env python3
"""Extract an analysis-only semantic-memory inventory from frozen traces."""
import argparse, gzip, hashlib, json, re
from pathlib import Path

LANDMARKS = ('sofa','couch','painting','door','doorway','table','chair','bed','kitchen','bedroom','hallway','corridor','stairs','staircase','rug','archway','podium','seat','pew','dining')

def digest(p):
    h=hashlib.sha256(); h.update(Path(p).read_bytes()); return h.hexdigest()

def main():
    p=argparse.ArgumentParser(); p.add_argument('--capture',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--dataset',type=Path)
    a=p.parse_args(); out=a.output; out.mkdir(parents=True,exist_ok=False)
    instructions={}
    if a.dataset:
        with gzip.open(a.dataset,'rt') as f: d=json.load(f)
        instructions={str(e['episode_id']):e.get('instruction',{}).get('instruction_text','') for e in d.get('episodes',[])}
    graph=[json.loads(x) for x in (a.capture/'graph_options/graph_options.jsonl').open()]
    trace=[json.loads(x) for x in (a.capture/'traces/worker_seed100.jsonl').open()]
    tr={(str(x['episode_id']),int(x['high_level_step'])):x for x in trace}
    rows=[]
    for g in graph:
        key=(str(g['episode_id']),int(g['high_level_step'])); t=tr.get(key); text=g.get('instruction') or instructions.get(key[0],'')
        phrases=[w for w in LANDMARKS if re.search(r'\b'+re.escape(w)+r'\b',text.lower())]
        node={'episode_id':key[0],'scene_id':g.get('scene_id'),'high_level_step':key[1],'node_id':g.get('current_vp'),'position':g.get('graph_position'),'instruction':text,'instruction_landmark_hypotheses':phrases,'visual_feature_status':'unavailable_no_panorama_join','place':None,'landmarks':[],'relations':[],'instruction_relevance':{'matched_phrase':None,'semantic_similarity':None,'likely_completed':None,'likely_next':None},'confidence':0.0,'node_stop_score':None,'native_logits':g.get('logits'),'privileged_future_input_used':False}
        rows.append(node)
        for o in g.get('options',[]):
            if not o.get('admissible',False) or o.get('action',{}).get('act')==0: continue
            rows.append({'episode_id':key[0],'scene_id':g.get('scene_id'),'high_level_step':key[1],'node_id':o.get('graph_id'),'position':o.get('action',{}).get('ghost_pos'),'instruction':text,'instruction_landmark_hypotheses':phrases,'visual_feature_status':'unavailable_no_panorama_join','place':None,'landmarks':[],'relations':[],'instruction_relevance':{'matched_phrase':None,'semantic_similarity':None,'likely_completed':None,'likely_next':None},'confidence':0.0,'native_logit':o.get('logit'),'privileged_future_input_used':False})
    with (out/'semantic_memory.jsonl').open('w') as f:
        for r in rows: f.write(json.dumps(r,sort_keys=True)+'\n')
    summary={'records':len(rows),'decision_nodes':len(graph),'visual_aligned_records':0,'missing_visual_alignment_records':len(rows),'instruction_hypothesis_records':sum(bool(r['instruction_landmark_hypotheses']) for r in rows),'semantic_evidence_records':0,'status':'NO-GO_ALIGNMENT_MISSING','capture_sha256':digest(a.capture/'graph_options/graph_options.jsonl')}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n'); (out/'manifest.json').write_text(json.dumps({'capture':str(a.capture),'dataset':str(a.dataset) if a.dataset else None,'semantic_encoder':'none_no_valid_panorama_join','raw_outcomes_not_used':True},indent=2)+'\n')
    print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
