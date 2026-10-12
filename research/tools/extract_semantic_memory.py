#!/usr/bin/env python3
"""Extract an analysis-only semantic-memory inventory from frozen traces."""
import argparse, gzip, hashlib, json, re
from pathlib import Path

LANDMARKS = ('sofa','couch','painting','door','doorway','table','chair','bed','kitchen','bedroom','hallway','corridor','stairs','staircase','rug','archway','podium','seat','pew','dining')

def digest(p):
    h=hashlib.sha256(); h.update(Path(p).read_bytes()); return h.hexdigest()

def main():
    p=argparse.ArgumentParser(); p.add_argument('--capture',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--dataset',type=Path); p.add_argument('--clip',action='store_true',help='run local CLIP image/text similarity on captured pre-action RGB')
    a=p.parse_args(); out=a.output; out.mkdir(parents=True,exist_ok=False)
    instructions={}
    if a.dataset:
        with gzip.open(a.dataset,'rt') as f: d=json.load(f)
        instructions={str(e['episode_id']):e.get('instruction',{}).get('instruction_text','') for e in d.get('episodes',[])}
    graph=[json.loads(x) for x in (a.capture/'graph_options/graph_options.jsonl').open()]
    trace=[json.loads(x) for x in (a.capture/'traces/worker_seed100.jsonl').open()]
    tr={(str(x['episode_id']),int(x['high_level_step'])):x for x in trace}
    clip_model = preprocess = None
    if a.clip:
        import numpy as np
        import torch
        from PIL import Image
        import clip
        clip_model, preprocess = clip.load('ViT-B/32', device='cuda' if torch.cuda.is_available() else 'cpu', download_root=str(Path.home()/'.cache/clip'))
        clip_model.eval()
        clip_device = next(clip_model.parameters()).device
        prompt_labels = list(LANDMARKS) + ['a hallway', 'a bedroom', 'a kitchen', 'a dining room', 'a living room', 'a staircase', 'a doorway', 'an outdoor path']
        prompt_text = clip.tokenize(['a photo of ' + x for x in prompt_labels]).to(clip_device)
        with torch.no_grad():
            prompt_features = clip_model.encode_text(prompt_text).float()
            prompt_features /= prompt_features.norm(dim=-1, keepdim=True)
    rows=[]
    visual_records=0
    for g in graph:
        key=(str(g['episode_id']),int(g['high_level_step'])); t=tr.get(key); text=g.get('instruction') or instructions.get(key[0],'')
        phrases=[w for w in LANDMARKS if re.search(r'\b'+re.escape(w)+r'\b',text.lower())]
        node={'episode_id':key[0],'scene_id':g.get('scene_id'),'high_level_step':key[1],'node_id':g.get('current_vp'),'position':g.get('graph_position'),'instruction':text,'instruction_landmark_hypotheses':phrases,'visual_feature_status':'unavailable_no_panorama_join','place':None,'landmarks':[],'relations':[],'instruction_relevance':{'matched_phrase':None,'semantic_similarity':None,'likely_completed':None,'likely_next':None},'confidence':0.0,'node_stop_score':None,'native_logits':g.get('logits'),'privileged_future_input_used':False}
        if a.clip and g.get('semantic_visual'):
            import numpy as np
            from PIL import Image
            import torch
            rgb = np.load(g['semantic_visual']['files']['rgb'], allow_pickle=False)[..., :3]
            image = preprocess(Image.fromarray(rgb)).unsqueeze(0).to(clip_device)
            phrase_text = clip.tokenize(['a photo showing ' + x for x in phrases] or ['a photo of a navigation scene']).to(clip_device)
            with torch.no_grad():
                image_feature = clip_model.encode_image(image).float(); image_feature /= image_feature.norm(dim=-1, keepdim=True)
                phrase_features = clip_model.encode_text(phrase_text).float(); phrase_features /= phrase_features.norm(dim=-1, keepdim=True)
                phrase_scores = (image_feature @ phrase_features.T)[0].cpu().tolist()
                broad_scores = (image_feature @ prompt_features.T)[0].cpu().tolist()
            scored = sorted(zip(phrases, phrase_scores), key=lambda x: x[1], reverse=True)
            broad = sorted(zip(prompt_labels, broad_scores), key=lambda x: x[1], reverse=True)
            node['visual_feature_status'] = 'clip_vit_b32_analysis_only'
            node['landmarks'] = [{'label': x, 'clip_similarity': float(s)} for x, s in broad[:5]]
            node['place'] = broad[0][0] if broad else None
            node['instruction_relevance'] = {'matched_phrase': scored[0][0] if scored else None, 'semantic_similarity': float(scored[0][1]) if scored else None, 'likely_completed': None, 'likely_next': None}
            node['confidence'] = float(max(0.0, min(1.0, ((scored[0][1] if scored else 0.0) + 1.0) / 2.0)))
            node['semantic_encoder'] = {'name': 'openai_clip', 'checkpoint': 'ViT-B/32', 'prompt_templates': ['a photo showing {instruction landmark}', 'a photo of {broad label}']}
            visual_records += 1
        rows.append(node)
        for o in g.get('options',[]):
            if not o.get('admissible',False) or o.get('action',{}).get('act')==0: continue
            rows.append({'episode_id':key[0],'scene_id':g.get('scene_id'),'high_level_step':key[1],'node_id':o.get('graph_id'),'position':o.get('action',{}).get('ghost_pos'),'instruction':text,'instruction_landmark_hypotheses':phrases,'visual_feature_status':'unavailable_no_panorama_join','place':None,'landmarks':[],'relations':[],'instruction_relevance':{'matched_phrase':None,'semantic_similarity':None,'likely_completed':None,'likely_next':None},'confidence':0.0,'native_logit':o.get('logit'),'privileged_future_input_used':False})
    with (out/'semantic_memory.jsonl').open('w') as f:
        for r in rows: f.write(json.dumps(r,sort_keys=True)+'\n')
    summary={'records':len(rows),'decision_nodes':len(graph),'visual_aligned_records':visual_records,'missing_visual_alignment_records':len(graph)-visual_records,'instruction_hypothesis_records':sum(bool(r['instruction_landmark_hypotheses']) for r in rows),'semantic_evidence_records':visual_records,'status':'ANALYSIS_ONLY_CLIP' if a.clip else 'NO-GO_ALIGNMENT_MISSING','capture_sha256':digest(a.capture/'graph_options/graph_options.jsonl')}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n'); (out/'manifest.json').write_text(json.dumps({'capture':str(a.capture),'dataset':str(a.dataset) if a.dataset else None,'semantic_encoder':'openai_clip_vit_b32_local' if a.clip else 'none_no_valid_panorama_join','raw_outcomes_not_used':True,'privileged_future_input_used':False},indent=2)+'\n')
    print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
