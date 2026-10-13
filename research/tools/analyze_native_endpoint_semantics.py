#!/usr/bin/env python3
import argparse, json, re
from pathlib import Path
import numpy as np
import torch, clip
from PIL import Image

LANDMARKS=('sofa','couch','painting','door','doorway','table','chair','bed','kitchen','bedroom','hallway','corridor','stairs','staircase','rug','archway','podium','seat','pew','dining')

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--capture',type=Path,required=True); ap.add_argument('--enum',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); a=ap.parse_args()
    graph={(str(r['episode_id']),int(r['high_level_step'])):r for r in (json.loads(x) for x in (a.capture/'graph_options/graph_options.jsonl').open() if x.strip())}
    rows=[json.loads(x) for x in (a.enum/'native_endpoint_options.jsonl').open() if x.strip()]
    device='cuda' if torch.cuda.is_available() else 'cpu'; model,pre=clip.load('ViT-B/32',device=device,download_root=str(Path.home()/'.cache/clip')); model.eval()
    output=[]
    for r in rows:
        text=graph[(str(r['episode_id']),int(r['high_level_step']))].get('instruction',''); phrases=[p for p in LANDMARKS if re.search(r'\b'+re.escape(p)+r'\b',text.lower())]
        prompts=['a photo showing '+p for p in phrases] or ['a photo of a navigation scene']; image=pre(Image.fromarray(np.load(r['endpoint_files']['rgb'],allow_pickle=False)[...,:3])).unsqueeze(0).to(device); tokens=clip.tokenize(prompts).to(device)
        with torch.no_grad():
            im=model.encode_image(image).float(); im/=im.norm(dim=-1,keepdim=True); tx=model.encode_text(tokens).float(); tx/=tx.norm(dim=-1,keepdim=True); scores=(im@tx.T)[0].cpu().tolist()
        r['instruction_phrases']=phrases; r['endpoint_phrase_score']=float(max(scores)); r['endpoint_best_phrase']=phrases[int(np.argmax(scores))] if phrases else None; output.append(r)
    groups={}
    for r in output: groups.setdefault((r['episode_id'],r['high_level_step']),[]).append(r)
    diagnostics=[]
    for key,items in groups.items():
        items.sort(key=lambda x:x['endpoint_phrase_score'],reverse=True); selected=next(x for x in items if x['selected_native']); rank=1+next(i for i,x in enumerate(items) if x['selected_native']); diagnostics.append({'episode_id':key[0],'high_level_step':key[1],'actions':len(items),'selected_native_rank_by_endpoint_semantics':rank,'selected_score':selected['endpoint_phrase_score'],'best_score':items[0]['endpoint_phrase_score'],'selected_is_best':rank==1})
    summary={'states':len(groups),'action_branches':len(output),'selected_native_semantic_top1_rate':sum(x['selected_is_best'] for x in diagnostics)/len(diagnostics),'mean_selected_rank':sum(x['selected_native_rank_by_endpoint_semantics'] for x in diagnostics)/len(diagnostics),'diagnostic_scope':'CLIP endpoint probe only; no outcome labels or policy changes'}
    a.output.mkdir(parents=True,exist_ok=True); (a.output/'endpoint_semantics.jsonl').write_text(''.join(json.dumps(x,sort_keys=True)+'\n' for x in output)); (a.output/'diagnostics.json').write_text(json.dumps({'summary':summary,'states':diagnostics},indent=2)+'\n'); print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
