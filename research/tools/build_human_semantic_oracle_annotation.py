#!/usr/bin/env python3
"""Freeze an outcome-blind human directional-semantic audit sample.

This tool reads only the frozen pre-action graph/trace capture and optional
frozen semantic preannotations.  It never reads causal pair outcomes.
"""
import argparse, hashlib, json, math, re
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
LANDMARKS = ('sofa','couch','painting','door','doorway','table','chair','bed','kitchen','bedroom','hallway','corridor','stairs','staircase','rug','archway','podium','seat','pew','dining','sink','fireplace','cabinet')

def load_rows(path): return [json.loads(x) for x in Path(path).open() if x.strip()]
def arr(x): return x.get('__array__', x) if isinstance(x,dict) else x
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()
def key(r): return (str(r['episode_id']),int(r['high_level_step']))

def qmul(a,b):
    ax,ay,az,aw=a; bx,by,bz,bw=b
    return [aw*bx+ax*bw+ay*bz-az*by,aw*by-ax*bz+ay*bw+az*bx,
            aw*bz+ax*by-ay*bx+az*bw,aw*bw-ax*bx-ay*by-az*bz]
def inv_rotate(q,v):
    qc=[-float(q[0]),-float(q[1]),-float(q[2]),float(q[3])]
    return qmul(qmul(qc,[float(v[0]),float(v[1]),float(v[2]),0]),[float(q[0]),float(q[1]),float(q[2]),float(q[3])])[:3]
def bearing(pre_pose, origin, target):
    d=[float(target[i])-float(origin[i]) for i in range(3)]
    v=inv_rotate(pre_pose['rotation'],d)
    return math.degrees(math.atan2(-v[0],-v[2]))%360.
def sector(b):
    return min(range(0,360,30),key=lambda x:min((x-b)%360.,(b-x)%360.))
def phrases(text):
    hits=[]
    for name in LANDMARKS:
        for m in re.finditer(r'\b'+re.escape(name)+r'\b',(text or '').lower()):
            left=(text or '')[max(0,m.start()-45):m.start()].lower(); verb='observe'
            if re.search(r'\bpass(?:ed|ing)?\b|\baround\b',left):verb='pass'
            elif re.search(r'\bturn\b',left):verb='turn at'
            elif re.search(r'\benter(?:ed|ing)?\b|\bthrough\b|\binto\b',left):verb='enter'
            elif re.search(r'\bstop\b|\bstand\b',left):verb='stop at'
            hits.append((m.start(),verb+' '+name))
    out=[]
    for _,p in sorted(hits):
        if p not in out:out.append(p)
    return out or [text.strip() or 'unknown destination']

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--capture',type=Path,default=ROOT/'research/results/semantic_node_memory_multiview_001');ap.add_argument('--semantic',type=Path,default=ROOT/'research/results/semantic_node_memory_features_004/semantic_node_memory.jsonl');ap.add_argument('--output',type=Path,default=ROOT/'research/annotations/directional_semantic_oracle_v1');ap.add_argument('--states-per-scene',type=int,default=12);a=ap.parse_args()
    out=a.output; out.mkdir(parents=True,exist_ok=True); (out/'images').mkdir(exist_ok=True); (out/'contact_sheets').mkdir(exist_ok=True)
    graphs=load_rows(a.capture/'graph_options/graph_options.jsonl'); traces={key(x):x for x in load_rows(a.capture/'traces/worker_seed100.jsonl')}
    sem={key(x):x for x in load_rows(a.semantic)}
    by=defaultdict(list)
    for g in graphs: by[g['scene_id']].append(g)
    scene_order=sorted(by,key=lambda s:hashlib.sha256(s.encode()).hexdigest())
    selected=scene_order[:4]; dev=selected[0]; tests=selected[1:]
    split={'development_scene':dev,'test_scenes':tests,'selection_rule':'Freeze before annotation: SHA-256 lexicographic order of scene_id; first four scenes only. No outcome, BENEFIT/HARM, evaluator metric, or semantic annotation was read.','selected_before_annotation':True,'scene_order':scene_order,'created_utc':datetime.now(timezone.utc).isoformat()}
    (out/'scene_split.json').write_text(json.dumps(split,indent=2)+'\n')
    selected_states=[]; sectors=[]
    for si,scene in enumerate(selected):
        candidates=sorted(by[scene],key=lambda g:hashlib.sha256((str(g['episode_id'])+':'+str(g['high_level_step'])).encode()).hexdigest())
        chosen=candidates[:a.states_per_scene]
        for g in chosen:
            k=key(g); t=traces[k]; state_id='{}__ep{}__step{:03d}__node{}'.format(scene.split('/')[-2],k[0],k[1],g['current_vp'])
            if k not in sem or sem[k].get('source_observation_hashes') != g['semantic_visual']['observation_hashes']:
                raise ValueError('preannotation hash mismatch for {}'.format(k))
            selected_states.append({'state_id':state_id,'scene_id':scene,'episode_id':k[0],'high_level_step':k[1],'node_id':str(g['current_vp']),'instruction':g['instruction'],'candidate_phrases':phrases(g['instruction']),'graph_ids':g['graph_ids'],'state_selection_rank':candidates.index(g),'development':scene==dev})
            bysec={}
            for o in g['options']:
                if not o['admissible'] or int(o['action']['act'])==0:continue
                b=bearing(t['pre_pose'],arr(g['graph_position']),arr(o['action']['ghost_pos'])); sec=sector(b)
                # Native logits are intentionally omitted from the annotator
                # artifact.  They are not needed to label visual evidence.
                bysec.setdefault(sec,[]).append({'index':int(o['index']),'graph_id':o.get('graph_id'),'bearing_deg':b})
            files=g['semantic_visual']['files']
            for sec,actions in sorted(bysec.items()):
                sensor='rgb' if sec==0 else 'rgb_'+str(sec); src=Path(files[sensor]); name=state_id+'__bearing{:03d}'.format(sec); png=out/'images'/(name+'.png')
                im=Image.fromarray(__import__('numpy').load(str(src),allow_pickle=False)[...,:3]).convert('RGB'); im.save(str(png),format='PNG',optimize=True)
                sectors.append({'sector_id':name,'state_id':state_id,'scene_id':scene,'episode_id':k[0],'high_level_step':k[1],'node_id':str(g['current_vp']),'bearing_deg':sec,'image_path':str(png.relative_to(out)),'artifact_image_path':str(png.relative_to(ROOT)),'image_sha256':sha(png),'source_npy':str(src),'instruction':g['instruction'],'candidate_phrases':phrases(g['instruction']),'candidate_actions':actions,'development':scene==dev,'pre_action_only':True})
            # Contact sheet for all sampled action-bearing sectors in this state.
            ims=[]
            for q in [x for x in sectors if x['state_id']==state_id] if False else []: pass
            state_secs=[x for x in sectors if x['scene_id']==scene and x['episode_id']==k[0] and x['high_level_step']==k[1]]
            tile=[]
            for q in state_secs:
                img=Image.open(out/q['image_path']).resize((224,168)); tile.append((q['bearing_deg'],img))
            if not tile:
                continue
            sheet=Image.new('RGB',(4*224,math.ceil(len(tile)/4)*198),'white'); d=ImageDraw.Draw(sheet)
            for j,(b,img) in enumerate(tile): x=(j%4)*224;y=(j//4)*198;sheet.paste(img,(x,y+25));d.text((x+4,y+4),'bearing {}°'.format(b),fill='black')
            sheet.save(str(out/'contact_sheets'/(state_id+'.png')),format='PNG')
    (out/'sampled_states.jsonl').write_text(''.join(json.dumps(x,sort_keys=True)+'\n' for x in selected_states))
    (out/'sampled_sectors.jsonl').write_text(''.join(json.dumps(x,sort_keys=True)+'\n' for x in sectors))
    (out/'sampling_manifest.json').write_text(json.dumps({
        'capture':str(a.capture.relative_to(ROOT)),
        'capture_graph_sha256':sha(a.capture/'graph_options/graph_options.jsonl'),
        'capture_trace_sha256':sha(a.capture/'traces/worker_seed100.jsonl'),
        'semantic_preannotation_source':str(a.semantic.relative_to(ROOT)),
        'states_per_scene':a.states_per_scene,'selected_states':len(selected_states),
        'unique_sectors':len(sectors),'selection_uses_outcomes':False,
        'action_filter':'admissible native non-STOP options, one nearest 30-degree sector per state',
        'scene_split_file':'scene_split.json'},indent=2)+'\n')
    # Existing Qwen rows are copied only as preannotation; they are not labels.
    pre=[]
    for q in sectors:
        s=sem.get((q['episode_id'],q['high_level_step']),{}); sec=q['bearing_deg']
        lm=[x['name'] for x in s.get('landmarks',[]) if x.get('bearing_deg') is not None and min((float(x['bearing_deg'])-sec)%360.,(sec-float(x['bearing_deg']))%360.)<=30]
        rel=[x.get('relation','uncertain') for x in s.get('relations',[]) if x.get('bearing_deg') is not None and min((float(x['bearing_deg'])-sec)%360.,(sec-float(x['bearing_deg']))%360.)<=30]
        ev=[x for x in s.get('instruction_evidence',[]) if x.get('bearing_deg') is not None and min((float(x['bearing_deg'])-sec)%360.,(sec-float(x['bearing_deg']))%360.)<=30]
        pre.append({'sector_id':q['sector_id'],'scene_id':q['scene_id'],'episode_id':q['episode_id'],'high_level_step':q['high_level_step'],'node_id':q['node_id'],'bearing_deg':sec,'image_sha256':q['image_sha256'],'landmarks':lm,'place':s.get('place',{}).get('label',''),'navigable_opening':'uncertain','relation':rel[0] if rel else 'uncertain','instruction_support':'yes' if any(float(x.get('score',0))*float(x.get('confidence',0))>0 for x in ev) else 'uncertain','source':'frozen_Qwen_preannotation_only','ground_truth':False})
    (out/'preannotations.jsonl').write_text(''.join(json.dumps(x,sort_keys=True)+'\n' for x in pre))
    protocol='''# Directional semantic oracle annotation protocol\n\nProtocol frozen: {now}\n\nScene split is frozen in `scene_split.json`; development is used only to test UI and join alignment. Formal test scenes are never used to change ontology.\n\nSampling uses only frozen pre-action graph options. For each selected state, every unique nearest 30° sector containing at least one admissible non-STOP native action is sampled once. STOP has no image sector. Selection is deterministic by SHA-256 state key, taking the first {n} states per selected scene.\n\nAnnotators see one current pre-action RGB sector, the instruction and candidate phrase. They must not see endpoint images, goal/reference route, evaluator metrics, BENEFIT/HARM labels, or native/alternative outcomes.\n\nLabels: landmark multi-select (doorway, door, sofa/couch, painting, table, chair, bed, stairs, hallway/corridor, archway, podium, pew, kitchen, bedroom, other, none, uncertain); navigability (navigable opening, not navigable, uncertain, not applicable); relation (ahead, left, right, behind, doorway/opening, corridor continues, room entrance, landmark cue only, none, uncertain); instruction support (yes/no/uncertain) for visible evidence in the current frame only; confidence (high/medium/low). Temporal completion is never annotated.\n\nJoin key: scene_id, episode_id, high_level_step, node_id, bearing_deg, image_sha256.\n'''.format(now=datetime.now(timezone.utc).isoformat(),n=a.states_per_scene)
    (out/'ANNOTATION_PROTOCOL.md').write_text(protocol)
    build_html(out,sectors,pre)
    print(json.dumps({'development_scene':dev,'test_scenes':tests,'states':len(selected_states),'unique_sectors':len(sectors),'scene_counts':{s:sum(x['scene_id']==s for x in sectors) for s in selected}},indent=2))

def build_html(out,sectors,pre):
    data=json.dumps({'sectors':sectors,'pre':{x['sector_id']:x for x in pre}},separators=(',',':'))
    html=r'''<!doctype html><meta charset="utf-8"><title>Directional Semantic Oracle v1</title>
<style>body{font:16px sans-serif;margin:0;background:#eee}main{display:grid;grid-template-columns:minmax(500px,2fr) minmax(360px,1fr);gap:18px;padding:18px}img{max-width:100%;max-height:70vh;display:block;margin:auto;background:#222}.panel{background:white;padding:16px;border-radius:8px}.row{margin:10px 0}label{display:inline-block;margin:3px 8px 3px 0}textarea{width:100%;height:50px}button{font-size:16px;padding:8px;margin:4px}.meta{color:#444;white-space:pre-wrap}</style>
<main><section class=panel><div id=counter></div><img id=img><div id=meta class=meta></div><button onclick="prev()">Previous</button><button onclick="next()">Next</button><button onclick="save()">Save JSONL</button><button onclick="clearOne()">Clear current</button></section><section class=panel><div id=pre class=meta></div><div class=row><b>Landmarks</b><div id=landmarks></div><label>other text <input id=other></label></div><div class=row><b>Navigability</b><div id=nav></div></div><div class=row><b>Relation</b><div id=rel></div></div><div class=row><b>Instruction support</b><div id=support></div></div><div class=row><b>Confidence</b><div id=conf></div></div></section></main>
<script>const DATA=__DATA__;let i=0,ans=JSON.parse(localStorage.oracleAnswers||'{}');const land=['doorway','door','sofa/couch','painting','table','chair','bed','stairs','hallway/corridor','archway','podium','pew','kitchen','bedroom','other','none','uncertain'];const nav=['navigable opening','not navigable','uncertain','not applicable'];const rel=['ahead','left','right','behind','doorway/opening','corridor continues','room entrance','landmark cue only','none','uncertain'];const yn=['yes','no','uncertain'];const cf=['high','medium','low'];function opts(id,a,multi){document.getElementById(id).innerHTML=a.map(x=>`<label><input type="${multi?'checkbox':'radio'}" name="${id}" value="${x}">${x}</label>`).join('')}opts('landmarks',land,true);opts('nav',nav,false);opts('rel',rel,false);opts('support',yn,false);opts('conf',cf,false);function cur(){return DATA.sectors[i]}function read(){let q=cur(),a=ans[q.sector_id]||{};document.querySelectorAll('input').forEach(x=>x.checked=!!((Array.isArray(a.landmarks)&&a.landmarks.includes(x.value))||a[x.name]===x.value));document.getElementById('other').value=a.other||''}function write(){let q=cur(),a={sector_id:q.sector_id,scene_id:q.scene_id,episode_id:q.episode_id,high_level_step:q.high_level_step,node_id:q.node_id,bearing_deg:q.bearing_deg,image_sha256:q.image_sha256,landmarks:[...document.querySelectorAll('#landmarks input:checked')].map(x=>x.value),navigable_opening:(document.querySelector('#nav input:checked')||{}).value||'uncertain',relation:(document.querySelector('#rel input:checked')||{}).value||'uncertain',instruction_support:(document.querySelector('#support input:checked')||{}).value||'uncertain',annotator_confidence:(document.querySelector('#conf input:checked')||{}).value||'low',other:document.getElementById('other').value};ans[q.sector_id]=a;localStorage.oracleAnswers=JSON.stringify(ans)}function render(){let q=cur(),p=DATA.pre[q.sector_id]||{};document.getElementById('counter').textContent=`${i+1}/${DATA.sectors.length}  ${q.scene_id.split('/').slice(-2,-1)[0]}  ep ${q.episode_id} step ${q.high_level_step} bearing ${q.bearing_deg}°`;document.getElementById('img').src=q.image_path;document.getElementById('meta').textContent=`Instruction: ${q.instruction}\nCandidate phrases: ${q.candidate_phrases.join(' | ')}\nCandidate actions: ${q.candidate_actions.map(x=>x.index+': '+x.bearing_deg.toFixed(1)+'°').join(', ')}`;document.getElementById('pre').textContent='Qwen preannotation (not ground truth): '+JSON.stringify(p);read()}function prev(){write();i=(i+DATA.sectors.length-1)%DATA.sectors.length;render()}function next(){write();i=(i+1)%DATA.sectors.length;render()}function clearOne(){delete ans[cur().sector_id];localStorage.oracleAnswers=JSON.stringify(ans);render()}function save(){write();let lines=DATA.sectors.map(q=>ans[q.sector_id]).filter(Boolean).map(x=>JSON.stringify(x)).join('\n')+'\n';let b=new Blob([lines],{type:'application/jsonl'}),u=URL.createObjectURL(b),a=document.createElement('a');a.href=u;a.download='annotations.jsonl';a.click();URL.revokeObjectURL(u)}render()</script>'''
    (out/'index.html').write_text(html.replace('__DATA__',data))

if __name__=='__main__': main()
