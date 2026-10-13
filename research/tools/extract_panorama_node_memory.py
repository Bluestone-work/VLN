#!/usr/bin/env python3
"""Frozen, outcome-blind structured extraction from existing pre-action captures."""
import argparse
import hashlib
import json
import re
from pathlib import Path

REVISION = '895c3a49bc3fa70a340399125c650a463535e71c'
PHRASE_PROMPT = '''Split this navigation instruction into ordered short navigation subgoals, preserving every landmark and spatial relation. Use only the instruction text. The last phrase must express the stopping destination/goal. Return ONLY JSON {"phrases": ["exit bedroom", "pass sofa", "turn left at painting", "stop in kitchen"]}. At most 8 phrases. Instruction: '''
VISUAL_PROMPT = '''These 12 numbered tiles show real views at one visited navigation node BEFORE action. Tile numbers are camera local bearings in degrees, positive LEFT: 0 front, 90 left, 180 behind, 270 right. They are not future viewpoints. Extract only visually supported facts; do not infer an object from its mention in the instruction. Return ONLY compact JSON with this schema:
{"place":{"label":"hallway","confidence":0.0},"landmarks":[{"name":"doorway","bearing_deg":0,"confidence":0.0}],"relations":[{"subject":"doorway","relation":"ahead","bearing_deg":0,"confidence":0.0}],"evidence":[{"phrase_index":0,"score":0.0,"confidence":0.0,"relation":"visible cue or unknown","bearing_deg":null}]}
List at most 6 objects, 3 spatial relations and exactly one evidence entry for every ordered phrase. Scores/confidences are 0 to 1; score means visible support for the phrase's required object AND spatial relation, NOT generic scene resemblance. Unknown evidence is score 0 and confidence 0. A still panorama cannot establish that an object was passed, a room was entered, or a turn completed: describe only visible cues, never claim completion. Say unknown for uncertain place or bearing. Do not output extra text.
Ordered instruction phrases: '''

LANDMARKS = ('sofa','couch','painting','door','doorway','table','chair','bed','kitchen',
             'bedroom','hallway','corridor','stairs','staircase','rug','archway','podium',
             'seat','pew','dining','sink','fireplace','cabinet')


def deterministic_phrases(text):
    """Extract ordered subgoals from instruction text only."""
    text = text or ''
    hits = []
    for name in LANDMARKS:
        for match in re.finditer(r'\b' + re.escape(name) + r'\b', text.lower()):
            left = text[max(0, match.start() - 45):match.start()].lower()
            verb = 'observe'
            if re.search(r'\bpass(?:ed|ing)?\b|\baround\b', left): verb = 'pass'
            elif re.search(r'\bturn\b', left): verb = 'turn at'
            elif re.search(r'\benter(?:ed|ing)?\b|\bthrough\b|\binto\b', left): verb = 'enter'
            elif re.search(r'\bstop\b|\bstand\b', left): verb = 'stop at'
            hits.append((match.start(), '{} {}'.format(verb, name)))
    hits.sort()
    result, seen = [], set()
    for _, phrase in hits:
        if phrase not in seen:
            result.append(phrase); seen.add(phrase)
    return result or [text.strip() or 'unknown destination']


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda: stream.read(1024*1024), b''): h.update(b)
    return h.hexdigest()


def parse_json(text):
    start, end = text.find('{'), text.rfind('}')
    if start < 0 or end < start: raise ValueError('no JSON object')
    return json.loads(text[start:end+1])


def unit(v):
    try:
        x = float(v)
        return max(0., min(1., x)) if x == x else 0.
    except (TypeError, ValueError): return 0.


def structural_embedding(row, dims=256):
    # Stable signed hashing of explicit structures, not RGB or policy tensors.
    tokens = ['place:' + str(row['place']['label'])]
    for obj in row['landmarks']:
        tokens.append('object:{}:{}'.format(obj['name'], obj['bearing_deg']))
    for rel in row['relations']:
        tokens.append('relation:{}:{}:{}'.format(rel['subject'], rel['relation'], rel['bearing_deg']))
    for item in row['instruction_evidence']:
        tokens.append('evidence:{}:{}'.format(item['phrase'], round(item['score'], 1)))
    vector = [0.] * dims
    for token in tokens:
        h = hashlib.sha256(token.encode()).digest()
        vector[int.from_bytes(h[:4], 'big') % dims] += 1. if h[4] % 2 else -1.
    norm = sum(v*v for v in vector)**.5
    return [v/norm if norm else 0. for v in vector]


def normalize(raw, phrases):
    bearings = set(range(0, 360, 30))
    def bearing(v):
        return v if isinstance(v, (int, float)) and v in bearings else None
    p = raw.get('place') or {}
    row = {'place': {'label': str(p.get('label', 'unknown')), 'confidence': unit(p.get('confidence'))},
           'landmarks': [], 'relations': [], 'instruction_evidence': []}
    for x in raw.get('landmarks', [])[:6]:
        if not isinstance(x, dict): continue
        b = bearing(x.get('bearing_deg'))
        direction = 'unknown' if b is None else ('front' if b <= 45 or b >= 315 else 'left' if b < 135 else 'behind' if b <= 225 else 'right')
        row['landmarks'].append({'name': str(x.get('name', 'unknown')), 'bearing_deg': b,
                                 'relative_direction': direction, 'confidence': unit(x.get('confidence'))})
    for x in raw.get('relations', [])[:3]:
        if isinstance(x, dict):
            row['relations'].append({'subject': str(x.get('subject', 'unknown')), 'relation': str(x.get('relation', 'unknown')),
                                     'bearing_deg': bearing(x.get('bearing_deg')), 'confidence': unit(x.get('confidence')),
                                     'status': 'model_visual_hypothesis'})
    evidence = {x.get('phrase_index'): x for x in raw.get('evidence', []) if isinstance(x, dict)}
    for i, phrase in enumerate(phrases):
        x = evidence.get(i, {})
        row['instruction_evidence'].append({'phrase': phrase, 'phrase_index': i, 'score': unit(x.get('score')),
                                            'relation': str(x.get('relation', 'unknown')), 'confidence': unit(x.get('confidence')),
                                            'bearing_deg': bearing(x.get('bearing_deg')), 'completed': None})
    last = row['instruction_evidence'][-1]
    row['global_confidence'] = last['confidence']
    row['semantic_stop_score'] = last['score'] * last['confidence']
    row['semantic_embedding'] = structural_embedding(row)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--capture', type=Path, required=True)
    ap.add_argument('--model', type=Path, required=True)
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--batch-size', type=int, default=4)
    ap.add_argument('--max-nodes', type=int)
    ap.add_argument('--resume', action='store_true')
    a = ap.parse_args()
    import numpy as np
    import torch
    from PIL import Image, ImageDraw
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    torch.set_num_threads(4)
    torch.manual_seed(100)
    out = a.output_dir
    out.mkdir(parents=True, exist_ok=a.resume)
    graph_path = a.capture/'graph_options/graph_options.jsonl'
    # Explicit allowlist: neither logits, native actions nor labels reach generator.
    inputs = []
    for line in graph_path.open():
        g = json.loads(line)
        assert str(g['current_vp']) in {str(n) for n, v in zip(g['graph_ids'], g['visited']) if v}
        inputs.append({k: g[k] for k in ['episode_id','high_level_step','current_vp','graph_position','instruction','semantic_visual']})
    model = Qwen2VLForConditionalGeneration.from_pretrained(str(a.model), torch_dtype=torch.bfloat16, device_map='cuda:0', attn_implementation='sdpa').eval()
    processor = AutoProcessor.from_pretrained(str(a.model), min_pixels=256*28*28, max_pixels=1024*28*28)
    processor.tokenizer.padding_side = 'left'
    def generate(prompts, images=None, max_tokens=650):
        chats = []
        for prompt in prompts:
            content = ([{'type':'image'}] if images else []) + [{'type':'text','text':prompt}]
            chats.append(processor.apply_chat_template([{'role':'user','content':content}], tokenize=False, add_generation_prompt=True))
        kwargs = dict(text=chats, padding=True, return_tensors='pt')
        if images: kwargs['images'] = images
        batch = processor(**kwargs).to(model.device)
        with torch.inference_mode(): result = model.generate(**batch, max_new_tokens=max_tokens, do_sample=False)
        return processor.batch_decode(result[:, batch.input_ids.shape[1]:], skip_special_tokens=True)
    phrases_path = out/'ordered_phrases.json'
    phrases = {str(g['episode_id']): {'instruction': g['instruction'],
                                      'phrases': deterministic_phrases(g['instruction']),
                                      'raw': None} for g in inputs}
    phrases_path.write_text(json.dumps(phrases, indent=2)+'\n')
    path = out/'semantic_node_memory.jsonl'
    done = set()
    if a.resume and path.exists():
        done = {(str(r['episode_id']),r['high_level_step']) for r in map(json.loads,path.open())}
    todo = [g for g in inputs if (str(g['episode_id']),g['high_level_step']) not in done]
    if a.max_nodes: todo = todo[:a.max_nodes]
    manifest = {'model':'Qwen/Qwen2-VL-2B-Instruct', 'revision':REVISION, 'model_path':str(a.model),
                'graph_sha256':digest(graph_path), 'script_sha256':digest(__file__), 'prompt':VISUAL_PROMPT, 'phrase_prompt':PHRASE_PROMPT,
                'do_sample':False, 'seed':100, 'confidence_calibrated':False, 'privileged_inputs':False,
                'feature_inputs':['pre_action_12_rgb_views','instruction_text','node_identity','position'],
                'embedding':'sha256 signed hashing of structures, 256 dimensions'}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    for offset in range(0,len(todo),a.batch_size):
        batch=todo[offset:offset+a.batch_size]; images=[]; prompts=[]
        for g in batch:
            canvas=Image.new('RGB',(4*224,3*244),'white'); draw=ImageDraw.Draw(canvas)
            files=g['semantic_visual']['files']
            for j,b in enumerate(range(0,360,30)):
                key='rgb' if b==0 else 'rgb_'+str(b)
                img=Image.fromarray(np.load(files[key],allow_pickle=False)[...,:3]).resize((224,224))
                x,y=(j%4)*224,(j//4)*244
                canvas.paste(img,(x,y+20));draw.text((x+3,y+3),str(b)+' degrees',fill='black')
            images.append(canvas)
            ps=phrases[str(g['episode_id'])]['phrases']
            prompts.append(VISUAL_PROMPT+json.dumps(dict(enumerate(ps))))
        answers=generate(prompts,images)
        with path.open('a') as stream:
            for g,answer in zip(batch,answers):
                ps=phrases[str(g['episode_id'])]['phrases']; error=None
                try: row=normalize(parse_json(answer),ps)
                except Exception as exc:
                    error=type(exc).__name__+': '+str(exc);row=normalize({},ps)
                row.update({'episode_id':str(g['episode_id']),'high_level_step':g['high_level_step'],
                            'node_id':g['current_vp'],'position':g['graph_position'],
                            'observation_status':'observed_360_RGB_model_hypotheses','privileged_future_input_used':False,
                            'source_observation_hashes':g['semantic_visual']['observation_hashes'],
                            'parse_error':error,'raw_model_output':answer})
                stream.write(json.dumps(row,sort_keys=True)+'\n')
        print('extracted {}/{} (previous {})'.format(min(offset+a.batch_size,len(todo)),len(todo),len(done)),flush=True)
    print('features frozen:',digest(path),flush=True)

if __name__=='__main__': main()
