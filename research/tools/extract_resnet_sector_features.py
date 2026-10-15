#!/usr/bin/env python3
"""Extract frozen ImageNet ResNet18 features from current-node RGB sectors."""
import argparse, json
from pathlib import Path
from collections import OrderedDict
import numpy as np
import torch
from torchvision.models import resnet18


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--dataset',type=Path,required=True); ap.add_argument('--graph',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--batch-size',type=int,default=64); args=ap.parse_args()
    rows=[json.loads(x) for x in (args.dataset/'rows.jsonl').open()]
    graph={}
    for line in args.graph.open():
        r=json.loads(line); graph[(str(r['episode_id']),int(r['high_level_step']))]=r
    model=resnet18(pretrained=True); model.fc=torch.nn.Identity(); model.eval()
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model.to(device)
    cache=OrderedDict(); keys=[]; images=[]
    for r in rows:
        if not r['visual_valid']:
            keys.append(None); continue
        k=(r['episode_id'],r['high_level_step'],int(r['sector_deg']))
        keys.append(k)
        if k in cache: continue
        g=graph[k[:2]]; files=g['semantic_visual']['files']; sec=k[2]; key='rgb_{}'.format(sec) if sec else 'rgb'
        arr=np.load(files[key],allow_pickle=False)[...,:3].astype(np.float32)/255.
        t=torch.from_numpy(arr.transpose(2,0,1)).float()
        t=torch.nn.functional.interpolate(t.unsqueeze(0),size=(224,224),mode='bilinear',align_corners=False).squeeze(0)
        t=(t-torch.tensor([.485,.456,.406])[:,None,None])/torch.tensor([.229,.224,.225])[:,None,None]
        cache[k]=t
    for k,t in cache.items(): images.append(t)
    feats=[]
    with torch.no_grad():
        for start in range(0,len(images),args.batch_size):
            batch=torch.stack(images[start:start+args.batch_size]).to(device)
            feats.append(model(batch).cpu().numpy())
    featmap={k:v for k,v in zip(cache.keys(),np.concatenate(feats,0))}
    out=np.zeros((len(rows),512),dtype=np.float32)
    valid=np.zeros(len(rows),dtype=np.uint8)
    for i,k in enumerate(keys):
        if k is not None: out[i]=featmap[k]; valid[i]=1
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(str(args.output),resnet=out,valid=valid)
    meta={'model':'torchvision resnet18 ImageNet pretrained, fc removed, frozen','rows':len(rows),'unique_sectors':len(cache),'valid_rows':int(valid.sum()),'device':str(device),'input':'current pre-action RGB sector only','future_endpoint_used':False}
    args.output.with_suffix('.json').write_text(json.dumps(meta,indent=2)+'\n'); print(json.dumps(meta,indent=2))

if __name__=='__main__':main()
