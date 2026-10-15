#!/usr/bin/env python3
"""Pilot action-conditioned latent transition/value probe.

Post-action RGB-D is used only as a training target for a frozen ResNet latent
dynamics fit. It is never supplied to the value classifier as an online input.
The pilot is scene-disjoint and outcome labels are used only by the value head.
"""
import argparse, json, math
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
import torch
from torchvision.models import resnet18


def auc_ap(y,p):
    y=np.asarray(y,bool);p=np.asarray(p,float);pos=int(y.sum());neg=len(y)-pos
    if not pos or not neg:return None,None
    order=np.argsort(-p,kind='mergesort'); yy=y[order]; pp=p[order]
    ends=np.r_[np.flatnonzero(np.diff(pp)!=0),len(p)-1];tp=np.cumsum(yy)[ends];fp=ends+1-tp
    return float(np.trapz(np.r_[0.,tp]/pos,np.r_[0.,fp]/neg)),float(np.sum(np.diff(np.r_[0.,tp])/pos*tp/(ends+1)))


def fit_logistic(x,y,w,xt):
    mu=x.mean(0);sd=x.std(0);sd[sd<1e-8]=1.; x=(x-mu)/sd;xt=(xt-mu)/sd
    keep=np.std(x,axis=0)>1e-8;x=x[:,keep];xt=xt[:,keep];w=w/max(w.mean(),1e-8)
    q=float(np.average(y,weights=w)); init=np.zeros(x.shape[1]+1);init[-1]=math.log(max(q,1e-6)/max(1-q,1e-6))
    def fun(v):
        z=x.dot(v[:-1])+v[-1];e=expit(z)-y
        loss=np.average(np.logaddexp(0,z)-y*z,weights=w)+.01*np.dot(v[:-1],v[:-1])
        grad=np.r_[((w*e)[:,None]*x).sum(0)/w.sum()+.02*v[:-1],(w*e).sum()/w.sum()]
        return loss,grad
    r=minimize(fun,init,jac=True,method='L-BFGS-B',options={'maxiter':1000,'ftol':1e-10})
    if not r.success:raise RuntimeError(str(r.message))
    return expit(xt.dot(r.x[:-1])+r.x[-1])


def balanced_weights(rows,y,train):
    groups=defaultdict(list)
    for i,r in enumerate(rows):
        if train[i]:groups[r['state_key']].append(i)
    w=np.zeros(len(rows),float)
    for ix in groups.values():
        for i in ix:w[i]=1./len(ix)
    for c in [0,1]:
        z=w[train&(y==c)].sum()
        if z:w[train&(y==c)]*=.5/z
    return w


def endpoint_features(paths, device):
    model=resnet18(pretrained=True);model.fc=torch.nn.Identity();model.eval().to(device)
    cache={};todo=[]
    for p in paths:
        if p not in cache:todo.append(p);cache[p]=None
    tensors=[]
    for p in todo:
        a=np.load(p,allow_pickle=False)[...,:3].astype(np.float32)/255.
        t=torch.from_numpy(a.transpose(2,0,1));t=torch.nn.functional.interpolate(t[None],size=(224,224),mode='bilinear',align_corners=False)[0]
        t=(t-torch.tensor([.485,.456,.406])[:,None,None])/torch.tensor([.229,.224,.225])[:,None,None];tensors.append(t)
    with torch.no_grad():
        for i in range(0,len(tensors),64):
            z=model(torch.stack(tensors[i:i+64]).to(device)).cpu().numpy()
            for p,v in zip(todo[i:i+64],z):cache[p]=v.astype(np.float32)
    return cache


def pairwise(rows,y,p):
    groups=defaultdict(list)
    for i,r in enumerate(rows):groups[r['state_key']].append(i)
    a=[];b=[]
    for ix in groups.values():
        for i in ix:
            if not y[i]:continue
            for j in ix:
                if y[j]:continue
                a.append(i);b.append(j)
    if not a:return None
    return float(np.mean((p[a]>p[b])+.5*(p[a]==p[b])))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--endpoint-dir',type=Path,required=True);ap.add_argument('--pairs',type=Path,required=True);ap.add_argument('--dataset',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    pairmap={(str(x['episode_id']),int(x['high_level_step']),int(x['alternative_index'])):x for x in map(json.loads,args.pairs.open())}
    base_rows=[json.loads(x) for x in (args.dataset/'rows.jsonl').open()];base=np.load(str(args.dataset/'features.npz'))
    resnet=np.load(str(args.dataset/'resnet18_features.npz'))['resnet']
    base_map={(str(r['episode_id']),int(r['high_level_step']),int(r['native_action_index'])):i for i,r in enumerate(base_rows)}
    endpoint_rows=[json.loads(x) for x in (args.endpoint_dir/'native_endpoint_options.jsonl').open()]
    rows=[]; endpoint_paths=[]; indices=[]
    for e in endpoint_rows:
        k=(str(e['episode_id']),int(e['high_level_step']),int(e['index']))
        if k not in pairmap:continue  # native sentinel has no alternative outcome label
        if 'rgb' not in e['endpoint_files']:continue
        if k not in base_map:continue
        p=pairmap[k];i=base_map[k];endpoint_rows_key=str(Path(e['endpoint_files']['rgb']).resolve())
        rows.append({'state_key':'{}|{}|{}'.format(p['scene_id'],p['episode_id'],p['high_level_step']),'scene_id':p['scene_id'],'episode_id':str(p['episode_id']),'high_level_step':int(p['high_level_step']),'pair_id':p['pair_id'],'label':p['label'],'endpoint_path':endpoint_rows_key,'base_index':i})
        endpoint_paths.append(endpoint_rows_key);indices.append(i)
    epmap=endpoint_features(endpoint_paths,torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    y=np.asarray([r['label']=='BENEFIT' for r in rows]); scenes=np.asarray([r['scene_id'] for r in rows]);
    native=base['native'][indices]; geometry=base['geometry'][indices]; current=resnet[indices]; target=np.stack([epmap[p] for p in endpoint_paths])
    blocks={'F0':native,'F0+geometry':np.c_[native,geometry]}
    preds={n:np.zeros(len(rows)) for n in blocks}; preds['F0+predicted_transition']=np.zeros(len(rows)); preds['F0+oracle_endpoint']=np.zeros(len(rows))
    unique=sorted(set(scenes))
    for scene in unique:
        test=scenes==scene;train=~test;w=balanced_weights(rows,y,train)
        for n,x in blocks.items():preds[n][test]=fit_logistic(x[train],y[train],w[train],x[test])
        # Fit a low-rank linear latent dynamics model on training observations.
        yt=target[train]; ym=yt.mean(0); _,s,v=np.linalg.svd(yt-ym,full_matrices=False); k=min(16,len(s)); basis=v[:k]; ztrain=(yt-ym).dot(basis.T)
        xt=np.c_[current[train],geometry[train],native[train][:,:3]]; xe=np.c_[current[test],geometry[test],native[test][:,:3]]
        xm=xt.mean(0); xs=xt.std(0);xs[xs<1e-8]=1.; xt=(xt-xm)/xs;xe=(xe-xm)/xs
        reg=10.; A=xt.T.dot(xt)+reg*np.eye(xt.shape[1]); W=np.linalg.solve(A,xt.T.dot(ztrain)); ztest=xe.dot(W)
        # Dynamics model is trained from endpoint pixels only; value labels fit a separate head.
        zall=np.zeros((len(rows),k));zall[train]=ztrain;zall[test]=ztest
        preds['F0+predicted_transition'][test]=fit_logistic(np.c_[blocks['F0'][train],ztrain],y[train],w[train],np.c_[blocks['F0'][test],ztest])
        preds['F0+oracle_endpoint'][test]=fit_logistic(np.c_[blocks['F0'][train],ztrain],y[train],w[train],np.c_[blocks['F0'][test],(target[test]-ym).dot(basis.T)])
    out={'status':'pilot','rows':len(rows),'states':len(set(r['state_key'] for r in rows)),'scenes':len(unique),'labels':dict(Counter('BENEFIT' if z else 'HARM' for z in y)),'endpoint_target_only':True,'models':{},'protocol':{'split':'leave-one-scene-out by scene','transition_input':'current pre-action RGB ResNet + action geometry + native logit summary','transition_target':'post-action endpoint RGB ResNet PCA-16 latent','online_endpoint_observation_used':False,'value_labels_used_only_for_value_head':True}}
    for n,p in preds.items():
        auc,ap=auc_ap(y,p);out['models'][n]={'auroc':auc,'auprc':ap,'pairwise':pairwise(rows,y,p),'predicted_benefit':int((p>=.5).sum())}
    rng=np.random.RandomState(20261016); boot={n:{'auroc':[],'auprc':[],'pairwise':[]} for n in preds}; uniq=sorted(set(scenes))
    for _ in range(2000):
        sample=rng.choice(uniq,len(uniq),replace=True); ix=np.concatenate([np.where(scenes==s)[0] for s in sample])
        rr=[rows[i] for i in ix]
        for n,p in preds.items():
            aa,pp=auc_ap(y[ix],p[ix]);boot[n]['auroc'].append(aa);boot[n]['auprc'].append(pp);boot[n]['pairwise'].append(pairwise(rr,y[ix],p[ix]))
    for n in preds: out['models'][n]['scene_bootstrap_ci']={k:np.percentile(v,[2.5,97.5]).tolist() for k,v in boot[n].items()}
    for n in ['F0+predicted_transition','F0+oracle_endpoint']:
        out.setdefault('paired_deltas',{})[n+' minus F0']={k:np.percentile(np.asarray(boot[n][k])-np.asarray(boot['F0'][k]),[2.5,97.5]).tolist() for k in ['auroc','auprc','pairwise']}
    with (args.output.parent/'oof.jsonl').open('w') as f:
        for i,r in enumerate(rows): f.write(json.dumps({'pair_id':r['pair_id'],'scene_id':r['scene_id'],'state_key':r['state_key'],'label':r['label'],'predictions':{n:float(p[i]) for n,p in preds.items()}})+'\n')
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))

if __name__=='__main__':main()
