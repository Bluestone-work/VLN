#!/usr/bin/env python3
"""Evaluate human directional labels against frozen causal pairs.

The script is intentionally fail-closed: incomplete or mismatched annotation
rows stop evaluation.  It reads outcome labels only after the annotation file
has been validated against the frozen image join keys.
"""
import argparse, json, math
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from scipy.special import expit
from scipy.optimize import minimize

def load(p): return [json.loads(x) for x in Path(p).open() if x.strip()]
def nearest(b): return min(range(0,360,30),key=lambda x:min((x-b)%360.,(b-x)%360.))
def relation_match(rel,b):
    rel=(rel or '').lower(); b=nearest(float(b))
    if rel in ('none','uncertain'): return 0.
    if rel in ('ahead','left','right','behind'):
        return float({'ahead':0,'left':90,'behind':180,'right':270}[rel]==b)
    return 1.
def phrase_tokens(r):
    return set((r['instruction']+' '+' '.join(r['candidate_phrases'])).lower().replace('/',' ').split())
def landmark_match(labels,r):
    terms=phrase_tokens(r); vals=[]
    for x in labels:
        x=x.lower().replace('/',' ').split()
        vals.append(float(any(t in terms for t in x)))
    return max(vals or [0.])
def features(a,q):
    labs=[x for x in a.get('landmarks',[]) if x not in ('none','uncertain')]
    return np.array([landmark_match(labs,q),float(a.get('instruction_support')=='yes'),
        float(a.get('navigable_opening')=='yes'),relation_match(a.get('relation'),q['bearing_deg']),
        float(a.get('instruction_support')=='uncertain'),float(a.get('annotator_confidence')=='high')])
def fit(x,y):
    mu=x.mean(0);sd=x.std(0);sd[sd<1e-8]=1.; z=(x-mu)/sd
    init=np.zeros(z.shape[1]+1);init[-1]=math.log(max(y.mean(),1e-6)/max(1-y.mean(),1e-6))
    def fn(w):
        p=expit(z.dot(w[:-1])+w[-1]);e=p-y
        return np.mean(np.logaddexp(0,z.dot(w[:-1])+w[-1])-y*(z.dot(w[:-1])+w[-1])),np.r_[z.T.dot(e)/len(y),e.mean()]
    f=minimize(fn,init,jac=True,method='L-BFGS-B'); return f.x,mu,sd
def pred(model,x):
    w,mu,sd=model;return expit(((x-mu)/sd).dot(w[:-1])+w[-1])
def auc(y,p):
    y=np.asarray(y,bool);p=np.asarray(p); pos=p[y];neg=p[~y]
    return None if not len(pos) or not len(neg) else float((pos[:,None]>neg[None,:]).mean()+.5*(pos[:,None]==neg[None,:]).mean())
def report_metrics(y,p,rows):
    y=np.asarray(y,bool);p=np.asarray(p);pred=p>=.5;tp=np.sum(pred&y);fp=np.sum(pred&~y);tn=np.sum(~pred&~y);fn=np.sum(~pred&y)
    groups=defaultdict(list)
    for i,r in enumerate(rows):groups[(r['scene_id'],r['episode_id'],r['high_level_step'])].append(i)
    n=d=0
    for ix in groups.values():
        ps=[p[i] for i in ix if y[i]];ns=[p[i] for i in ix if not y[i]]
        for x in ps:
            for z in ns:n+=(x>z)+.5*(x==z);d+=1
    order=np.argsort(-p,kind='mergesort'); yy=y[order]; pp=p[order]; total=max(1,int(y.sum())); ap=0.; seen=0; got=0; i=0
    while i<len(order):
        j=i+1
        while j<len(order) and pp[j]==pp[i]: j+=1
        g=int(yy[i:j].sum()); seen+=j-i; got+=g
        if g: ap += (got/float(seen))*(g/float(total))
        i=j
    return {'auroc':auc(y,p),'auprc':float(ap) if y.sum() else None,'pairwise':None if not d else float(n/d),'benefit_precision':None if not tp+fp else float(tp/(tp+fp)),'benefit_recall':None if not tp+fn else float(tp/(tp+fn)),'harm_precision':None if not tn+fn else float(tn/(tn+fn)),'n':len(y)}
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--annotations',type=Path,required=True);ap.add_argument('--pairs',type=Path,default=Path('research/results/native_single_action_full_return_44_002/directional_pairs_002/pairs.jsonl'));ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    split=json.loads((a.root/'scene_split.json').read_text()); sectors=load(a.root/'sampled_sectors.jsonl'); ann=load(a.annotations); pairs=[x for x in load(a.pairs) if x['label']!='UNCERTAIN']; by={x['sector_id']:x for x in ann}
    if len(by)!=len(ann):raise ValueError('duplicate annotation sector_id')
    needed={x['sector_id'] for x in sectors}; missing=needed-set(by)
    if missing:raise ValueError('missing annotations: {}'.format(len(missing)))
    meta={x['sector_id']:x for x in sectors}
    for k,v in by.items():
        if k not in meta or v.get('image_sha256')!=meta[k]['image_sha256'] or v.get('scene_id')!=meta[k]['scene_id']:raise ValueError('annotation join mismatch '+k)
    amap={(m['scene_id'],str(m['episode_id']),int(m['high_level_step']),int(m['bearing_deg'])):by[m['sector_id']] for m in meta.values()}
    selected=[]
    for p in pairs:
        k=(p['scene_id'],str(p['episode_id']),int(p['high_level_step'])); b=p['features']['bearing_deg']
        if b is None:continue
        q=amap.get((k[0],k[1],k[2],nearest(b)))
        if q is not None:selected.append((p,q))
    test=[x for x in selected if x[0]['scene_id'] in split['test_scenes']];dev=[x for x in selected if x[0]['scene_id']==split['development_scene']]
    if not test or not dev:raise ValueError('need annotated development and test pairs')
    def run(rs):
        y=np.array([p['label']=='BENEFIT' for p,a in rs]); hx=np.array([features(a,p) for p,a in rs]);
        nx=np.array([[p['features']['native_logit'],p['features']['native_selected_logit'],p['features']['native_logit_margin'],p['features']['action_index'],float(p['features']['is_stop'])] for p,a in rs])
        return y,nx,hx
    yd,nd,hd=run(dev); yt,nt,ht=run(test)
    f0=fit(nd,yd); hm=fit(hd,yd); both=fit(np.c_[nd,hd],yd)
    p0=pred(f0,nt); ph=pred(hm,ht); pb=pred(both,np.c_[nt,ht])
    result={'scene_split':split,'development_pairs':len(dev),'test_pairs':len(test),'annotation_rows':len(ann),
        'F0':report_metrics(yt,p0,test),'Human-only':report_metrics(yt,ph,test),'F0+Human':report_metrics(yt,pb,test),
        'feature_definition':'F0 native logits/action controls; human six-feature semantic vector; each model fit on development scene only; test outcomes never used for fitting','labels':dict(Counter(p['label'] for p,a in test))}
    nc=(p0>=.5)==yt; hc=(ph>=.5)==yt
    result['complementarity']={'native_correct_human_correct':int(np.sum(nc&hc)),'native_correct_human_wrong':int(np.sum(nc&~hc)),'native_wrong_human_correct':int(np.sum(~nc&hc)),'native_wrong_human_wrong':int(np.sum(~nc&~hc))}
    groups=defaultdict(list)
    for i,(p,a) in enumerate(test): groups[(p['scene_id'],p['episode_id'],p['high_level_step'])].append(i)
    rescue=[]
    for k,ix in groups.items():
        bi=[i for i in ix if test[i][0]['label']=='BENEFIT']
        if not bi: continue
        native_order=sorted(ix,key=lambda i:p0[i],reverse=True); human_order=sorted(ix,key=lambda i:ph[i],reverse=True)
        rescue.append({'state':list(k),'benefit_actions':len(bi),'native_best_rank':min(native_order.index(i)+1 for i in bi),'human_top_rank':min(human_order.index(i)+1 for i in bi),'native_selected_index':test[ix[0]][0]['native_index']})
    result['rescue_opportunity']={'states':len(rescue),'human_top1':sum(x['human_top_rank']==1 for x in rescue),'human_top2':sum(x['human_top_rank']<=2 for x in rescue),'native_top1':sum(x['native_best_rank']==1 for x in rescue),'details':rescue}
    result['scene_results']={}
    for s in split['test_scenes']:
        ix=[i for i,(p,a) in enumerate(test) if p['scene_id']==s]
        result['scene_results'][s]={'pairs':len(ix),'F0':report_metrics(yt[ix],p0[ix],[test[i][0] for i in ix]),'Human-only':report_metrics(yt[ix],ph[ix],[test[i][0] for i in ix]),'F0+Human':report_metrics(yt[ix],pb[ix],[test[i][0] for i in ix])}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
