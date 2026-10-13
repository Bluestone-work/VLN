#!/usr/bin/env python3
"""Offline LOSO probes; labels are used only to fit a linear audit classifier.

No navigation policy is fitted. Raw-text diagnostics bypass model confidence
explicitly, and must not be interpreted as validated deployable semantics.
"""
import argparse, hashlib, json, math, re
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from scipy.special import expit
from scipy.optimize import minimize

NAMES=('F0','F0+F1','F0+F2','F0+F3','F3_without_native','F0+F1+F3','F0+gated_F3')


def text_hash(tokens, dim=64):
    out=np.zeros(dim)
    for token in tokens:
        h=hashlib.sha256(token.lower().encode()).digest()
        out[int.from_bytes(h[:4],'big')%dim]+=1. if h[4]%2 else -1.
    return out.tolist()


def blocks(r):
    f=r['features']; b=f['bearing_deg']; theta=math.radians(b or 0.)
    base=[f['native_logit'],f['native_selected_logit'],f['native_logit_margin'],float(f['is_stop']),
          float(f['current_proposal']),f['graph_size'],f['back_path_length']]+f['native_embedding']
    geo=[math.sin(theta) if b is not None else 0., math.cos(theta) if b is not None else 0.,f['distance'],float(b is None)]
    tokens=['object:'+x['name'] for x in f['sector_landmarks']]
    tokens+=['relation:'+x['subject']+':'+x['relation'] for x in f['sector_relations']]
    sem=[f['semantic_landmark_count'],f['semantic_relation_count']]+text_hash(tokens)
    ins=[f['instruction_evidence'],f['instruction_evidence_count'],f['semantic_global_confidence']]
    # Include matched instruction relation and phrase content only where the
    # extractor supplies a directional evidence entry; unknown bearing abstains.
    ins+=text_hash([x['phrase']+':'+x['relation'] for x in f['sector_evidence']])
    gated=[x*max(f['semantic_landmark_confidence'],f['semantic_relation_confidence'],f['semantic_global_confidence']) for x in sem]+ins
    return base,geo,sem,ins,gated


def matrices(rows):
    out={n:[] for n in NAMES}
    for r in rows:
        base,geo,sem,ins,gated=blocks(r)
        for n,v in zip(NAMES,(base,base+geo,base+sem,base+sem+ins,sem+ins,base+geo+sem+ins,base+gated)):
            out[n].append(v)
    return {n:np.array(v,float) for n,v in out.items()}


def fit_predict(x,y,xt):
    mu=x.mean(0);sd=x.std(0);sd[sd<1e-8]=1.
    x=(x-mu)/sd;xt=(xt-mu)/sd
    # Drop train-constant columns, preserving equivalent fits for zero evidence.
    keep=np.std(x,axis=0)>1e-8;x=x[:,keep];xt=xt[:,keep]
    init=np.zeros(x.shape[1]+1);init[-1]=math.log(max(y.mean(),1e-6)/max(1-y.mean(),1e-6))
    def fun(w):
        z=x.dot(w[:-1])+w[-1];e=expit(z)-y
        loss=np.mean(np.logaddexp(0,z)-y*z)+.005*np.dot(w[:-1],w[:-1])
        grad=np.r_[x.T.dot(e)/len(y)+.01*w[:-1],e.mean()]
        return loss,grad
    fit=minimize(fun,init,jac=True,method='L-BFGS-B',options={'maxiter':1000,'ftol':1e-11,'gtol':1e-7})
    if not fit.success:raise ValueError('probe did not converge: '+str(fit.message))
    return expit(xt.dot(fit.x[:-1])+fit.x[-1])


def auc_ap(y,p):
    y=np.asarray(y,bool);p=np.asarray(p,float);npos=int(y.sum());nneg=len(y)-npos
    if not npos or not nneg:return None,None
    order=np.argsort(-p,kind='mergesort');yy=y[order];pp=p[order]
    ends=np.r_[np.flatnonzero(np.diff(pp)!=0),len(p)-1]
    tp=np.cumsum(yy)[ends];fp=ends+1-tp
    auc=float(np.trapz(np.r_[0,tp]/npos,np.r_[0,fp]/nneg))
    ap=float(np.sum(np.diff(np.r_[0,tp])/npos*tp/(ends+1)))
    return auc,ap


def metrics(y,p):
    y=np.asarray(y,bool);p=np.asarray(p,float);pred=p>=.5
    tp=int(np.sum(pred&y));fp=int(np.sum(pred&~y));tn=int(np.sum(~pred&~y));fn=int(np.sum(~pred&y));ece=0.
    for i in range(10):
        mask=(p>=i/10.)&(p<((i+1)/10.) if i<9 else p<=1)
        if mask.any():ece+=mask.mean()*abs(p[mask].mean()-y[mask].mean())
    auc,ap=auc_ap(y,p)
    return {'auroc':auc,'auprc':ap,'benefit_precision':tp/(tp+fp) if tp+fp else None,
        'benefit_recall':tp/(tp+fn) if tp+fn else None,'harm_precision':tn/(tn+fn) if tn+fn else None,
        'harm_recall':tn/(tn+fp) if tn+fp else None,'predicted_benefit':int(pred.sum()),
        'brier':float(np.mean((p-y)**2)),'ece_10bin':float(ece),'confusion':{'tp':tp,'fp':fp,'tn':tn,'fn':fn}}


def within_state_pairs(rows,y):
    groups=defaultdict(list)
    for i,r in enumerate(rows):groups[(r['scene_id'],r['episode_id'],r['high_level_step'])].append(i)
    pos=[];neg=[];scenes=[]
    for k,ix in groups.items():
        for a in ix:
            if not y[a]:continue
            for b in ix:
                if y[b]:continue
                pos.append(a);neg.append(b);scenes.append(k[0])
    return np.array(pos,int),np.array(neg,int),np.array(scenes)


def paired_bootstrap(y,preds,scenes,pair_indices,n=2000):
    unique=sorted(set(scenes));by={s:np.where(scenes==s)[0] for s in unique}
    pi,ni,ps=pair_indices;pb={s:np.where(ps==s)[0] for s in unique}
    scores={name:((p[pi]>p[ni]).astype(float)+.5*(p[pi]==p[ni])) for name,p in preds.items()}
    vals={name:defaultdict(list) for name in preds};rng=np.random.RandomState(100)
    for _ in range(n):
        sample=rng.choice(unique,len(unique),replace=True)
        ix=np.concatenate([by[s] for s in sample]);px=np.concatenate([pb[s] for s in sample])
        for name,p in preds.items():
            au,ap=auc_ap(y[ix],p[ix])
            if au is not None:vals[name]['auroc'].append(au);vals[name]['auprc'].append(ap)
            if len(px):vals[name]['pairwise'].append(float(scores[name][px].mean()))
    def interval(v):return np.percentile(v,[2.5,97.5]).tolist() if len(v) else None
    cis={name:{k:interval(v) for k,v in vv.items()} for name,vv in vals.items()}
    deltas={}
    for other,base in [('F0+F3','F0'),('F0+F1+F3','F0+F1'),('F0+F3','F0+F2'),('F0+gated_F3','F0')]:
        deltas[other+' minus '+base]={k:interval(np.array(vals[other][k])-np.array(vals[base][k])) for k in ('auroc','auprc','pairwise')}
    return cis,deltas


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--pairs',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    allrows=[json.loads(x) for x in a.pairs.open() if x.strip()]
    rows=[r for r in allrows if r['label']!='UNCERTAIN'];y=np.array([r['label']=='BENEFIT' for r in rows]);scenes=np.array([r['scene_id'] for r in rows])
    xs=matrices(rows);preds={name:np.zeros(len(rows)) for name in NAMES}
    for name in NAMES:
        for scene in sorted(set(scenes)):
            test=scenes==scene;preds[name][test]=fit_predict(xs[name][~test],y[~test],xs[name][test])
        print('fitted',name,flush=True)
    pp=within_state_pairs(rows,y);pi,ni,_=pp
    cis,deltas=paired_bootstrap(y,preds,scenes,pp)
    result={'pairs':len(rows),'uncertain_excluded':len(allrows)-len(rows),'scenes':len(set(scenes)),
        'labels':dict(Counter(r['label'] for r in rows)),'models':{},'paired_scene_bootstrap_deltas':deltas,
        'within_state_benefit_harm_comparisons':len(pi),'bootstrap_replicates':2000}
    for name,p in preds.items():
        result['models'][name]=dict(metrics(y,p),features=xs[name].shape[1],
            within_state_pairwise_accuracy=float(((p[pi]>p[ni])+.5*(p[pi]==p[ni])).mean()),scene_bootstrap_ci=cis[name])
    four={};nc=(preds['F0']>=.5)==y
    for variant in ('F3_without_native','F0+F3'):
        sc=(preds[variant]>=.5)==y
        four[variant]={'native_correct_semantic_correct':int(np.sum(nc&sc)),
            'native_correct_semantic_wrong':int(np.sum(nc&~sc)),
            'native_wrong_semantic_correct':int(np.sum(~nc&sc)),
            'native_wrong_semantic_wrong':int(np.sum(~nc&~sc))}
    result['complementarity']=four
    result['native_wrong_stratum']={n:metrics(y[~nc],p[~nc]) for n,p in preds.items()}
    # Outcome-defined stratum for analysis only: native selected action was
    # inferior to at least one fully continued alternative.
    good_states={(r['episode_id'],r['high_level_step']) for r in rows if r['label']=='BENEFIT'}
    mask=np.array([(r['episode_id'],r['high_level_step']) in good_states for r in rows])
    result['native_suboptimal_states']=len(good_states)
    result['native_suboptimal_state_stratum']={n:metrics(y[mask],p[mask]) for n,p in preds.items()}
    result['zero_confidence_abstentions']=sum(r['features']['semantic_abstained'] for r in rows)
    result['oof_prediction_correlation']=float(np.corrcoef(preds['F0'],preds['F0+F3'])[0,1])
    result['protocol']={'optimizer':'unweighted L2 logistic, lambda=.01, LBFGS, maxiter=1000',
        'threshold':.5,'split':'leave_one_scene_out','standardization':'train_fold_only','bootstrap':'paired cluster resampling of fixed OOF predictions; conditional on fits',
        'F2_caveat':'Visual-field ablation of an instruction-conditioned extractor; NOT independent visual-only extraction',
        'raw_text_caveat':'F2/F3 use unweighted structured text as diagnostics; confidence gate rejects all current records',
        'future_inputs':False,'threshold_tuning':False,'native_embedding':'full 768d frozen contextual action embedding'}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
    with (a.output.parent/'probe_oof.jsonl').open('w') as stream:
        for i,r in enumerate(rows):stream.write(json.dumps({'pair_id':r['pair_id'],'scene_id':r['scene_id'],'label':r['label'],
            'episode_id':r['episode_id'],'high_level_step':r['high_level_step'],'predictions':{n:float(p[i]) for n,p in preds.items()}})+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
