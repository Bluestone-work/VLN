#!/usr/bin/env python3
"""Scene-disjoint value probes on the current-node RGB-D dataset."""
import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


def auc_ap(y, p):
    y = np.asarray(y, bool); p = np.asarray(p, float)
    pos = int(y.sum()); neg = len(y) - pos
    if not pos or not neg: return None, None
    order = np.argsort(-p, kind='mergesort'); yy = y[order]; pp = p[order]
    ends = np.r_[np.flatnonzero(np.diff(pp) != 0), len(p)-1]
    tp = np.cumsum(yy)[ends]; fp = ends + 1 - tp
    auc = float(np.trapz(np.r_[0., tp]/pos, np.r_[0., fp]/neg))
    ap = float(np.sum(np.diff(np.r_[0., tp])/pos * tp/(ends+1)))
    return auc, ap


def weighted_fit_predict(x, y, weights, xt):
    mu = x.mean(0); sd = x.std(0); sd[sd < 1e-8] = 1.
    x = (x-mu)/sd; xt = (xt-mu)/sd
    keep = np.std(x, axis=0) > 1e-8
    x = x[:, keep]; xt = xt[:, keep]
    weights = weights / max(weights.mean(), 1e-8)
    init = np.zeros(x.shape[1]+1)
    init[-1] = math.log(max(float(np.average(y, weights=weights)), 1e-6) /
                        max(1-float(np.average(y, weights=weights)), 1e-6))
    def fun(w):
        z = x.dot(w[:-1]) + w[-1]
        e = expit(z)-y
        loss = np.average(np.logaddexp(0, z)-y*z, weights=weights) + .01*np.dot(w[:-1], w[:-1])
        grad = np.r_[((weights*e)[:, None]*x).sum(0)/weights.sum()+.02*w[:-1],
                     (weights*e).sum()/weights.sum()]
        return loss, grad
    fit = minimize(fun, init, jac=True, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-10})
    if not fit.success: raise RuntimeError(str(fit.message))
    return expit(xt.dot(fit.x[:-1]) + fit.x[-1])


def fold_weights(rows, y, train):
    # Equalize state contribution first, then class contribution.
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        if train[i]: groups[r['state_key']].append(i)
    w = np.zeros(len(rows), float)
    for ix in groups.values():
        for i in ix: w[i] = 1. / len(ix)
    for cls in [0, 1]:
        total = w[train & (y == cls)].sum()
        if total: w[train & (y == cls)] *= .5 / total
    return w


def metrics(y, p):
    auc, ap = auc_ap(y, p); pred = p >= .5
    return {'auroc': auc, 'auprc': ap,
            'benefit_precision': float(np.sum(pred & y)/max(np.sum(pred), 1)),
            'benefit_recall': float(np.sum(pred & y)/max(np.sum(y), 1)),
            'predicted_benefit': int(pred.sum()),
            'brier': float(np.mean((p-y)**2))}


def pair_indices(rows, y):
    groups = defaultdict(list)
    for i, r in enumerate(rows): groups[r['state_key']].append(i)
    pos=[]; neg=[]; scenes=[]
    for state, ix in groups.items():
        for i in ix:
            if not y[i]: continue
            for j in ix:
                if y[j]: continue
                pos.append(i); neg.append(j); scenes.append(next(r['scene_id'] for r in [rows[i]]))
    return np.asarray(pos), np.asarray(neg), np.asarray(scenes)


def pairwise(rows, y, p):
    pos, neg, _ = pair_indices(rows, y)
    if not len(pos): return None
    return float(np.mean((p[pos] > p[neg]) + .5*(p[pos] == p[neg])))


def rescue(rows, y, p, threshold=.5):
    groups = defaultdict(list)
    for i, r in enumerate(rows): groups[r['state_key']].append(i)
    rescue_count = destroy_count = native_wrong = native_correct = 0
    for ix in groups.values():
        has_benefit = bool(np.any(y[ix]))
        if has_benefit: native_wrong += 1
        else: native_correct += 1
        best = ix[int(np.argmax(p[ix]))]
        override = p[best] >= threshold
        if has_benefit and override and y[best]: rescue_count += 1
        if not has_benefit and override and not y[best]: destroy_count += 1
    return {'states': len(groups), 'native_wrong_states': native_wrong,
            'native_correct_states': native_correct,
            'rescue_opportunity_states': native_wrong,
            'native_wrong_model_correct': rescue_count,
            'native_correct_model_wrong': destroy_count,
            'rescue_rate': rescue_count/max(native_wrong, 1),
            'destruction_rate': destroy_count/max(native_correct, 1)}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--dataset', type=Path, required=True); ap.add_argument('--output', type=Path, required=True); args=ap.parse_args()
    d=np.load(str(args.dataset/'features.npz')); rows=[json.loads(x) for x in (args.dataset/'rows.jsonl').open()]
    y=d['labels'].astype(bool); scenes=np.asarray([r['scene_id'] for r in rows]);
    blocks={'F0': d['native'], 'F0+geometry': np.c_[d['native'], d['geometry']],
            'F0+visual': np.c_[d['native'], d['visual']],
            'F0+geometry+visual': np.c_[d['native'], d['geometry'], d['visual']],
            'visual-only': d['visual']}
    preds={n:np.zeros(len(rows), float) for n in blocks}
    for name,x in blocks.items():
        for scene in sorted(set(scenes)):
            test=scenes==scene; train=~test
            w=fold_weights(rows, y, train)
            preds[name][test]=weighted_fit_predict(x[train], y[train], w[train], x[test])
        print('fitted', name, flush=True)
    out={'dataset':str(args.dataset), 'rows':len(rows), 'states':len(set(r['state_key'] for r in rows)),
         'scenes':len(set(scenes)), 'labels':dict(Counter('BENEFIT' if z else 'HARM' for z in y)),
         'split':'leave-one-scene-out', 'weighting':'inverse candidate count per state then class balanced per train fold',
         'models':{}, 'paired_deltas':{}, 'rescue':{}}
    for name,p in preds.items():
        out['models'][name]=dict(metrics(y,p), within_state_pairwise=pairwise(rows,y,p), rescue=rescue(rows,y,p))
    rng=np.random.RandomState(20261015); unique=sorted(set(scenes)); vals={n:defaultdict(list) for n in blocks}
    for _ in range(2000):
        sample=rng.choice(unique,len(unique),replace=True)
        for name,p in preds.items():
            ix=np.concatenate([np.where(scenes==s)[0] for s in sample])
            auc,ap=auc_ap(y[ix],p[ix]); vals[name]['auroc'].append(auc); vals[name]['auprc'].append(ap)
            vals[name]['pairwise'].append(pairwise([rows[i] for i in ix], y[ix], p[ix]))
    for name in vals:
        out['models'][name]['scene_bootstrap_ci']={k:np.percentile(v,[2.5,97.5]).tolist() for k,v in vals[name].items()}
    for other,base in [('F0+visual','F0'),('F0+geometry+visual','F0+geometry'),('visual-only','F0')]:
        out['paired_deltas'][other+' minus '+base]={}
        for metric in ['auroc','auprc','pairwise']:
            delta=np.asarray(vals[other][metric])-np.asarray(vals[base][metric])
            out['paired_deltas'][other+' minus '+base][metric]=np.percentile(delta,[2.5,97.5]).tolist()
    # OOF predictions are retained for later conditional/error audits.
    with (args.output.parent/'visual_value_oof.jsonl').open('w') as f:
        for i,r in enumerate(rows): f.write(json.dumps({'pair_id':r['pair_id'],'state_key':r['state_key'],'scene_id':r['scene_id'],'label':r['label'],'predictions':{n:float(p[i]) for n,p in preds.items()}})+'\n')
    args.output.write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(out, indent=2))


if __name__=='__main__': main()
