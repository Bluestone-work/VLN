#!/usr/bin/env python3
"""Fit an execution-distance control using only the prior training census."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import numpy as np
from graph_value_transfer import critical_records
from graph_value_feasibility import FEATURE_NAMES, dominates, fit_pairwise

ACTIVE = ('graph_logit', 'logit_rank_percentile', 'ghost_distance_m', 'back_path_length_m')


def main():
    ap = argparse.ArgumentParser()
    for key in ['plan','run','full-model','output']:
        ap.add_argument('--'+key, type=Path, required=True)
    a = ap.parse_args()
    if a.output.exists(): raise ValueError('Never overwrite a frozen model')
    old = json.loads(a.full_model.read_text())
    if old['feature_names'] != list(FEATURE_NAMES): raise ValueError('Schema mismatch')
    if old['hashes'][str(a.run/'results.jsonl')] != hashlib.sha256((a.run/'results.jsonl').read_bytes()).hexdigest():
        raise ValueError('Training outcomes differ from FULL')
    rows = critical_records(a.plan, a.run)
    mean = np.mean([r['features'] for r in rows], axis=0)
    scale = np.std([r['features'] for r in rows], axis=0); scale[scale < 1e-8] = 1.0
    if not np.array_equal(mean,old['mean']) or not np.array_equal(scale,old['scale']):
        raise ValueError('Training normalization differs')
    active = [FEATURE_NAMES.index(k) for k in ACTIVE]
    groups = collections.defaultdict(list)
    for r in rows:
        r['z'] = ((r['features']-mean)/scale)[active]
        groups[(r['episode_id'],r['step'])].append(r)
    pairs = []
    for items in groups.values():
        for i,left in enumerate(items):
            for right in items[i+1:]:
                if dominates(left['metrics'],right['metrics']): pairs.append((left,right))
                elif dominates(right['metrics'],left['metrics']): pairs.append((right,left))
    if len(pairs) != old['strict_pairs']: raise ValueError('Pair population differs')
    fit = fit_pairwise(pairs)
    if fit is None or not np.isfinite(fit).all(): raise ValueError('Invalid fit')
    weights = np.zeros(len(FEATURE_NAMES)); weights[active] = fit
    paths=[a.plan,a.run/'results.jsonl',a.full_model,Path(__file__),Path('research/tools/graph_value_feasibility.py'),Path('research/tools/graph_value_transfer.py')]
    result={'schema_version':1,'feature_names':list(FEATURE_NAMES),'active_features':list(ACTIVE),
            'weights_standardized':weights.tolist(),'mean':mean.tolist(),'scale':scale.tolist(),
            'train_states':len(groups),'train_actions':len(rows),'strict_pairs':len(pairs),'ridge_lambda':10.,
            'target_labels_used':False,'training_plan':str(a.plan),'training_run':str(a.run),
            'hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f: json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({'active_features':list(ACTIVE),'pairs':len(pairs),'weights':fit.tolist()}))

if __name__=='__main__': main()
