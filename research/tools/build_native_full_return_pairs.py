#!/usr/bin/env python3
"""Assemble exact disabled/native and enabled/alternative full-return pairs.

This consumes only frozen traces and evaluator output.  Outcome fields are
kept separate from online features; the latter contain no future or final
metrics.  The label rule is fixed in the schedule and is never fitted here.
"""
import argparse, hashlib, json, math
from pathlib import Path

DEADBAND_NDTW = 1e-6
DEADBAND_COST = 1e-5

def rows(path):
    return [json.loads(x) for x in Path(path).open() if x.strip()]

def metrics(run):
    files = list((Path('data/logs/eval_results') / run).glob('stats_ep_*.json'))
    if len(files) != 1: raise ValueError('expected one episode metrics file: '+str(run))
    return json.loads(files[0].read_text())

def trace(run):
    files = list((Path(run)/'traces').glob('worker_seed*.jsonl'))
    if len(files) != 1: raise ValueError('expected one trace: '+str(run))
    return rows(files[0])

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def cost(r):
    return sum(1 for x in r.get('primitives',[]) if x.get('collided') is True)

def outcome(stats, trace_rows):
    # evaluator values are episode-final; primitives provide exact counts.
    ordered=sorted(trace_rows,key=lambda r:int(r['high_level_step']))
    x=dict(stats)
    x['primitive_action_count']=sum(len(r.get('primitives',[])) for r in ordered)
    x['collision_events']=sum(cost(r) for r in ordered)
    x['goal_distance']=float(ordered[-1]['distance_after'])
    return x

def label(native, alt):
    if bool(native['success']) != bool(alt['success']):
        return 'BENEFIT' if alt['success'] > native['success'] else 'HARM'
    for m in ('ndtw','sdtw'):
        d=float(alt[m])-float(native[m])
        if abs(d)>DEADBAND_NDTW: return 'BENEFIT' if d>0 else 'HARM'
    for m in ('path_length','primitive_action_count','collision_events'):
        d=float(alt[m])-float(native[m])
        if m=='path_length' and abs(d)<=DEADBAND_COST: continue
        if d != 0: return 'BENEFIT' if d<0 else 'HARM'
    return 'UNCERTAIN'

def unpack(v):
    if isinstance(v,dict) and '__array__' in v: return v['__array__']
    return v

def main():
    raise SystemExit('Deprecated unsafe assembler. Use audit_native_full_return.py, then '
                     'build_directional_causal_pairs.py. Previous prefix_exact values '
                     'were not checked and are not admissible scientific evidence.')

if __name__ == '__main__': main()
