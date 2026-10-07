#!/usr/bin/env python3
"""Freeze one H=1 event per eligible route before altered continuation outcomes."""
import argparse
import collections
import hashlib
import json
from pathlib import Path

from audit_single_intervention import read_rows, key
from run_option_capture import digest


def select_events(states, minimum, seed):
    groups = collections.defaultdict(list)
    for row in states:
        if row['route_matched_gain_m'] >= minimum:
            groups[str(row['episode_id'])].append(row)
    selected = []
    for episode, rows in groups.items():
        selected.append(min(rows, key=lambda r: hashlib.sha256('{}|{}|{}|{}'.format(
            seed, r['scene_id'], episode, r['high_level_step']).encode()).hexdigest()))
    return selected


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    cfg = json.loads(args.config.read_text())
    route_dir = args.root / 'route_001'
    route_summary = json.loads((route_dir / 'summary.json').read_text())
    if not route_summary['baseline_reconstruction_gate_passed']:
        raise ValueError('Route fidelity failed')
    for f,flag in [('integrity_001/summary.json','integrity_gate_passed'),
                   ('noninterference_001/summary.json','baseline_noninterference_gate_passed')]:
        if not json.loads((args.root/f).read_text())[flag]:
            raise ValueError('Incomplete prior gate')
    states_path = route_dir / 'states.jsonl'
    states = list(read_rows(states_path).values())
    selected = select_events(states, cfg['continuation']['minimum_progress_gain_m'],
                             cfg['continuation']['event_selection_seed'])
    graphs_path = args.root / 'capture_001/graph_options/graph_options.jsonl'
    graphs = read_rows(graphs_path)
    events = []
    for state in selected:
        graph = graphs[key(state)]
        old = next(o for o in graph['options'] if o['index'] == state['selected_index'])
        new = next(o for o in graph['options'] if o['index'] == state['route_best_index'])
        if old['index'] == new['index'] or old['action']['act'] != 4 or new['action']['act'] != 4:
            raise ValueError('Invalid prospective replacement')
        events.append({'scene_id':state['scene_id'],'episode_id':str(state['episode_id']),
                       'high_level_step':state['high_level_step'],'trajectory_id':state['trajectory_id'],
                       'baseline_index':old['index'],'alternative_index':new['index'],
                       'baseline_action':old['action'],'alternative_action':new['action'],
                       'expected_graph_ids':graph['graph_ids'],'expected_graph_position':graph['graph_position'],
                       'local_route_matched_gain_m':state['route_matched_gain_m'],
                       'baseline_episode_success':state['baseline_episode_success']})
    schedule = {'experiment_id':cfg['experiment_id'], 'status':'frozen_before_continuation',
                'analysis_only_oracle':True,'expected_event_count':len(events),
                'base_experiment_config':str(args.config),'protocol':cfg['continuation_protocol'],
                'selection_rule':'Every H=1-eligible route, one SHA256-selected state; no altered future outcomes',
                'events':events,'hashes':{str(f):digest(f) for f in [args.config,
                    Path(cfg['continuation_protocol']),Path(__file__),states_path,graphs_path,
                    route_dir/'summary.json',args.root/'sampling_001/manifest.json']}}
    with args.output.open('x') as f:
        json.dump(schedule,f,indent=2);f.write('\n')
    print(json.dumps({'events':len(events),'scenes':len({e['scene_id'] for e in events}),
                      'episode_ids':[e['episode_id'] for e in events],
                      'status':'ready' if events else 'no_eligible_events'},indent=2))


if __name__ == '__main__':
    main()
