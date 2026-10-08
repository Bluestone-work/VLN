#!/usr/bin/env python3
"""Freeze Gate A dense-action cases from audited critical states."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from vlnce_baselines.adaptive_action.dense_candidate_specs import geometry_for_entries
from vlnce_baselines.adaptive_action.option_calibration import pack, unpack
from vlnce_baselines.models.graph_utils import estimate_cand_pos


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--critical-plan', type=Path, required=True)
    p.add_argument('--dense-capture', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--max-cases', type=int, default=-1)
    p.add_argument('--experiment-config', type=Path)
    p.add_argument('--failure-subset-dataset', type=Path)
    p.add_argument('--probe', default='probe_full001')
    p.add_argument('--noninterference', default='noninterference_001')
    args = p.parse_args()
    critical = json.loads(args.critical_plan.read_text())
    rows = {}
    with (args.dense_capture / 'graph_options/graph_options.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line)
            rows[(str(row['episode_id']), int(row['high_level_step']))] = row
    cases = []
    for state in critical['states']:
        ep = str(state['episode_id'])
        step = int(state['high_level_step'])
        if int(state.get('effective_index', 0)) == 0:
            continue
        row = rows.get((ep, step))
        if row is None:
            raise ValueError('Missing dense capture state {}'.format((ep, step)))
        provenance = row.get('dense_candidate_provenance')
        if not provenance or not provenance['nested_a0_in_a1']:
            raise ValueError('Invalid dense provenance at {}'.format((ep, step)))
        new_entries = [
            e for e in provenance['a1']
            if (e['angle_index'], e['distance_index']) not in {
                (x['angle_index'], x['distance_index']) for x in provenance['a0']
            }
        ]
        pose = unpack(row['options'][0]['action']['front_pos']) if row['options'][0]['action'].get('front_pos') is not None else unpack(row['graph_position'])
        # The controller trace has the exact orientation; graph capture stores
        # position only, so join it from the dense worker trace below.
        trace_path = args.dense_capture / 'traces/worker_seed100.jsonl'
        traces = getattr(main, '_traces', None)
        if traces is None:
            traces = {}
            with trace_path.open() as trace_stream:
                for line in trace_stream:
                    t = json.loads(line)
                    traces[(str(t['episode_id']), int(t['high_level_step']))] = t
            main._traces = traces
        trace = traces[(ep, step)]
        pre = unpack(trace['pre_pose'])
        position = np.asarray(pre['position'], dtype=np.float32)
        rotation = np.asarray(pre['rotation'], dtype=np.float32)
        for local_index, entry in enumerate(geometry_for_entries(new_entries)):
            candidate_pos = estimate_cand_pos(
                position, rotation, [entry['angle']], [entry['distance']]
            )[0]
            action = {
                'act': 4,
                'cur_vp': row['current_vp'],
                'front_vp': row['current_vp'],
                'front_pos': pack(position),
                'ghost_vp': 'dense_{}_{}_{}'.format(ep, step, local_index),
                'ghost_pos': pack(candidate_pos),
                'back_path': [],
                'tryout': False,
            }
            cases.append({
                'case_id': 'dense_{}_{}_{}'.format(ep, step, local_index),
                'episode_id': ep,
                'scene_id': row['scene_id'],
                'high_level_step': step,
                'mode': 'dense_action',
                'action': action,
                'candidate_source': 'A1_new_same_heatmap_union',
                'candidate_provenance': entry,
                'state_reasons': state.get('reasons', []),
            })
            if args.max_cases > 0 and len(cases) >= args.max_cases:
                break
        if args.max_cases > 0 and len(cases) >= args.max_cases:
            break
    base_cfg = json.loads(args.experiment_config.read_text()) if args.experiment_config else {}
    plan = {
        'config': {
            'experiment_id': 'GATE-A-DENSE-FULL-RETURN-{}-001'.format(base_cfg.get('split', 'train').upper()),
            'source_root': str(args.dense_capture.parent),
            'probe': args.probe,
            'noninterference': args.noninterference,
            'failure_subset_dataset': str(args.failure_subset_dataset or 'research/results/critical_causal_train64/failure_routes_9.json.gz'),
            'protocol': 'research/GATE_A_DENSE_CANDIDATE_PROTOCOL.md',
            'seed': int(base_cfg.get('seed', 100)), 'max_high_level_decisions': 15,
            'max_interrupt_routes': 0,
            'training': False, 'oracle_analysis_only': True,
        },
        'failed_episodes': critical['failed_episodes'],
        'states': critical['states'],
        'cases': cases,
        'candidate_sets': {'A0': 'native graph options from dense capture', 'A1': 'A0 union frozen dense NMS proposals', 'A2': None},
        'frozen_before_outcomes': True,
        'privileged_analysis_only': True,
        'scope': 'critical states from the audited unresolved train64 cohort; A1-new full-return branches only',
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, indent=2) + '\n')
    print(json.dumps({'cases': len(cases), 'states': len(plan['states']), 'routes': len(plan['failed_episodes'])}, indent=2))


if __name__ == '__main__':
    main()
