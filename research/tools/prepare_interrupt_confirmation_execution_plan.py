#!/usr/bin/env python3
"""Build the execution plan from the frozen, outcome-blind timing schedule."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--schedule', type=Path, required=True); ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args(); cfg = json.loads(args.config.read_text()); schedule = json.loads(args.schedule.read_text())
    root = Path('research/results/interrupt_confirmation_train128')
    source_root = root
    replay = sorted(str(r['episode_id']) for r in schedule['routes'])
    events = list(schedule['events']) + list(schedule['uniform_cuts'])
    # Smoke identities are selected by the frozen schedule order, before any native metric is read.
    smoke = []
    for e in [schedule['events'][0], schedule['events'][-1]]:
        smoke.append([e['episode_id'], e['high_level_step'], e['cut_primitive'], e['cut_kind'], e.get('timing_seed')])
    plan_cfg = {'experiment_id': cfg['experiment_id'], 'source_root': str(source_root),
                'failure_subset_dataset': cfg['dataset_path'], 'probe': 'probe_full002',
                'noninterference': 'noninterference_002', 'protocol': cfg['protocol'],
                'seed': cfg['seed'], 'max_high_level_decisions': cfg['max_high_level_decisions'],
                'max_interrupt_routes': len(replay), 'training': False, 'oracle_analysis_only': True,
                'base_config': cfg['base_config'], 'checkpoint': cfg['checkpoint']}
    paths = [args.config, args.schedule, Path(cfg['protocol']), Path(__file__),
             root/'capture_001/manifest.json', root/'capture_001/traces/worker_seed100.jsonl',
             root/'capture_001/graph_options/graph_options.jsonl', root/'noninterference_002/summary.json',
             root/'probe_full002/branch_traces.jsonl', root/'probe_full002/summary.json',
             root/'sampling_001/manifest.json', Path(cfg['dataset_path'])]
    plan = {'experiment_id': cfg['experiment_id'], 'status': 'execution_plan_frozen', 'config': plan_cfg,
            'replay_episodes': replay, 'failed_episodes': [], 'states': [], 'cases': [],
            'interrupt_events': events, 'smoke_event_keys': smoke,
            'scope': 'All phase-known event cuts plus three outcome-blind same-option timing cuts per exposed route; native controls for all 128 routes.',
            'outcome_blind_schedule': True, 'schedule_sha256': digest(args.schedule),
            'hashes': {str(p): digest(p) for p in paths}}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(plan, indent=2)+'\n')
    print(json.dumps({'routes': len(replay), 'events': len(events), 'smoke_event_keys': smoke}, indent=2))


if __name__ == '__main__': main()
