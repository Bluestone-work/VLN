#!/usr/bin/env python3
"""Freeze event and same-option uniform timing schedules from native prefixes."""
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


def keyed(seed, row, cut):
    value = '{}|{}|{}|{}|{}'.format(seed, row['scene_id'], row['episode_id'], row['high_level_step'], cut)
    return hashlib.sha256(value.encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--capture-dir', type=Path, required=True); ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args(); cfg = json.loads(args.config.read_text())
    traces = []
    path = next(args.capture_dir.joinpath('traces').glob('worker_seed*.jsonl'))
    with path.open() as f:
        for line in f: traces.append(json.loads(line))
    by_episode = {}
    for row in traces:
        by_episode.setdefault(str(row['episode_id']), []).append(row)
    sample = json.loads(Path(cfg['sampling_manifest']).read_text())
    expected = {str(e['episode_id']) for e in sample['episodes']}
    assert set(by_episode) == expected
    for ep in by_episode:
        rows = sorted(by_episode[ep], key=lambda r: r['high_level_step'])
        assert [r['high_level_step'] for r in rows] == list(range(len(rows)))
    events = []; unsupported = []; uniform = []
    # No final-return, goal, success, or reference-path fields are read below.
    for ep in sorted(by_episode):
        rows = by_episode[ep]; chosen = None
        for row in rows:
            action = row['action']
            if action['act'] != 4 or action.get('back_path') != []:
                continue
            primitives = row['primitives']
            collision_cuts = [i + 1 for i, p in enumerate(primitives)
                              if p['action'] == 1 and p['collided'] is True and i + 1 < len(primitives)]
            if collision_cuts:
                chosen = (row, min(collision_cuts)); break
        if chosen is None:
            unsupported.append({'episode_id': ep, 'reason': 'no phase-known empty-back-path non-final collision'})
            continue
        row, event_cut = chosen
        forward_cuts = [i + 1 for i, p in enumerate(row['primitives'])
                        if p['action'] == 1 and i + 1 < len(row['primitives'])]
        if len(forward_cuts) < 3:
            unsupported.append({'episode_id': ep, 'reason': 'fewer than three outcome-blind interior forward cuts',
                                'event_step': row['high_level_step'], 'event_cut': event_cut})
            continue
        event = {'episode_id': ep, 'scene_id': row['scene_id'], 'high_level_step': row['high_level_step'],
                 'cut_primitive': event_cut, 'cut_kind': 'event_first_forward_collision',
                 'action': row['action'], 'baseline_primitive_count': len(row['primitives']),
                 'empty_back_path': True}
        events.append(event)
        for seed in cfg['timing_seeds']:
            cut = sorted(forward_cuts, key=lambda x: keyed(seed, row, x))[0]
            uniform.append({'episode_id': ep, 'scene_id': row['scene_id'], 'high_level_step': row['high_level_step'],
                            'cut_primitive': cut, 'cut_kind': 'uniform_interior_forward', 'timing_seed': seed,
                            'action': row['action'], 'baseline_primitive_count': len(row['primitives']),
                            'empty_back_path': True, 'event_cut_primitive': event_cut})
    all_routes = [{'episode_id': ep, 'scene_id': rows[0]['scene_id'],
                   'native_decisions': len(rows), 'event_supported': any(e['episode_id'] == ep for e in events),
                   'uniform_supported': any(u['episode_id'] == ep for u in uniform)}
                  for ep, rows in sorted(by_episode.items())]
    assert len({e['episode_id'] for e in events}) == len(events)
    assert len(uniform) == len(events) * len(cfg['timing_seeds'])
    plan = {'experiment_id': cfg['experiment_id'], 'status': 'schedule_frozen_before_intervention',
            'config': cfg, 'sample_manifest': sample, 'capture_dir': str(args.capture_dir),
            'native_trace': str(path), 'route_count': len(all_routes), 'routes': all_routes,
            'events': events, 'uniform_cuts': uniform, 'unsupported': unsupported,
            'selection_rule': 'first phase-known native final-ghost forward collision per route; uniform cut is SHA256 keyed by frozen timing seed over all interior forward commands in that same option',
            'outcome_fields_read': [], 'outcome_blind_schedule': True,
            'hashes': {str(p): digest(p) for p in [args.config, Path(cfg['protocol']), Path(__file__), path, Path(cfg['sampling_manifest'])]}}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(plan, indent=2)+'\n')
    print(json.dumps({'routes': len(all_routes), 'events': len(events), 'uniform_cuts': len(uniform),
                      'unsupported': len(unsupported), 'scenes': len({e['scene_id'] for e in events})}, indent=2))


if __name__ == '__main__': main()
