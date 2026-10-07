#!/usr/bin/env python3
"""Select routes before outcomes: equal scene coverage, one instruction per route."""
import argparse
import collections
import gzip
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select(episodes, routes_per_scene, seed):
    grouped = collections.defaultdict(lambda: collections.defaultdict(list))
    for episode in episodes:
        grouped[episode['scene_id']][str(episode['trajectory_id'])].append(episode)
    selected = []
    for scene in sorted(grouped):
        routes = grouped[scene]
        if len(routes) < routes_per_scene:
            raise ValueError('Not enough distinct routes in ' + scene)
        def key(value):
            return hashlib.sha256('{}|{}|{}'.format(seed, scene, value).encode()).hexdigest()
        for route in sorted(routes, key=key)[:routes_per_scene]:
            selected.append(min(routes[route], key=lambda ep: key('{}|{}'.format(route, ep['episode_id']))))
    return selected


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--train-source', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--routes-per-scene', type=int, default=6)
    p.add_argument('--selection-seed', type=int, default=20261007)
    args = p.parse_args()
    if args.routes_per_scene < 1:
        raise ValueError('routes-per-scene must be positive')
    data = json.load(gzip.open(str(args.source), 'rt'))
    chosen = select(data['episodes'], args.routes_per_scene, args.selection_seed)
    train = json.load(gzip.open(str(args.train_source), 'rt'))
    train_routes = {(e['scene_id'], str(e['trajectory_id'])) for e in train['episodes']}
    chosen_routes = {(e['scene_id'], str(e['trajectory_id'])) for e in chosen}
    overlap = train_routes & chosen_routes
    if overlap:
        raise ValueError('Training route overlap: {}'.format(sorted(overlap)))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    output = args.output_dir / 'val_unseen_balanced66_bertidx.json.gz'
    # Preserve every source episode field and the existing token vocabulary.
    subset = dict(data)
    subset['episodes'] = chosen
    with output.open('xb') as stream:
        with gzip.GzipFile(fileobj=stream, mode='wb', filename='', mtime=0) as zipped:
            zipped.write(json.dumps(subset, ensure_ascii=False).encode('utf8'))
    manifest = {'source': str(args.source), 'source_sha256': digest(args.source),
                'train_source': str(args.train_source), 'train_source_sha256': digest(args.train_source),
                'dataset_path': str(output), 'dataset_sha256': digest(output),
                'selection_seed': args.selection_seed, 'routes_per_scene': args.routes_per_scene,
                'selection_rule': 'SHA256 seeded route order per scene, then SHA256 instruction order within route; no outcomes used',
                'source_episodes': len(data['episodes']), 'selected_episodes': len(chosen),
                'selected_routes': len(chosen_routes), 'train_route_overlap': len(overlap),
                'scene_counts': dict(collections.Counter(e['scene_id'] for e in chosen)),
                'episodes': [{'episode_id': str(e['episode_id']), 'scene_id': e['scene_id'],
                              'trajectory_id': str(e['trajectory_id'])} for e in chosen],
                'scope': 'Balanced diagnostic subset of val_unseen, not the full benchmark; fixed before collection',
                'source_script_sha256': digest(Path(__file__))}
    (args.output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({k: v for k, v in manifest.items() if k != 'episodes'}, indent=2))


if __name__ == '__main__':
    main()
