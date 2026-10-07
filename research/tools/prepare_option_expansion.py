#!/usr/bin/env python3
"""Freeze broader training coverage using metadata only; exclude pilot routes."""
import argparse
import collections
import gzip
import hashlib
import json
from pathlib import Path

from prepare_balanced_calibration import select, digest


def choose(episodes, excluded, scenes, routes_per_scene, seed):
    available = [e for e in episodes if (e['scene_id'], str(e['trajectory_id'])) not in excluded]
    grouped = collections.defaultdict(set)
    for e in available:
        grouped[e['scene_id']].add(str(e['trajectory_id']))
    eligible = [s for s, routes in grouped.items() if len(routes) >= routes_per_scene]
    ordered = sorted(eligible, key=lambda s: hashlib.sha256('{}|{}'.format(seed, s).encode()).hexdigest())
    if len(ordered) < scenes:
        raise ValueError('Insufficient eligible scenes')
    selected_scenes = ordered[:scenes]
    return select([e for e in available if e['scene_id'] in selected_scenes], routes_per_scene, seed)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    cfg = json.loads(args.config.read_text())
    sampling = cfg['sampling']
    source = Path(sampling['source'])
    data = json.load(gzip.open(str(source), 'rt'))
    pilot_path = Path(sampling['exclude_manifest'])
    pilot = json.loads(pilot_path.read_text())
    excluded = {(e['scene_id'], str(e['trajectory_id'])) for e in pilot['episodes']}
    rows = choose(data['episodes'], excluded, sampling['scenes'], sampling['routes_per_scene'], sampling['seed'])
    routes = {(e['scene_id'], str(e['trajectory_id'])) for e in rows}
    if len(rows) != cfg['episodes'] or len(routes) != len(rows) or routes & excluded:
        raise ValueError('Wrong route count or pilot overlap')
    validation = {}
    for split in ['val_seen', 'val_unseen']:
        path = Path('data/datasets/R2R_VLNCE_v1-2_preprocessed_BERTidx') / split / (split + '_bertidx.json.gz')
        records = json.load(gzip.open(str(path), 'rt'))['episodes']
        overlap = routes & {(e['scene_id'], str(e['trajectory_id'])) for e in records}
        if overlap:
            raise ValueError('Validation route overlap')
        validation[split] = {'source': str(path), 'sha256': digest(path), 'route_overlap': len(overlap)}
    # Scene availability is checked before running, never used to select outcomes.
    scene_root = Path('data/scene_datasets')
    missing = sorted({e['scene_id'] for e in rows if not (scene_root / e['scene_id']).exists()})
    if missing:
        raise ValueError('Missing selected scenes: {}'.format(missing))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    output = Path(cfg['dataset_path'])
    if output.parent != args.output_dir or Path(cfg['sampling_manifest']).parent != args.output_dir:
        raise ValueError('Output does not match frozen config')
    subset = dict(data)
    subset['episodes'] = rows
    with output.open('xb') as stream:
        with gzip.GzipFile(fileobj=stream, mode='wb', filename='', mtime=0) as zipped:
            zipped.write(json.dumps(subset, ensure_ascii=False).encode('utf8'))
    manifest = {'experiment_id': cfg['experiment_id'], 'split': 'train', 'selection_seed': sampling['seed'],
                'scenes': sorted({e['scene_id'] for e in rows}), 'routes_per_scene': sampling['routes_per_scene'],
                'episode_count': len(rows), 'distinct_routes': len(routes), 'pilot_route_overlap': 0,
                'selection_rule': 'Exclude pilot routes; seeded SHA256 scene order, route order, instruction order; no outcomes',
                'source': str(source), 'source_sha256': digest(source), 'validation_disjointness': validation,
                'dataset_path': str(output), 'dataset_sha256': digest(output),
                'hashes': {str(path): digest(path) for path in [args.config, Path(__file__), pilot_path]},
                'episodes': [{'scene_id': e['scene_id'], 'episode_id': str(e['episode_id']),
                              'trajectory_id': str(e['trajectory_id'])} for e in rows]}
    Path(cfg['sampling_manifest']).write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({k: v for k, v in manifest.items() if k != 'episodes'}, indent=2))


if __name__ == '__main__':
    main()
