#!/usr/bin/env python3
"""Metadata-only new-scene sample for prospective continuation label checks."""
import argparse
import collections
import gzip
import hashlib
import json
from pathlib import Path

from prepare_balanced_calibration import select
from run_option_capture import digest


def normalize_scene(scene):
    return scene.split('data/scene_datasets/')[-1]


def choose_new_scenes(episodes, excluded, scene_count, routes_per_scene, seed):
    grouped = collections.defaultdict(set)
    for e in episodes:
        scene = normalize_scene(e['scene_id'])
        if scene not in excluded:
            grouped[scene].add(str(e['trajectory_id']))
    eligible = [s for s, routes in grouped.items() if len(routes) >= routes_per_scene]
    ordered = sorted(eligible, key=lambda s: hashlib.sha256(
        '{}|continuation|{}'.format(seed, s).encode()).hexdigest())
    if len(ordered) < scene_count:
        raise ValueError('Insufficient previously uninspected training scenes')
    scenes = set(ordered[:scene_count])
    return select([e for e in episodes if normalize_scene(e['scene_id']) in scenes],
                  routes_per_scene, seed)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, required=True)
    args = p.parse_args()
    cfg = json.loads(args.config.read_text())
    sample = cfg['sampling']
    source = Path(sample['source'])
    with gzip.open(str(source), 'rt') as stream:
        data = json.load(stream)
    excluded, hashes, counts = set(), {}, {}
    for path in map(Path, sample['exclude_sources']):
        local = set()
        if path.suffix == '.jsonl':
            with path.open() as stream:
                for line in stream:
                    row = json.loads(line)
                    local.add(normalize_scene(row['scene_id']))
        else:
            local = {normalize_scene(e['scene_id']) for e in json.loads(path.read_text())['episodes']}
        hashes[str(path)] = digest(path)
        counts[str(path)] = len(local)
        excluded.update(local)
    rows = choose_new_scenes(data['episodes'], excluded, sample['scenes'],
                             sample['routes_per_scene'], sample['seed'])
    routes = {(normalize_scene(e['scene_id']), str(e['trajectory_id'])) for e in rows}
    scenes = {normalize_scene(e['scene_id']) for e in rows}
    if len(rows) != cfg['episodes'] or len(routes) != len(rows) or scenes & excluded:
        raise ValueError('Wrong population or prior-scene overlap')
    validation = {}
    for split in ['val_seen', 'val_unseen']:
        path = Path('data/datasets/R2R_VLNCE_v1-2_preprocessed_BERTidx') / split / (split + '_bertidx.json.gz')
        with gzip.open(str(path), 'rt') as stream:
            records = json.load(stream)['episodes']
        overlap = routes & {(normalize_scene(e['scene_id']), str(e['trajectory_id'])) for e in records}
        if overlap:
            raise ValueError('Validation route overlap')
        validation[split] = {'route_overlap': 0, 'source_sha256': digest(path)}
    missing = [s for s in scenes if not (Path('data/scene_datasets') / s).is_file()]
    if missing:
        raise ValueError('Missing frozen scenes: {}'.format(missing))
    output = Path(cfg['dataset_path'])
    directory = output.parent
    if Path(cfg['sampling_manifest']).parent != directory:
        raise ValueError('Manifest directory differs')
    directory.mkdir(parents=True, exist_ok=False)
    subset = dict(data)
    subset['episodes'] = rows
    with output.open('xb') as stream:
        with gzip.GzipFile(fileobj=stream, mode='wb', filename='', mtime=0) as zipped:
            zipped.write(json.dumps(subset, ensure_ascii=False).encode('utf8'))
    for path in [source, args.config, Path(__file__), Path(cfg['continuation_protocol'])]:
        hashes[str(path)] = digest(path)
    manifest = {'experiment_id': cfg['experiment_id'], 'split': 'train', 'seed': sample['seed'],
                'episode_count': len(rows), 'distinct_routes': len(routes), 'scenes': sorted(scenes),
                'excluded_scenes': sorted(excluded), 'excluded_source_scene_counts': counts,
                'prior_scene_overlap': 0, 'validation_disjointness': validation,
                'rule': 'Exclude prior training diagnostic scenes, seeded SHA256 scenes/routes/instructions; no outcomes',
                'dataset_path': str(output), 'dataset_sha256': digest(output), 'hashes': hashes,
                'episodes': [{k: str(e[k]) for k in ['scene_id', 'episode_id', 'trajectory_id']} for e in rows]}
    with Path(cfg['sampling_manifest']).open('x') as stream:
        json.dump(manifest, stream, indent=2); stream.write('\n')
    print(json.dumps({k: v for k, v in manifest.items() if k not in ['episodes', 'hashes']}, indent=2))


if __name__ == '__main__':
    main()
