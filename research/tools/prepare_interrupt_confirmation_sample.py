#!/usr/bin/env python3
"""Outcome-blind route sampler with exact route identity exclusion."""
import argparse
import collections
import gzip
import hashlib
import json
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def norm(scene):
    return scene.split('data/scene_datasets/')[-1]


def route_key(e):
    return norm(e['scene_id']), str(e['trajectory_id'])


def records(path, source_by_episode):
    path = Path(path)
    if path.suffix == '.jsonl':
        raw = [json.loads(line) for line in path.open()]
    else:
        with (gzip.open(str(path), 'rt') if path.suffix == '.gz' else path.open()) as f:
            obj = json.load(f)
        raw = obj.get('route_episode_ids', obj.get('episodes', [])) if isinstance(obj, dict) else obj
        if isinstance(raw, int):
            raise ValueError('Route count without route identities: {}'.format(path))
    out = []
    for row in raw:
        if row.get('trajectory_id') is not None and row.get('scene_id'):
            out.append((norm(row['scene_id']), str(row['trajectory_id'])))
        elif row.get('episode_id') is not None and str(row['episode_id']) in source_by_episode:
            out.append(route_key(source_by_episode[str(row['episode_id'])]))
        else:
            raise ValueError('Cannot resolve route identity in {}: {}'.format(path, row))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--config', type=Path, required=True)
    args = ap.parse_args(); cfg = json.loads(args.config.read_text())
    source = Path(cfg['source'])
    with gzip.open(str(source), 'rt') as f: data = json.load(f)
    episodes = data['episodes']; source_by_episode = {str(e['episode_id']): e for e in episodes}
    excluded = set(); metadata = {}
    for name in cfg['excluded_route_sources']:
        p = Path(name)
        if not p.is_file(): raise ValueError('Missing exclusion source: '+name)
        local = set(records(p, source_by_episode)); excluded |= local
        metadata[name] = {'records': len(local), 'sha256': digest(p)}
    # Validation episodes are excluded by exact route identity as well.
    validation = {}
    for name in cfg['excluded_validation_sources']:
        p = Path(name); local = set(records(p, source_by_episode)); excluded |= local
        validation[name] = {'records': len(local), 'sha256': digest(p)}
    available = [e for e in episodes if route_key(e) not in excluded]
    grouped = collections.defaultdict(lambda: collections.defaultdict(list))
    for e in available: grouped[norm(e['scene_id'])][str(e['trajectory_id'])].append(e)
    scenes = [s for s, routes in grouped.items() if len(routes) >= cfg['routes_per_scene']]
    scenes.sort(key=lambda s: hashlib.sha256('{}|interrupt-confirmation|{}'.format(cfg['selection_seed'], s).encode()).hexdigest())
    scenes = scenes[:cfg['scenes']]
    if len(scenes) != cfg['scenes']: raise ValueError('Insufficient scenes after exact exclusions')
    selected = []
    for scene in scenes:
        routes = grouped[scene]
        ordered = sorted(routes, key=lambda t: hashlib.sha256('{}|{}|{}'.format(cfg['selection_seed'], scene, t).encode()).hexdigest())
        for trajectory in ordered[:cfg['routes_per_scene']]:
            chosen = sorted(routes[trajectory], key=lambda e: hashlib.sha256('{}|{}|{}'.format(cfg['selection_seed'], trajectory, e['episode_id']).encode()).hexdigest())[0]
            selected.append(chosen)
    selected_routes = {route_key(e) for e in selected}
    if len(selected) != cfg['scenes']*cfg['routes_per_scene'] or len(selected_routes) != len(selected): raise ValueError('Invalid selected route set')
    if selected_routes & excluded: raise ValueError('Selected route overlaps exclusion')
    for name in cfg['excluded_validation_sources']:
        p = Path(name); validation[name]['selected_overlap'] = len(selected_routes & set(records(p, source_by_episode)))
        if validation[name]['selected_overlap']: raise ValueError('Validation overlap')
    out = Path(cfg['dataset_path']); out.parent.mkdir(parents=True, exist_ok=False)
    subset = dict(data); subset['episodes'] = selected
    with out.open('xb') as raw:
        with gzip.GzipFile(fileobj=raw, mode='wb', filename='', mtime=0) as gz:
            gz.write(json.dumps(subset, ensure_ascii=False).encode('utf8'))
    for p in [args.config, Path(cfg['protocol']), Path(__file__), source]: metadata[str(p)] = {'sha256': digest(p)}
    manifest = {'experiment_id': cfg['experiment_id'], 'source': str(source), 'source_sha256': digest(source),
                'dataset_path': str(out), 'dataset_sha256': digest(out), 'selection_seed': cfg['selection_seed'],
                'scenes': scenes, 'routes_per_scene': cfg['routes_per_scene'], 'selected_routes': len(selected_routes),
                'selected_episodes': len(selected), 'excluded_routes': len(excluded),
                'selection_rule': 'metadata-only scene/trajectory order, one instruction per trajectory; no outcomes or graph branches',
                'route_overlap': 0, 'scene_counts': dict(collections.Counter(norm(e['scene_id']) for e in selected)),
                'episodes': [{'episode_id': str(e['episode_id']), 'scene_id': e['scene_id'], 'trajectory_id': str(e['trajectory_id'])} for e in selected],
                'excluded_route_sources': metadata, 'excluded_validation_sources': validation,
                'source_script_sha256': digest(Path(__file__)), 'config_sha256': digest(args.config)}
    manifest_path = Path(cfg['sampling_manifest']); manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({k: v for k, v in manifest.items() if k not in ['episodes', 'excluded_route_sources', 'excluded_validation_sources']}, indent=2))


if __name__ == '__main__': main()
