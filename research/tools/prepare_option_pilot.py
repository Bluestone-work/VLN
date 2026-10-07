#!/usr/bin/env python3
"""Freeze an eight-route, four-scene training pilot before option outcomes."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

from prepare_balanced_calibration import select, digest


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--seed', type=int, default=20261007)
    args = p.parse_args()
    source = Path('data/datasets/R2R_VLNCE_v1-2_preprocessed_BERTidx/train/train_bertidx.json.gz')
    data = json.load(gzip.open(str(source), 'rt'))
    scenes = sorted({r['scene_id'] for r in data['episodes']},
                    key=lambda x: hashlib.sha256('{}|{}'.format(args.seed, x).encode()).hexdigest())[:4]
    rows = select([e for e in data['episodes'] if e['scene_id'] in scenes], 2, args.seed)
    assert len(rows) == 8 and len({(r['scene_id'], r['trajectory_id']) for r in rows}) == 8
    args.output_dir.mkdir(parents=True, exist_ok=False)
    output = args.output_dir / 'train_pilot8_bertidx.json.gz'
    selected = dict(data)
    selected['episodes'] = rows
    with output.open('xb') as stream:
        with gzip.GzipFile(fileobj=stream, mode='wb', filename='', mtime=0) as zipped:
            zipped.write(json.dumps(selected, ensure_ascii=False).encode('utf8'))
    manifest = {'split': 'train', 'selection_seed': args.seed, 'scenes': scenes,
                'routes_per_scene': 2, 'episode_count': 8, 'distinct_routes': 8,
                'selection_rule': 'Seeded SHA256 scene order, route order, then one instruction per route; no outcomes',
                'source': str(source), 'source_sha256': digest(source),
                'dataset_path': str(output), 'dataset_sha256': digest(output),
                'source_script_sha256': digest(Path(__file__)),
                'episodes': [{'scene_id': r['scene_id'], 'episode_id': str(r['episode_id']),
                              'trajectory_id': str(r['trajectory_id'])} for r in rows]}
    (args.output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
