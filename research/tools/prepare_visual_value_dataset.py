#!/usr/bin/env python3
"""Prepare a state-balanced, current-node-only visual value dataset.

The labels are exact native full-return labels. Features use only the current
node's pre-action RGB-D sector and frozen native graph fields; endpoint images,
goal geometry and outcomes are never loaded as features.
"""
import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np


def array_value(v):
    if isinstance(v, dict) and '__array__' in v:
        return v['__array__']
    return v


def digest(a):
    return hashlib.sha256(np.asarray(a).tobytes()).hexdigest()


def visual_vector(rgb, depth):
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[-1] < 3:
        raise ValueError('RGB sector has unexpected shape {}'.format(rgb.shape))
    rgb = rgb[..., :3].astype(np.float32) / 255.0
    depth = np.asarray(depth).squeeze().astype(np.float32)
    if depth.ndim != 2:
        raise ValueError('Depth sector has unexpected shape {}'.format(depth.shape))
    # Low-dimensional, deterministic features: spatial color layout and depth
    # affordance statistics. No pretrained semantic model is used here.
    pooled = cv2.resize(rgb, (4, 4), interpolation=cv2.INTER_AREA).reshape(-1)
    stats = np.r_[rgb.mean((0, 1)), rgb.std((0, 1))]
    hist = []
    for c in range(3):
        hist.extend(np.histogram(rgb[..., c], bins=8, range=(0., 1.), density=False)[0] / float(rgb.shape[0] * rgb.shape[1]))
    finite = np.isfinite(depth) & (depth > 0)
    dv = depth[finite]
    if len(dv):
        dstat = [float(np.mean(dv)), float(np.std(dv)), float(np.percentile(dv, 10)),
                 float(np.percentile(dv, 50)), float(np.percentile(dv, 90)), float(np.mean(finite))]
    else:
        dstat = [0.] * 6
    dsmall = np.nan_to_num(depth, nan=0., posinf=0., neginf=0.)
    dsmall = cv2.resize(dsmall, (4, 4), interpolation=cv2.INTER_AREA).reshape(-1)
    return np.asarray(list(pooled) + list(stats) + hist + dstat + list(dsmall), dtype=np.float32)


def native_block(f):
    return np.asarray([f['native_logit'], f['native_selected_logit'],
                       f['native_logit_margin'], float(f['is_stop']),
                       float(f['current_proposal']), f['graph_size'],
                       f['back_path_length']] + list(f['native_embedding']), dtype=np.float32)


def geometry_block(f):
    bearing = f.get('bearing_deg')
    if bearing is None:
        return np.zeros(4, dtype=np.float32)
    theta = math.radians(float(bearing))
    return np.asarray([math.sin(theta), math.cos(theta), f['distance'], 0.], dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pairs', type=Path, required=True)
    ap.add_argument('--graph', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    pairs = [json.loads(x) for x in args.pairs.open() if x.strip()]
    graph = {}
    for line in args.graph.open():
        row = json.loads(line)
        graph[(str(row['episode_id']), int(row['high_level_step']))] = row
    groups = defaultdict(list)
    for row in pairs:
        if row['label'] == 'UNCERTAIN':
            continue
        groups[(row['scene_id'], str(row['episode_id']), int(row['high_level_step']))].append(row)
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    vectors = []
    rows = []
    missing = []
    for state_key, state_rows in sorted(groups.items()):
        scene, episode, step = state_key
        g = graph.get((episode, step))
        if g is None or 'semantic_visual' not in g:
            missing.append({'state': state_key, 'reason': 'missing_graph_or_visual_record'})
            continue
        files = g['semantic_visual']['files']
        for row in state_rows:
            f = row['features']
            sector = f.get('sector_deg')
            if sector is None:
                visual = np.zeros(100, dtype=np.float32)
                visual_valid = False
            else:
                rgb_key = 'rgb_{}'.format(int(sector)) if int(sector) else 'rgb'
                depth_key = 'depth_{}'.format(int(sector)) if int(sector) else 'depth'
                rgb_path = Path(files[rgb_key])
                depth_path = Path(files[depth_key])
                if not rgb_path.exists() or not depth_path.exists():
                    missing.append({'pair_id': row['pair_id'], 'reason': 'missing_sector_file',
                                    'rgb': str(rgb_path), 'depth': str(depth_path)})
                    continue
                rgb = np.load(str(rgb_path), allow_pickle=False)
                depth = np.load(str(depth_path), allow_pickle=False)
                # Recheck the recorded source hashes before using the view.
                expected_rgb = g['semantic_visual']['observation_hashes'][rgb_key]
                expected_depth = g['semantic_visual']['observation_hashes'][depth_key]
                if digest(rgb) != expected_rgb or digest(depth) != expected_depth:
                    raise ValueError('Observation hash mismatch for {}'.format(row['pair_id']))
                visual = visual_vector(rgb, depth)
                visual_valid = True
            native = native_block(f)
            geometry = geometry_block(f)
            key = '{}__{}__{}'.format(episode, step, row['pair_id'])
            row_out = {'pair_id': row['pair_id'], 'scene_id': scene,
                       'episode_id': episode, 'high_level_step': step,
                       'label': row['label'], 'native_action_index': row['alternative_index'],
                       'native_selected_index': row['native_index'], 'sector_deg': sector,
                       'visual_valid': visual_valid, 'state_key': '{}|{}|{}'.format(scene, episode, step),
                       'source_pair': str(args.pairs)}
            row_out['state_size'] = len(state_rows)
            row_out['state_weight'] = 1.0 / len(state_rows)
            vectors.append((native, geometry, visual, row_out))
            rows.append(row_out)
    if missing:
        (output / 'missing.json').write_text(json.dumps(missing, indent=2) + '\n')
    if not vectors:
        raise RuntimeError('No usable rows')
    native = np.stack([x[0] for x in vectors])
    geometry = np.stack([x[1] for x in vectors])
    visual = np.stack([x[2] for x in vectors])
    labels = np.asarray([x[3]['label'] == 'BENEFIT' for x in vectors], dtype=np.uint8)
    np.savez_compressed(str(output / 'features.npz'), native=native, geometry=geometry,
                        visual=visual, labels=labels)
    for row in rows:
        row['index'] = rows.index(row)
    (output / 'rows.jsonl').write_text(''.join(json.dumps(x, sort_keys=True) + '\n' for x in rows))
    scenes = sorted(set(x['scene_id'] for x in rows))
    states = sorted(set(x['state_key'] for x in rows))
    report = {'schema_version': 1, 'source_pairs': str(args.pairs),
              'source_graph': str(args.graph), 'rows': len(rows), 'states': len(states),
              'scenes': len(scenes), 'scene_ids': scenes,
              'labels': Counter(x['label'] for x in rows),
              'benefit_states': len(set(x['state_key'] for x in rows if x['label'] == 'BENEFIT')),
              'mixed_states': sum(set(x['label'] for x in state_rows) == {'BENEFIT', 'HARM'}
                                  for state_rows in groups.values()),
              'visual_valid_rows': sum(x['visual_valid'] for x in rows),
              'stop_rows': sum(not x['visual_valid'] for x in rows),
              'feature_dimensions': {'native': int(native.shape[1]), 'geometry': int(geometry.shape[1]),
                                    'visual': int(visual.shape[1])},
              'split_protocol': 'leave-one-scene-out; rows from one state never split across folds',
              'training_weight_protocol': 'inverse candidate count per state, then class balance inside each training fold',
              'excluded_uncertain': sum(1 for x in pairs if x['label'] == 'UNCERTAIN'),
              'missing_records': len(missing),
              'label_features_separation': 'causal labels are targets only; no outcome/endpoint/reference/goal feature'}
    (output / 'dataset_manifest.json').write_text(json.dumps(report, indent=2, default=list) + '\n')
    print(json.dumps(report, indent=2, default=list))


if __name__ == '__main__':
    main()
