#!/usr/bin/env python3
"""Freeze the larger outcome-blind A0/A1 control replication cohort."""
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRAIN_SRC = ROOT / 'research/results/graph_option_train64/sampling_001/train64_bertidx.json.gz'
UNSEEN_SRC = ROOT / 'research/results/option_calibration/unseen_balanced66_sampling/val_unseen_balanced66_bertidx.json.gz'
GATE_A_TABLE = ROOT / 'research/results/gate_a_dense_candidate_combined/analysis_001/per_route.csv'
OUT = ROOT / 'research/results/gate_a_native_policy_replication_001'
CONFIG_DIR = ROOT / 'research/configs'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load(path):
    with gzip.open(path, 'rt') as stream:
        return json.load(stream)


def write_subset(data, ids, path):
    wanted = {str(x) for x in ids}
    episodes = [episode for episode in data['episodes'] if str(episode['episode_id']) in wanted]
    found = {str(episode['episode_id']) for episode in episodes}
    if found != wanted:
        raise RuntimeError('missing episode ids: %s' % sorted(wanted - found))
    with gzip.open(path, 'wt') as stream:
        json.dump({'episodes': episodes, 'instruction_vocab': data['instruction_vocab']}, stream)


def main():
    if OUT.exists() and not (OUT / 'manifest.json').exists():
        raise RuntimeError('incomplete replication output exists: %s' % OUT)
    OUT.mkdir(parents=True, exist_ok=True)
    gate_ids = {row['episode_id'] for row in __import__('csv').DictReader(GATE_A_TABLE.open())}
    source_specs = [('train', TRAIN_SRC), ('val_unseen', UNSEEN_SRC)]
    route_rows = []
    counts = {}
    for cohort, source in source_specs:
        data = load(source)
        selected = [episode for episode in data['episodes'] if str(episode['episode_id']) not in gate_ids]
        selected = sorted(selected, key=lambda episode: (str(episode['scene_id']), int(episode['episode_id'])))
        counts[cohort] = len(selected)
        for episode in selected:
            route_rows.append({
                'episode_id': str(episode['episode_id']),
                'trajectory_id': str(episode.get('trajectory_id', '')),
                'scene_id': episode['scene_id'],
                'cohort': cohort,
                'role': 'metadata_control_replication',
            })
        output_name = 'train_replication.json.gz' if cohort == 'train' else 'val_unseen_replication.json.gz'
        write_subset(data, [row['episode_id'] for row in route_rows if row['cohort'] == cohort], OUT / output_name)
    manifest = {
        'experiment_id': 'GATE-A-NATIVE-POLICY-REPLICATION-001',
        'selection_frozen_utc': '2026-10-09',
        'selection_rule': 'all routes in the already frozen outcome-blind train64 and unseen66 source samplers except the 31 Gate-A routes; sorted metadata only; no evaluator outcome read',
        'no_outcome_selection': True,
        'excluded_gate_a_route_count': len(gate_ids),
        'control_route_count': len(route_rows),
        'control_scene_count': len(set(row['scene_id'] for row in route_rows)),
        'counts_by_cohort': counts,
        'route_rows': route_rows,
        'source_manifests': [
            {'cohort': cohort, 'path': str(source), 'sha256': sha(source)}
            for cohort, source in source_specs
        ],
        'gate_a_table_sha256': sha(GATE_A_TABLE),
        'checkpoint': 'data/logs/checkpoints/release_r2r/ckpt.iter12000.pth',
        'seed': 100,
        'a0': {'max_predictions': 5, 'sigma': [7.0, 5.0]},
        'a1': {'construction': 'A0 union dense NMS from same heatmap', 'max_predictions': 12, 'sigma': [4.0, 3.0]},
        'controller': 'native ETPNav control; no reranker/oracle/recovery',
    }
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    cohort_config = {
        'experiment_id': manifest['experiment_id'],
        'manifest': str(OUT / 'manifest.json'),
        'checkpoint': manifest['checkpoint'], 'seed': manifest['seed'],
        'train_dataset': str(OUT / 'train_replication.json.gz'),
        'val_unseen_dataset': str(OUT / 'val_unseen_replication.json.gz'),
        'selection_frozen_before_a1': True,
    }
    config_path = CONFIG_DIR / 'GATE_A_NATIVE_POLICY_REPLICATION_001.json'
    config_path.write_text(json.dumps(cohort_config, indent=2) + '\n')
    for cohort, split, count in [('train', 'train', counts['train']), ('val_unseen', 'val_unseen', counts['val_unseen'])]:
        for arm in ('a0', 'a1'):
            config = {
                'experiment_id': '%s-%s-%s' % (manifest['experiment_id'], cohort.upper(), arm.upper()),
                'base_config': 'run_r2r/iter_train.yaml', 'checkpoint': manifest['checkpoint'],
                'split': split, 'episodes': count, 'seed': manifest['seed'],
                'dataset_path': cohort_config['train_dataset'] if cohort == 'train' else cohort_config['val_unseen_dataset'],
                'capture_graph_options': True, 'capture_waypoint_provenance': True,
                'action_abstraction': 'default' if arm == 'a0' else 'dense_native_union',
                'allow_sliding': True, 'tryout_effective': False, 'back_algo': 'control',
                'cohort': cohort, 'arm': arm, 'frozen_manifest': str(config_path),
            }
            (CONFIG_DIR / ('GATE_A_NATIVE_POLICY_REPLICATION_%s_%s.json' % (cohort.upper(), arm.upper()))).write_text(json.dumps(config, indent=2) + '\n')
    print(json.dumps({'control_routes': len(route_rows), 'control_scenes': manifest['control_scene_count'], 'counts_by_cohort': counts}, indent=2))


if __name__ == '__main__':
    main()
