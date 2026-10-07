#!/usr/bin/env python3
"""Audit candidate identity and uncertainty; preserve all historical results.

Graph label aggregation below is a sensitivity analysis, NOT an estimate of
the actual merged-target/controller outcome. No score is deployed.
"""
import argparse
import collections
import gzip
import hashlib
import json
import math
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from train_embedding_ranker import fit_pairwise, fit_pointwise


def indexed_jsonl(path):
    rows = {}
    with open(str(path)) as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            key = (str(row['episode_id']), int(row['high_level_step']))
            if key in rows:
                raise ValueError('Duplicate decision {} in {}'.format(key, path))
            rows[key] = row
    return rows


def assert_geometry(waypoint, oracle, tolerance):
    delta = abs((float(waypoint['angle']) - float(oracle['angle']) + math.pi)
                % (2 * math.pi) - math.pi)
    if delta > tolerance or abs(float(waypoint['distance']) - float(oracle['distance'])) > tolerance:
        raise ValueError('Candidate geometry/order mismatch')


def cluster_ci(values, cluster_ids, seed, samples):
    """Resample entire clusters; estimate the decision-weighted mean delta."""
    groups = collections.defaultdict(list)
    for value, key in zip(values, cluster_ids):
        groups[key].append(float(value))
    sums = np.asarray([sum(v) for v in groups.values()])
    counts = np.asarray([len(v) for v in groups.values()])
    result = {'clusters': len(groups), 'decision_weighted_mean_m': float(np.mean(values)),
              'equal_cluster_mean_m': float(np.mean(sums / counts)), 'ci95_m': None}
    if len(groups) < 2:
        result['warning'] = 'Insufficient clusters; no cross-cluster confidence interval.'
        return result
    rng = np.random.RandomState(seed)
    draws = rng.randint(0, len(groups), size=(samples, len(groups)))
    estimates = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    result['ci95_m'] = np.percentile(estimates, [2.5, 97.5]).tolist()
    if len(groups) < 10:
        result['warning'] = 'Few clusters; interval is exploratory.'
    return result


def graph_pool(labels, scores, graph_ids, reduction):
    groups = collections.OrderedDict()
    for i, graph_id in enumerate(graph_ids):
        groups.setdefault(graph_id, []).append(i)
    functions = {'mean': np.mean, 'min': np.min, 'max': np.max}
    y, s = [], []
    for indices in groups.values():
        if not np.allclose(scores[indices], scores[indices[0]], rtol=0, atol=1e-7):
            raise ValueError('One graph ID has different scores')
        y.append(float(functions[reduction](labels[indices])))
        s.append(float(scores[indices[0]]))
    return np.asarray(y), np.asarray(s)


def load_dataset(config, tag):
    root = Path(config['input_root'])
    prefix = root / ('ranker_embedding_' + tag)
    oracle = indexed_jsonl(str(prefix) + '_oracle.jsonl')
    with gzip.open(config['dataset_template'].format(split=config['splits'][tag]), 'rt') as stream:
        episodes = {str(e['episode_id']): e for e in json.load(stream)['episodes']}
    states, seen = {}, set()
    counts = collections.Counter()
    all_episodes, all_scenes, all_routes = set(), set(), set()
    scene_episodes = collections.defaultdict(set)
    alias_spreads = []
    with open(str(prefix) + '_diagnostics.jsonl') as stream:
        for line in stream:
            d = json.loads(line)
            key = (str(d['episode_id']), int(d['high_level_step']))
            if key in seen:
                raise ValueError('Duplicate diagnostic {}'.format(key))
            seen.add(key)
            if key not in oracle:
                raise ValueError('Missing oracle {}'.format(key))
            o = oracle[key]
            scene = Path(d['scene_id']).stem
            ep = episodes[key[0]]
            if scene != Path(o['scene_id']).stem or scene != Path(ep['scene_id']).stem:
                raise ValueError('Scene mismatch {}'.format(key))
            if abs(d['distance_to_goal_before'] - o['distance_to_goal_before']) > 1e-5:
                raise ValueError('State goal-distance mismatch {}'.format(key))
            route = (scene, str(ep['trajectory_id']))
            all_episodes.add(key[0]); all_scenes.add(scene); all_routes.add(route)
            scene_episodes[scene].add(key[0])
            candidates = o['levels']['default']
            mappings = d['candidate_graph_mapping']
            if len(candidates) != len(mappings) or len(candidates) != len(d['waypoint_candidates']):
                raise ValueError('Candidate count mismatch {}'.format(key))
            rows = []
            for m, waypoint, candidate in zip(mappings, d['waypoint_candidates'], candidates):
                assert_geometry(waypoint, candidate, config['geometry_tolerance'])
                counts['checked_candidates'] += 1
                if not m['graph_valid'] or m['graph_visited']:
                    counts['ineligible_candidates'] += 1
                    continue
                embedding = np.asarray(m['graph_embedding'], dtype=np.float64)
                if embedding.shape != (768,) or not np.all(np.isfinite(embedding)):
                    raise ValueError('Bad embedding')
                if not math.isfinite(m['graph_logit']) or not math.isfinite(candidate['progress']):
                    raise ValueError('Nonfinite score/label')
                rows.append({'key': key, 'features': embedding, 'progress': candidate['progress'],
                             'graph_logit': m['graph_logit'], 'graph_id': m['graph_id']})
            if [int(m['candidate_index']) for m in mappings] != list(range(len(mappings))):
                raise ValueError('Candidate indices reordered')
            if not rows:
                counts['states_without_eligible_candidates'] += 1
                continue
            groups = collections.defaultdict(list)
            for r in rows:
                groups[r['graph_id']].append(r)
            state_alias = False
            for members in groups.values():
                counts['unique_eligible_graph_actions'] += 1
                if len(members) < 2:
                    continue
                state_alias = True
                counts['aliased_eligible_graph_groups'] += 1
                for m in members[1:]:
                    if not np.array_equal(m['features'], members[0]['features']) or m['graph_logit'] != members[0]['graph_logit']:
                        raise ValueError('Aliased graph features/logits disagree')
                spread = max(m['progress'] for m in members) - min(m['progress'] for m in members)
                alias_spreads.append(float(spread))
                for i, left in enumerate(members):
                    for right in members[i + 1:]:
                        if abs(left['progress'] - right['progress']) > 1e-8:
                            counts['identical_embedding_conflicting_label_pairs'] += 1
            counts['eligible_states_with_aliasing'] += int(state_alias)
            states[key] = {'key': key, 'episode': key[0], 'scene': scene,
                           'route': route, 'rows': rows}
    if seen != set(oracle):
        raise ValueError('Unpaired oracle decisions')
    counts.update({'raw_decisions': len(seen), 'eligible_decisions': len(states),
                   'episodes': len(all_episodes), 'scenes': len(all_scenes), 'routes': len(all_routes)})
    info = dict(counts)
    info['episodes_by_scene'] = {s: len(v) for s, v in sorted(scene_episodes.items())}
    info['alias_label_spread_mean_m'] = float(np.mean(alias_spreads)) if alias_spreads else None
    info['alias_label_spread_max_m'] = max(alias_spreads) if alias_spreads else None
    return [states[k] for k in sorted(states)], info


def evaluate_states(states, weights, mean, scale, config):
    output = {}
    for convention in ['raw_first', 'raw_last'] + config['aggregation_sensitivity']:
        losses = collections.defaultdict(list)
        hits = collections.defaultdict(list)
        for state in states:
            rows = state['rows']
            labels = np.asarray([r['progress'] for r in rows])
            # Match the legacy row-wise dot product exactly. Batched BLAS may
            # perturb tied alias scores at machine precision, changing argmax
            # among identical graph features with different candidate labels.
            scores = {name: np.asarray([float(np.dot((r['features'] - mean) / scale, w))
                                      for r in rows]) for name, w in weights.items()}
            scores['graph_logit'] = np.asarray([r['graph_logit'] for r in rows])
            ids = [r['graph_id'] for r in rows]
            for name, score in scores.items():
                y, s = labels, score
                if convention in config['aggregation_sensitivity']:
                    y, s = graph_pool(y, s, ids, convention)
                chosen = (len(s) - 1 - int(np.argmax(s[::-1]))) if convention == 'raw_last' else int(np.argmax(s))
                regret = float(y.max() - y[chosen])
                losses[name].append(regret)
                hits[name].append(float(regret <= 1e-8))
        methods = {}
        for name in losses:
            delta = np.asarray(losses[name]) - np.asarray(losses['graph_logit'])
            methods[name] = {'regret_mean_m': float(np.mean(losses[name])),
                             'top1_rate': float(np.mean(hits[name]))}
            if name != 'graph_logit':
                methods[name]['paired_regret_delta_vs_graph'] = {
                    unit: cluster_ci(delta, [s[unit] for s in states], config['bootstrap_seed'], config['bootstrap_samples'])
                    for unit in ['episode', 'route', 'scene']}
        output[convention] = methods
    return output


def sha256(path):
    h = hashlib.sha256()
    with open(str(path), 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    config = json.loads(args.config.read_text())
    (args.output_dir / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    train, train_info = load_dataset(config, config['train_tag'])
    records = [r for s in train for r in s['rows']]
    matrix = np.asarray([r['features'] for r in records])
    mean, scale = matrix.mean(0), matrix.std(0)
    scale[scale < 1e-8] = 1
    del matrix
    print('Verified training data: ' + json.dumps(train_info), flush=True)
    pointwise = fit_pointwise(records, mean, scale, config['ridge_lambda'])
    pairwise, pairs = fit_pairwise(records, mean, scale, config['ridge_lambda'])
    weights = {'pointwise_embedding': pointwise, 'pairwise_embedding': pairwise}
    np.savez_compressed(str(args.output_dir / 'frozen_rankers.npz'), mean=mean, scale=scale, **weights)
    result = {'experiment_id': config['experiment_id'], 'train': train_info,
              'pair_count_in_legacy_fit': pairs, 'evaluations': {},
              'interpretation': config['label_limit'], 'navigation_intervention': False}
    for tag in config['validation_tags']:
        states, coverage = load_dataset(config, tag)
        evaluations = evaluate_states(states, weights, mean, scale, config)
        result['evaluations'][tag] = {'coverage': coverage, 'label_conventions': evaluations}
        print(tag + ' ' + json.dumps({'coverage': coverage, 'raw_first': evaluations['raw_first']}), flush=True)
    (args.output_dir / 'metrics.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    paths = [args.config, Path(__file__), Path(__file__).with_name('train_embedding_ranker.py'), Path(config['checkpoint'])]
    for tag in [config['train_tag']] + config['validation_tags']:
        paths += [Path(config['input_root']) / ('ranker_embedding_' + tag + suffix)
                  for suffix in ['_diagnostics.jsonl', '_oracle.jsonl']]
        paths.append(Path(config['dataset_template'].format(split=config['splits'][tag])))
    manifest = {'time_utc': datetime.now(timezone.utc).isoformat(), 'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip(),
                'working_tree_dirty': bool(subprocess.check_output(['git', 'status', '--porcelain'])),
                'python': platform.python_version(), 'numpy': np.__version__, 'cpu': platform.processor(),
                'hashes_at_analysis_time': {str(p): sha256(p) for p in paths},
                'note': 'Hashes recorded now cannot reconstruct historical uncommitted source snapshots.'}
    (args.output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')


if __name__ == '__main__':
    main()
