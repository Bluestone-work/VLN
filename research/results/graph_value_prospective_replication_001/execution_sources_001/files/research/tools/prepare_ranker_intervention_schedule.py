#!/usr/bin/env python3
"""One first-disagreement intervention, or matched random control; no labels."""
import argparse
import collections
import hashlib
import json
import random
from pathlib import Path
import numpy as np
from graph_value_feasibility import action_features, FEATURE_NAMES


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def admissible_moves(row):
    return [o for o in row['options'] if o.get('admissible') and o['index'] > 0
            and o['action']['act'] == 4 and not row['visited'][o['index']]
            and row['mask'][o['index']]]


def first_disagreement(rows, model):
    if model['feature_names'] != list(FEATURE_NAMES):
        raise ValueError('Feature schema mismatch')
    w = np.asarray(model['weights_standardized'])
    mean, scale = np.asarray(model['mean']), np.asarray(model['scale'])
    if w.shape != mean.shape or mean.shape != scale.shape or np.any(scale <= 0):
        raise ValueError('Invalid model dimensions/normalization')
    for row in sorted(rows, key=lambda x: x['high_level_step']):
        if row['effective_index'] <= 0 or row['budget_stop'] or row['no_vp_left']:
            continue
        options = admissible_moves(row)
        if len(options) < 2:
            continue
        native = next(o for o in options if o['index'] == row['effective_index'])
        scored = [(float(w.dot((action_features(row, o) - mean) / scale)), o) for o in options]
        if not all(np.isfinite(score) for score, _ in scored):
            raise ValueError('Nonfinite candidate score')
        _, best = max(scored, key=lambda x: (x[0], -x[1]['index']))
        if best['index'] != native['index']:
            return row, native, best
    return None


def main():
    ap = argparse.ArgumentParser()
    for name in ['root', 'model', 'config', 'output']:
        ap.add_argument('--' + name, type=Path, required=True)
    ap.add_argument('--kind', choices=['learned', 'random'], default='learned')
    ap.add_argument('--seed', type=int, default=20261021)
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text())
    model = json.loads(args.model.read_text())
    sample_path = Path(cfg['sampling_manifest'])
    sample = json.loads(sample_path.read_text())
    graph_path = args.root / 'capture_001/graph_options/graph_options.jsonl'
    graphs = collections.defaultdict(list)
    for line in graph_path.open():
        row = json.loads(line)
        graphs[str(row['episode_id'])].append(row)
    events, omitted, eligible = [], [], []
    for entry in sample['episodes']:
        ep = str(entry['episode_id'])
        if not graphs[ep]:
            raise ValueError('Missing episode ' + ep)
        chosen = first_disagreement(graphs[ep], model)
        if chosen is None:
            omitted.append({'episode_id': ep, 'reason': 'no_ranker_disagreement'})
            continue
        row, native, best = chosen
        eligible.append({'episode_id': ep, 'high_level_step': row['high_level_step']})
        if args.kind == 'random':
            salt = '{}|{}|{}|{}'.format(args.seed, row['scene_id'], ep, row['high_level_step'])
            rng = random.Random(int(hashlib.sha256(salt.encode()).hexdigest(), 16))
            best = rng.choice(sorted(admissible_moves(row), key=lambda o: o['index']))
        if best['index'] == native['index']:
            omitted.append({'episode_id': ep, 'reason': 'random_chose_native',
                            'high_level_step': row['high_level_step']})
            continue
        events.append({'scene_id': row['scene_id'], 'episode_id': ep,
                       'high_level_step': row['high_level_step'], 'trajectory_id': str(row['trajectory_id']),
                       'baseline_index': native['index'], 'alternative_index': best['index'],
                       'baseline_action': native['action'], 'alternative_action': best['action'],
                       'expected_graph_ids': row['graph_ids'], 'expected_graph_position': row['graph_position'],
                       'selection_kind': args.kind, 'native_logit': native['logit'],
                       'alternative_logit': best['logit']})
    protocol = Path('research/GRAPH_VALUE_ROUTE_INTERVENTION_PROTOCOL.md')
    paths = [args.config, args.model, graph_path, sample_path, protocol, Path(__file__),
             Path('research/tools/graph_value_feasibility.py'), args.root / 'capture_001/manifest.json']
    result = {'experiment_id': cfg['experiment_id'] + '-' + args.kind.upper() + '-' + str(args.seed),
              'status': 'frozen_before_this_arm_continuation', 'analysis_only_oracle': False,
              'learned_model_evaluated': args.kind == 'learned', 'kind': args.kind, 'random_seed': args.seed,
              'expected_event_count': len(events), 'base_experiment_config': str(args.config),
              'protocol': str(protocol), 'selection_rule': 'First ranker/native disagreement; same states for random; preserve native STOP; no outcome features',
              'events': events, 'eligible_states': eligible, 'omitted_routes_retained_as_controls': omitted,
              'limits': 'Exploratory development follow-up after target census inspection; route-disjoint, scene-overlapping, not checkpoint-training holdout or benchmark.',
              'hashes': {str(p): digest(p) for p in paths}}
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2); stream.write('\n')
    print(json.dumps({'kind': args.kind, 'events': len(events), 'omitted': len(omitted),
                      'eligible_states': len(eligible), 'scenes': len({e['scene_id'] for e in events})}))


if __name__ == '__main__':
    main()
