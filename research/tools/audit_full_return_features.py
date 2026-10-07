#!/usr/bin/env python3
"""Post-hoc, nonfitted audit of deployed features versus observed full returns."""
import argparse
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from audit_single_intervention import key, read_rows
from analyze_full_returns import classify
from run_option_capture import digest


def point(value):
    a = np.asarray(value['__array__'], dtype=float)
    if a.shape != (3,) or not np.isfinite(a).all():
        raise ValueError('Invalid graph coordinate')
    return a


def command_length(state, option):
    """Graph commanded polyline, not geodesic distance or realized motion."""
    action = option['action']
    if action['act'] != 4:
        raise ValueError('Feature comparison requires non-STOP options')
    points = [point(state['graph_position'])]
    points += [point(node['__tuple__'][1]) for node in action['back_path']]
    points += [point(action['front_pos']), point(action['ghost_pos'])]
    return sum(float(np.linalg.norm(b-a)) for a, b in zip(points, points[1:]))


def features(state, index):
    # Only pre-decision graph fields are accessed here. No trace/outcome input.
    if state['privileged_labels_in_features'] or state['budget_stop'] or state['no_vp_left']:
        raise ValueError('Invalid deployable feature state')
    opts = {o['index']: o for o in state['options'] if o['admissible']}
    valid = [i for i in range(len(state['logits'])) if state['mask'][i] and not state['visited'][i]]
    if set(valid) != set(opts):
        raise ValueError('Options and graph mask differ')
    logits = np.asarray([state['logits'][i] for i in valid], dtype=float)
    if not np.isfinite(logits).all():
        raise ValueError('Nonfinite valid logit')
    selected = state['effective_index']
    if selected != valid[int(np.argmax(logits))] or index not in opts:
        raise ValueError('Unexpected baseline argmax or invalid alternative')
    o, native = opts[index], opts[selected]
    probabilities = np.exp(logits - np.max(logits))
    probabilities /= probabilities.sum()
    entropy = -float(np.sum(probabilities * np.log(np.maximum(probabilities, 1e-300))))
    return {'logit_gap': float(native['logit'] - o['logit']),
            'normalized_entropy': entropy / math.log(len(valid)) if len(valid) > 1 else 0.,
            'commanded_polyline_delta_m': command_length(state, o) - command_length(state, native),
            'back_path_nodes_delta': len(o['action']['back_path']) - len(native['action']['back_path']),
            'current_proposal': bool(o['current_proposal'])}


def spearman(x, y):
    if len(x) < 3 or len(set(x)) < 2 or len(set(y)) < 2:
        return None
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, required=True)
    args = p.parse_args()
    cfg = json.loads(args.config.read_text())
    root, out = Path(cfg['source_root']), Path(cfg['output_dir'])
    out.mkdir(parents=True, exist_ok=False)
    graph_path = root / 'capture_001/graph_options/graph_options.jsonl'
    sources = [args.config, Path(__file__), Path(cfg['analysis_path']), graph_path]
    analysis = json.loads(Path(cfg['analysis_path']).read_text())
    # Verify the exact accepted traces, schedules and metric reconstructions.
    for path, expected in analysis['hashes'].items():
        if digest(path) != expected:
            raise ValueError('Changed accepted analysis source: ' + path)
    graphs = read_rows(graph_path)
    outcomes = {(r['kind'], r['episode_id']): r for r in analysis['events']}
    events = []
    for kind in ['top_logit', 'seeded_graph_id']:
        schedule_path = root / ('schedule_' + kind + '_001.json')
        sources.append(schedule_path)
        schedule = json.loads(schedule_path.read_text())
        for path, expected in schedule['hashes'].items():
            if digest(path) != expected:
                raise ValueError('Changed frozen schedule source: ' + path)
        for event in schedule['events']:
            row = graphs[key(event)]
            options = {o['index']: o for o in row['options']}
            for role in ['baseline', 'alternative']:
                if options[event[role + '_index']]['action'] != event[role + '_action']:
                    raise ValueError('Action dictionary mismatch')
            outcome = outcomes[(kind, event['episode_id'])]
            if outcome['step'] != event['high_level_step'] or outcome['scene_id'] != event['scene_id']:
                raise ValueError('Outcome/state mismatch')
            events.append({'kind': kind, 'episode_id': event['episode_id'], 'scene_id': event['scene_id'],
                           'step': event['high_level_step'], 'alternative_index': event['alternative_index'],
                           'features': features(row, event['alternative_index']),
                           'outcome': outcome['alternative'], 'baseline': outcome['baseline'],
                           'delta': outcome['delta'], 'category': outcome['category']})
    if len(events) != 32 or len(outcomes) != 32:
        raise ValueError('This config requires all 32 observed alternatives')
    by_kind = {k: {r['episode_id']: r for r in events if r['kind'] == k}
               for k in ['top_logit', 'seeded_graph_id']}
    pairs = []
    for ep, top in sorted(by_kind['top_logit'].items()):
        other = by_kind['seeded_graph_id'][ep]
        if (top['scene_id'], top['step']) != (other['scene_id'], other['step']):
            raise ValueError('Unmatched candidate comparison')
        delta = {m: top['outcome'][m] - other['outcome'][m] for m in cfg['metrics']}
        pairs.append({'episode_id': ep, 'scene_id': top['scene_id'], 'top_minus_seeded': delta,
                      'top_category_vs_seeded': classify(delta)})
    scenes = sorted(set(r['scene_id'] for r in pairs))
    rng = np.random.RandomState(cfg['bootstrap_seed'])
    draws = rng.randint(0, len(scenes), (cfg['bootstrap_samples'], len(scenes)))
    comparison = {}
    for m in cfg['metrics']:
        means = np.asarray([np.mean([r['top_minus_seeded'][m] for r in pairs if r['scene_id'] == s]) for s in scenes])
        comparison[m] = {'mean_top_minus_seeded': float(means.mean()),
                         'scene_bootstrap_95': np.percentile(means[draws].mean(axis=1), [2.5, 97.5]).tolist()}
    correlations = {}
    for kind in ['top_logit', 'seeded_graph_id', 'pooled']:
        subset = [r for r in events if kind == 'pooled' or r['kind'] == kind]
        correlations[kind] = {f: {m: spearman([r['features'][f] for r in subset],
                                              [r['delta'][m] for r in subset])
                                   for m in cfg['metrics']} for f in cfg['features']}
    result = {'experiment_id': cfg['experiment_id'], 'status': 'complete', 'post_hoc': True,
              'no_model_trained': True, 'no_new_rollout': True, 'routes': 16, 'scenes': len(scenes),
              'events': events, 'pairs': pairs, 'paired_comparison': comparison,
              'correlations_exploratory_no_significance_claim': correlations,
              'limits': cfg['limits'], 'source_hashes': {str(s): digest(s) for s in sources}}
    (out / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    manifest = json.loads((root / 'capture_001/manifest.json').read_text())
    provenance = {'created_utc': datetime.now(timezone.utc).isoformat(), 'config': cfg,
                  'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip(),
                  'python': sys.version, 'argv': sys.argv, 'source_capture_manifest': manifest,
                  'analysis_execution': 'CPU only; no new checkpoint load or simulator run',
                  'source_hashes': result['source_hashes']}
    (out / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    (out / 'git_diff.patch').write_bytes(subprocess.check_output(['git', 'diff', '--binary']))
    (out / 'git_status.txt').write_bytes(subprocess.check_output(['git', 'status', '--porcelain']))
    print(json.dumps({'pairs': len(pairs), 'metrics': comparison,
                      'categories': {c: sum(r['top_category_vs_seeded'] == c for r in pairs)
                                     for c in ['dominates', 'dominated', 'mixed', 'tied']}}, indent=2))


if __name__ == '__main__':
    main()
