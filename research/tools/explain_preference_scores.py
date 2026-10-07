#!/usr/bin/env python3
"""Conserve the frozen pair score under a descriptive feature-group split."""
import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from audit_preference_continuation_mechanisms import digest


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args(); cfg = json.loads(args.config.read_text())
    model = json.loads(Path(cfg['model']).read_text()); names = model['feature_names']
    group_names = [n for values in cfg['groups'].values() for n in values]
    if len(group_names) != len(set(group_names)) or set(group_names) != set(names):
        raise ValueError('Groups do not partition model features')
    w = np.asarray(model['weights_standardized']); scale = np.asarray(model['scale']); mean = np.asarray(model['mean'])
    if not (len(w) == len(scale) == len(mean) == len(names)) or np.any(scale <= 0):
        raise ValueError('Model shape/scale mismatch')
    root = Path(cfg['output']); root.mkdir(exist_ok=False)
    registration = {'created_utc': datetime.now(timezone.utc).isoformat(), 'config': cfg,
                    'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip(),
                    'hashes': {str(p): digest(p) for p in [args.config, Path(__file__), Path(cfg['model']), Path(cfg['events'])]}}
    with (root / 'specification.json').open('x') as stream:
        json.dump(registration, stream, indent=2); stream.write('\n')
    rows = []
    for event in json.loads(Path(cfg['events']).read_text()):
        if event['batch'] not in cfg['batches']:
            continue
        native = np.asarray([event['pre_action']['native_features'][n] for n in names])
        alternative = np.asarray([event['pre_action']['alternative_features'][n] for n in names])
        components = dict(zip(names, map(float, w * (alternative - native) / scale)))
        groups = {g: sum(components[n] for n in ns) for g, ns in cfg['groups'].items()}
        actual = float(w.dot((alternative - mean) / scale) - w.dot((native - mean) / scale))
        if abs(sum(groups.values()) - actual) > 1e-10 or actual < -1e-10:
            raise ValueError('Pair score does not agree with frozen choice')
        rows.append({'batch': event['batch'], 'episode_id': event['episode_id'],
                     'cost_reversal': event['cost_reversal'], 'groups': groups,
                     'features': components, 'frozen_pair_margin': actual,
                     'largest_positive_group': max(groups, key=groups.get)})
    summary = {}
    for name in cfg['batches']:
        summary[name] = {}
        for label, subset in [('all_events', [r for r in rows if r['batch'] == name]),
                              ('cost_reversals', [r for r in rows if r['batch'] == name and r['cost_reversal']])]:
            summary[name][label] = {'events': len(subset),
                'largest_positive_group_counts': {g: sum(r['largest_positive_group'] == g for r in subset) for g in cfg['groups']},
                'mean_group_contribution': {g: float(np.mean([r['groups'][g] for r in subset])) if subset else None for g in cfg['groups']}}
    result = {'experiment_id': cfg['experiment_id'], 'all_score_conservation_checks_passed': True,
              'batches': summary, 'events': rows, 'limits': cfg['limits']}
    with (root / 'summary.json').open('x') as stream:
        json.dump(result, stream, indent=2); stream.write('\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
