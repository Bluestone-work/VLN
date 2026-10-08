#!/usr/bin/env python3
"""Confirmation variant: compare matched traced/untraced runs with frozen default tolerance."""
import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path


def compare_metrics(expected, actual, tolerance):
    issues = []
    differences = []
    if set(expected) != set(actual):
        issues.append({'kind': 'episode_ids', 'missing': sorted(set(expected) - set(actual)),
                       'extra': sorted(set(actual) - set(expected))})
    for episode in sorted(set(expected) & set(actual)):
        if set(expected[episode]) != set(actual[episode]):
            issues.append({'kind': 'metric_keys', 'episode_id': episode})
        for key in sorted(set(expected[episode]) & set(actual[episode])):
            a, b = expected[episode][key], actual[episode][key]
            if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
                raise TypeError('Non-numeric metric: {} {}'.format(episode, key))
            delta = abs(a - b)
            differences.append(delta)
            if not math.isfinite(a) or not math.isfinite(b) or delta > tolerance:
                issues.append({'kind': 'metric_value', 'episode_id': episode, 'metric': key,
                               'expected': a, 'actual': b, 'absolute_delta': delta})
    return {'passed': not issues and bool(expected), 'episodes': len(expected),
            'metric_comparisons': len(differences), 'max_absolute_delta': max(differences) if differences else None,
            'issues': issues}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--capture-dir', type=Path, required=True)
    p.add_argument('--control-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifests = {kind: json.loads((directory / 'manifest.json').read_text())
                 for kind, directory in [('capture', args.capture_dir), ('control', args.control_dir)]}
    capture, control = manifests['capture'], manifests['control']
    if not capture['trace'] or control['trace']:
        raise ValueError('Expected traced capture and untraced control')
    if capture['config'] != control['config']:
        raise ValueError('Different protocol configurations')
    for key in ['git_commit', 'hashes', 'hardware', 'python', 'environment']:
        if capture[key] != control[key]:
            raise ValueError('Provenance mismatch: ' + key)
    allowed = {'ENV_NAME'}
    if capture['config'].get('capture_graph_options'):
        if (capture['overrides'].get('TRAINER_NAME') != 'SS-ETP-OptionCapture'
                or control['overrides'].get('TRAINER_NAME') != 'SS-ETP'):
            raise ValueError('Unexpected graph capture/control trainer combination')
        allowed.add('TRAINER_NAME')
    co = {k: v for k, v in capture['overrides'].items() if k not in allowed}
    bo = {k: v for k, v in control['overrides'].items() if k not in allowed}
    if co != bo:
        raise ValueError('Overrides differ beyond the validated tracing classes')
    sources, data = {}, {}
    for kind, manifest in manifests.items():
        directory = Path('data/logs/eval_results') / manifest['exp_name']
        data[kind] = {}
        for metric_type, pattern in [('episodes', 'stats_ep_*.json'), ('aggregate', 'stats_ckpt_*.json')]:
            paths = list(directory.glob(pattern))
            if len(paths) != 1:
                raise ValueError('Expected exactly one {} result in {}'.format(metric_type, directory))
            path = paths[0]
            key = '{}_{}'.format(kind, metric_type)
            sources[key] = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
            shutil.copyfile(str(path), str(args.output_dir / (key + '.json')))
            data[kind][metric_type] = json.loads(path.read_text())
    tolerance = capture['config'].get('gates', {}).get('baseline_episode_metrics_tolerance', 1e-6)
    ep = compare_metrics(data['control']['episodes'], data['capture']['episodes'], tolerance)
    agg = compare_metrics({'aggregate': data['control']['aggregate']}, {'aggregate': data['capture']['aggregate']}, tolerance)
    if ep['episodes'] != capture['config']['episodes']:
        raise ValueError('Completed episode count differs from requested protocol')
    summary = {'experiment_id': capture['config']['experiment_id'],
               'baseline_noninterference_gate_passed': ep['passed'] and agg['passed'],
               'episode_comparison': ep, 'aggregate_comparison': agg,
               'sources': sources, 'config': capture['config'], 'git_commit': capture['git_commit'],
               'metrics': data['capture']['aggregate'],
               'limits': 'Metric equality tests tracing only; does not establish alternative-action validity or runtime neutrality.'}
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
