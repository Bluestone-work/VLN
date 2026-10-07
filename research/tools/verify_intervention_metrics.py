#!/usr/bin/env python3
"""Independently reconstruct native final metrics from retained actual traces."""
import argparse
import gzip
import json
from pathlib import Path
import numpy as np
from fastdtw import fastdtw

from audit_single_intervention import read_rows, group_rows, one
from route_alignment import trace_path
from run_option_capture import digest


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((args.run_dir / 'manifest.json').read_text())
    raw_path = one(args.run_dir / 'traces', 'worker_seed*.jsonl')
    native_path = one(Path('data/logs/eval_results') / manifest['exp_name'], 'stats_ep_*.json')
    gt_path = Path('data/datasets/R2R_VLNCE_v1-2_preprocessed/train/train_gt.json.gz')
    native = json.loads(native_path.read_text())
    gt = json.load(gzip.open(str(gt_path), 'rt'))
    grouped = group_rows(read_rows(raw_path))
    if set(native) != set(grouped):
        raise ValueError('Incomplete metric/trace population')
    checks = []
    for episode, records in grouped.items():
        points = []
        for record in records:
            segment = trace_path(record)
            if points and not np.array_equal(points[-1], segment[0]):
                raise ValueError('Discontinuous actual path')
            points.extend(segment if not points else segment[1:])
        # Match the original Position measure and evaluator's dtype/arithmetic.
        path = np.asarray(points, dtype=np.float32)
        reference = np.asarray(gt[episode]['locations'], dtype=np.float64)
        length = float(np.linalg.norm(path[1:] - path[:-1], axis=1).sum())
        error = records[-1]['distance_after']
        success = 1. if error <= 3. else 0.
        distance = fastdtw(path, reference, dist=lambda a, b: np.linalg.norm(b - a))[0]
        ndtw = float(np.exp(-distance / (len(reference) * 3.)))
        shortest = records[0]['distance_before']
        metrics = {'distance_to_goal': error, 'success': success, 'path_length': length,
                   'spl': success * shortest / max(shortest, length), 'ndtw': ndtw, 'sdtw': ndtw * success}
        differences = {k: abs(v - native[episode][k]) for k, v in metrics.items()}
        checks.append({'episode_id': episode, 'reconstructed': metrics,
                       'differences': differences, 'passed': max(differences.values()) <= 1e-6})
    summary = {'episodes': len(checks), 'metric_comparisons': len(checks) * 6,
               'all_required_gates_passed': all(r['passed'] for r in checks),
               'max_absolute_delta': max(max(r['differences'].values()) for r in checks),
               'unchanged_native_success_distance_m': 3.,
               'hashes': {str(path): digest(path) for path in [Path(__file__), raw_path, native_path, gt_path]}}
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    (args.output_dir / 'episodes.json').write_text(json.dumps(checks, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
    if not summary['all_required_gates_passed']:
        raise ValueError('Counterfactual continuation native-metric reconstruction failed')


if __name__ == '__main__':
    main()
