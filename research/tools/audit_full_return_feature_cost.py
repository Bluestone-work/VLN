#!/usr/bin/env python3
"""Separate immediate and continuation costs before interpreting feature signals."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from audit_single_intervention import read_rows, group_rows, one
from audit_full_return_features import spearman
from decompose_continuation_costs import breakdown
from run_option_capture import digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    root = Path(cfg['source_root'])
    source = json.loads(Path(cfg['feature_result']).read_text())
    base_path = one(root / 'capture_001/traces', '*.jsonl')
    base = group_rows(read_rows(base_path))
    events, groups = [], {}
    paths = [args.config, Path(__file__), Path(cfg['feature_result']), base_path,
             Path('research/tools/decompose_continuation_costs.py'),
             Path('research/tools/audit_full_return_features.py')]
    for kind in ['top_logit', 'seeded_graph_id']:
        trace_path = one(root / (kind + '_enabled_' + cfg['accepted_tag']) / 'traces', '*.jsonl')
        new = group_rows(read_rows(trace_path))
        paths.append(trace_path)
        for event in [e for e in source['events'] if e['kind'] == kind]:
            ep, step = event['episode_id'], event['step']
            before, after = breakdown(base[ep], step), breakdown(new[ep], step)
            delta = {p: {m: after[p][m]-before[p][m] for m in before[p]} for p in before}
            for native, metric in [('primitive_action_count', 'primitives'), ('collision_events', 'collision_events')]:
                if sum(d[metric] for d in delta.values()) != event['delta'][native]:
                    raise ValueError('Cost accounting differs from accepted native metric')
            events.append({'kind': kind, 'episode_id': ep, 'scene_id': event['scene_id'],
                           'feature': event['features'][cfg['feature']], 'parts': delta,
                           'final_delta': event['delta']})
        subset = [r for r in events if r['kind'] == kind]
        groups[kind] = {'total_primitive_deltas': {p: sum(r['parts'][p]['primitives'] for r in subset) for p in cfg['parts']},
                        'correlations': {p: spearman([r['feature'] for r in subset],
                                                    [r['parts'][p]['primitives'] for r in subset]) for p in cfg['parts']},
                        'later_total_correlation': spearman([r['feature'] for r in subset],
                            [sum(d['primitives'] for p, d in r['parts'].items() if p != 'immediate') for r in subset])}
    output = Path(cfg['output_dir'])
    output.mkdir(parents=True, exist_ok=False)
    result = {'experiment_id': cfg['experiment_id'], 'status': 'complete', 'post_hoc': True,
              'config': cfg, 'events': events, 'groups': groups, 'all_accounting_checks_passed': True,
              'no_model_trained': True, 'no_new_rollout': True, 'source_hashes': {str(p): digest(p) for p in paths},
              'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip(),
              'python': sys.version, 'argv': sys.argv}
    (output / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(groups, indent=2))


if __name__ == '__main__':
    main()
