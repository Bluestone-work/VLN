#!/usr/bin/env python3
"""Post-hoc descriptive strata and geometry for interpreting a frozen route audit."""
import argparse
import collections
import gzip
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from route_alignment import trace_path


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--analysis-dir', type=Path, required=True)
    p.add_argument('--probe-dir', type=Path, required=True)
    p.add_argument('--prior-analysis-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    summary = json.loads((args.analysis_dir / 'summary.json').read_text())
    states = [json.loads(l) for l in (args.analysis_dir / 'states.jsonl').open()]
    original = [json.loads(l) for l in (args.prior_analysis_dir / 'states.jsonl').open()]
    key = lambda r: (r['scene_id'], r['episode_id'], r['high_level_step'])
    prior = {key(r): r for r in original}
    if {key(r) for r in states} != set(prior):
        raise ValueError('Different eligible state populations')
    for s in states:
        if s['joint_cost_gain_m'] != prior[key(s)]['both_costs_capped_gain_m']:
            raise ValueError('Original cost-controlled result changed')
        if not (0 <= s['route_matched_gain_m'] <= s['endpoint_only_gain_m'] <= s['joint_cost_gain_m']):
            raise ValueError('Route/cost gate inclusion order violated')
    manifest = json.loads((args.probe_dir / 'manifest.json').read_text())
    split = manifest['config']['split']
    reference_path = Path('data/datasets/R2R_VLNCE_v1-2_preprocessed') / split / (split + '_gt.json.gz')
    refs = json.load(gzip.open(str(reference_path), 'rt'))
    dataset_path = Path(manifest['config']['dataset_path'])
    episodes = json.load(gzip.open(str(dataset_path), 'rt'))['episodes']
    instructions = {str(e['episode_id']): e['instruction'].get('instruction_text', '') for e in episodes}
    material = [r for r in states if r['route_matched_gain_m'] >= .25]
    result = {'source_experiment': summary['source_experiment'], 'analysis_type': 'post-hoc descriptive only',
              'state_population_and_cost_results_match_prior': True,
              'route_gate_inclusion_order_verified': True, 'states_verified': len(states),
              'route_material_states': len(material),
              'selected_goal_progress_negative': sum(r['selected_progress_m'] < 0 for r in material),
              'alternative_goal_progress_positive': sum(r['selected_progress_m'] + r['route_matched_gain_m'] > 0 for r in material),
              'baseline_episode_successful': sum(r['baseline_episode_success'] > 0 for r in material),
              'selected_reference_endpoint_at_start': sum(r['selected_reference_endpoint'] == 0 for r in material),
              'selected_reference_endpoint_at_end': sum(r['selected_reference_endpoint'] == len(refs[r['episode_id']]['locations']) - 1 for r in material),
              'limits': 'Categories overlap and are selected post-hoc. No causal failure labels or semantic instruction verification. A retained gain can be less-negative goal progress, not actual forward progress.'}
    # Representative cases are chosen by fixed descriptive rules, not for a
    # metric or training label: largest retained gain, largest fully rejected
    # state, and largest retained gain whose alternative still moves away.
    retained = sorted(material, key=lambda r: -r['route_matched_gain_m'])
    rejected = sorted([r for r in states if r['joint_cost_gain_m'] >= .25 and r['route_matched_gain_m'] == 0], key=lambda r: -r['joint_cost_gain_m'])
    away = sorted([r for r in material if r['selected_progress_m'] + r['route_matched_gain_m'] <= 0], key=lambda r: -r['route_matched_gain_m'])
    examples = []
    for label, rows in [('largest_retained', retained), ('largest_fully_rejected', rejected), ('less_negative_not_forward', away)]:
        if rows:
            examples.append((label, rows[0]))
    wanted = {}
    for label, s in examples:
        alternate = s['goal_best_index'] if label == 'largest_fully_rejected' else s['route_best_index']
        wanted[(key(s), s['selected_index'])] = None
        wanted[(key(s), alternate)] = None
    raw_path = args.probe_dir / 'branch_traces.jsonl'
    with raw_path.open() as raw:
        for line in raw:
            branch = json.loads(line)
            identity = (tuple(branch['decision_key']), branch['index'])
            if branch['order'] == 'forward' and identity in wanted:
                if wanted[identity] is not None:
                    raise ValueError('Duplicate example branch')
                wanted[identity] = trace_path(branch['trace'])
    if any(v is None for v in wanted.values()):
        raise ValueError('Missing example branch')
    captions = []
    for label, s in examples:
        alternate = s['goal_best_index'] if label == 'largest_fully_rejected' else s['route_best_index']
        chosen, alternative = wanted[(key(s), s['selected_index'])], wanted[(key(s), alternate)]
        ref = np.asarray(refs[s['episode_id']]['locations'])
        fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.3))
        for ax, dims, names in zip(axes, [(0, 2), (0, 1)], [('x', 'z'), ('x', 'height y')]):
            x, y = dims
            ax.plot(ref[:, x], ref[:, y], color='#999999', linestyle='--', label='Ordered reference')
            ax.plot(chosen[:, x], chosen[:, y], color='#bd4138', marker='.', markersize=3, label='Selected option')
            ax.plot(alternative[:, x], alternative[:, y], color='#267fa3', marker='.', markersize=3, label='Alternative option')
            ax.scatter(chosen[0, x], chosen[0, y], c='black', marker='x', s=55, label='Shared state')
            ax.scatter(ref[-1, x], ref[-1, y], c='#777777', marker='*', s=90, label='Reference end')
            ax.set_xlabel(names[0] + ' (m)')
            ax.set_ylabel(names[1] + ' (m)')
            ax.set_aspect('equal', adjustable='datalim')
            ax.grid(alpha=.2)
        axes[0].legend(fontsize=7)
        gain = s['joint_cost_gain_m'] if label == 'largest_fully_rejected' else s['route_matched_gain_m']
        fig.suptitle('Episode {} / decision {}: {}'.format(s['episode_id'], s['high_level_step'], label.replace('_', ' ')), fontsize=11)
        fig.text(.5, .012, 'Goal progress: selected {:.3f} m; alternative {:.3f} m. Offline geometry, no continuation.'.format(s['selected_progress_m'], s['selected_progress_m'] + gain), ha='center', fontsize=9)
        fig.tight_layout(rect=(0, .06, 1, .94))
        for extension in ['png', 'pdf']:
            fig.savefig(str(args.output_dir / (label + '.' + extension)), dpi=160)
        plt.close(fig)
        captions.append({'selection_rule': label, 'state': s, 'instruction': instructions[s['episode_id']],
                         'alternate_graph_index': alternate, 'plot': label + '.png',
                         'note': 'Horizontal and elevation projections only; walls are not rendered. No semantic correctness claim.'})
    result['examples'] = captions
    files = [Path(__file__), args.analysis_dir / 'summary.json', args.analysis_dir / 'states.jsonl',
             args.prior_analysis_dir / 'states.jsonl', reference_path, dataset_path]
    result['hashes'] = {str(path): digest(path) for path in files}
    result['raw_trace_sha256'] = digest(raw_path)
    (args.output_dir / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ['examples', 'hashes']}, indent=2))


if __name__ == '__main__':
    main()
