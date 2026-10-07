#!/usr/bin/env python3
"""Produce paired tables and best/worst nDTW continuation geometry after gates."""
import argparse
import gzip
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from audit_single_intervention import read_rows, group_rows, one
from route_alignment import trace_path
from run_option_capture import digest


def episode_path(records):
    points = []
    for record in records:
        path = trace_path(record)
        if points and not np.array_equal(points[-1], path[0]):
            raise ValueError('Discontinuous actual continuation')
        points.extend(path if not points else path[1:])
    return np.asarray(points)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--audit-dir', type=Path, required=True)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    summary = json.loads((args.audit_dir / 'summary.json').read_text())
    if not summary['all_required_gates_passed'] or summary['mode'] != 'enabled':
        raise ValueError('Accepted enabled continuation audit required')
    pairs = json.loads((args.audit_dir / 'paired_episodes.json').read_text())
    selected = [r for r in pairs if r['scheduled']]
    manifest = json.loads((args.run_dir / 'manifest.json').read_text())
    original_path = one(Path(manifest['source_capture']) / 'traces', '*.jsonl')
    actual_path = one(args.run_dir / 'traces', '*.jsonl')
    original = group_rows(read_rows(original_path))
    actual = group_rows(read_rows(actual_path))
    refs_path = Path('data/datasets/R2R_VLNCE_v1-2_preprocessed/train/train_gt.json.gz')
    refs = json.load(gzip.open(str(refs_path), 'rt'))
    dataset = json.load(gzip.open(manifest['config']['dataset_path'], 'rt'))
    instructions = {str(e['episode_id']): e['instruction']['instruction_text'] for e in dataset['episodes']}
    events = {e['episode_id']: e for e in manifest['schedule']['events']}
    table = ['| Episode / step | SR before → after | SPL delta (pp) | nDTW delta (pp) | Path delta (m) | Primitive delta |',
             '| --- | --- | ---: | ---: | ---: | ---: |']
    for row in selected:
        a, b, d = row['baseline'], row['intervention'], row['delta']
        table.append('| {} / {} | {} → {} | {:+.3f} | {:+.3f} | {:+.3f} | {:+.0f} |'.format(
            row['episode_id'], events[row['episode_id']]['high_level_step'],
            int(a['success']), int(b['success']), d['spl'] * 100, d['ndtw'] * 100,
            d['path_length'], d['primitive_action_count']))
    (args.output_dir / 'paired_table.md').write_text('\n'.join(table) + '\n')
    examples = []
    for name, choose in [('largest_ndtw_delta', max), ('smallest_ndtw_delta', min)]:
        row = choose(selected, key=lambda r: (r['delta']['ndtw'], r['episode_id']))
        episode = row['episode_id']
        before, after = episode_path(original[episode]), episode_path(actual[episode])
        reference = np.asarray(refs[episode]['locations'])
        event = events[episode]
        position = np.asarray(actual[episode][event['high_level_step']]['pre_pose']['position'])
        fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8))
        for ax, dims, labels in zip(axes, [(0, 2), (0, 1)], [('x', 'z'), ('x', 'height y')]):
            x, y = dims
            ax.plot(reference[:, x], reference[:, y], '--', color='#999999', label='Ordered reference')
            ax.plot(before[:, x], before[:, y], color='#355c8b', alpha=.85, label='Original continuation')
            ax.plot(after[:, x], after[:, y], color='#d67125', alpha=.85, label='Single intervention continuation')
            ax.scatter(position[x], position[y], marker='x', color='black', s=65, label='Intervention state')
            ax.scatter(before[-1, x], before[-1, y], marker='s', facecolors='none', edgecolors='#355c8b', s=60)
            ax.scatter(after[-1, x], after[-1, y], marker='s', facecolors='none', edgecolors='#d67125', s=60)
            ax.scatter(reference[-1, x], reference[-1, y], marker='*', color='#555555', s=90, label='Reference end')
            ax.set_aspect('equal', adjustable='datalim')
            ax.set_xlabel(labels[0] + ' (m)')
            ax.set_ylabel(labels[1] + ' (m)')
            ax.grid(alpha=.2)
        axes[0].legend(fontsize=7)
        fig.suptitle('Episode {}: success {} -> {}; nDTW {:+.2f} pp; path {:+.2f} m'.format(
            episode, int(row['baseline']['success']), int(row['intervention']['success']),
            row['delta']['ndtw'] * 100, row['delta']['path_length']), fontsize=11)
        fig.text(.5, .015, 'Privileged single-action diagnostic; actual native continuations. Walls are not rendered.', ha='center', fontsize=9)
        fig.tight_layout(rect=(0, .055, 1, .95))
        for extension in ['png', 'pdf']:
            fig.savefig(str(args.output_dir / (name + '.' + extension)), dpi=160)
        plt.close(fig)
        examples.append({'selection_rule': name, 'paired_metrics': row, 'instruction': instructions[episode],
                         'intervention_step': event['high_level_step'], 'figure': name + '.png'})
    output = {'experiment_id': summary['experiment_id'], 'analysis_type': 'post-hoc descriptive best/worst nDTW examples',
              'examples': examples, 'all_seven_routes_in_table': True,
              'hashes': {str(path): digest(path) for path in [Path(__file__), args.audit_dir / 'summary.json',
                  args.audit_dir / 'paired_episodes.json', original_path, actual_path, refs_path]}}
    (args.output_dir / 'summary.json').write_text(json.dumps(output, indent=2) + '\n')
    print('\n'.join(table))


if __name__ == '__main__':
    main()
