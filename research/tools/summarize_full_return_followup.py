#!/usr/bin/env python3
"""Descriptive scene uncertainty and terminal symptoms; no fitted selector."""
import argparse
import json
from pathlib import Path
import numpy as np
from audit_single_intervention import read_rows, group_rows, one
from run_option_capture import digest


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--tag', required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    source = args.root / 'full_return_analysis_001/summary.json'
    data = json.loads(source.read_text())
    cfg = json.loads((args.root / 'capture_001/manifest.json').read_text())['config']
    seeds = cfg['analysis']
    metrics = ['success', 'spl', 'ndtw', 'sdtw', 'distance_to_goal',
               'path_length', 'primitive_action_count', 'high_level_steps', 'collision_events']
    intervals = {}
    for kind in ['top_logit', 'seeded_graph_id']:
        events = [e for e in data['events'] if e['kind'] == kind]
        scenes = sorted(set(e['scene_id'] for e in events))
        scene_means = np.asarray([[np.mean([e['delta'][m] for e in events if e['scene_id'] == s])
                                  for m in metrics] for s in scenes])
        rng = np.random.RandomState(seeds['bootstrap_seed'])
        indices = rng.randint(0, len(scenes), size=(seeds['bootstrap_samples'], len(scenes)))
        samples = scene_means[indices].mean(axis=1)
        intervals[kind] = {m: {'mean_delta': float(scene_means[:, i].mean()),
                              'scene_percentile_95': np.percentile(samples[:, i], [2.5, 97.5]).tolist()}
                           for i, m in enumerate(metrics)}
    # Endpoint-only proximity is an observable symptom, never a causal oracle STOP gain.
    traces, terminal, hashes = {}, {}, {str(source): digest(source), str(Path(__file__)): digest(Path(__file__))}
    for kind, directory in [('baseline', 'capture_001'),
                            ('top_logit', 'top_logit_enabled_' + args.tag),
                            ('seeded_graph_id', 'seeded_graph_id_enabled_' + args.tag)]:
        path = one(args.root / directory / 'traces', '*.jsonl')
        traces[kind] = group_rows(read_rows(path))
        hashes[str(path)] = digest(path)
        terminal[kind] = []
        native = {e['episode_id']: e['baseline' if kind == 'baseline' else 'alternative']
                  for e in data['events'] if e['kind'] == ('top_logit' if kind == 'baseline' else kind)}
        for ep, rr in sorted(traces[kind].items()):
            distances = [r['distance_after'] for r in rr]
            inside = [i for i, d in enumerate(distances) if d < 3.0]
            terminal[kind].append({'episode_id': ep,
                'success': native[ep]['success'],
                'endpoint_neighborhood_visited': bool(inside),
                'first_endpoint_inside_step': inside[0] if inside else None,
                'minimum_endpoint_distance_m': min(distances),
                'final_goal_distance_m': distances[-1],
                'left_neighborhood_and_failed': bool(inside) and not bool(native[ep]['success']),
                'stop_primitives': len(rr[-1]['primitives'])})
    result = {'experiment_id': cfg['experiment_id'] + '-DESCRIPTIVE',
              'scene_bootstrap': intervals, 'bootstrap_seed': seeds['bootstrap_seed'],
              'bootstrap_samples': seeds['bootstrap_samples'], 'terminal_symptoms': terminal,
              'limits': 'Post-hoc descriptive analysis, eight scene clusters, one simulator seed. Endpoint visits miss between-endpoint proximity. No causal STOP gain, no threshold tuning or new policy.',
              'hashes': hashes}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for kind, label in [('baseline', 'Baseline'), ('top_logit', 'Top-logit alternative'),
                        ('seeded_graph_id', 'Seeded alternative')]:
        rr = traces[kind]['2508']
        xx = np.arange(len(rr) + 1)
        yy = [rr[0]['distance_before']] + [r['distance_after'] for r in rr]
        axes[0].plot(xx, yy, marker='o', label=label)
    axes[0].axhline(3, linestyle='--', color='gray', label='Success distance')
    axes[0].axvline(3, linestyle=':', color='black')
    axes[0].set(xlabel='Completed high-level decisions', ylabel='Goal distance (m)',
                title='Episode 2508: delayed rescue')
    axes[0].legend(fontsize=8)
    for kind, marker in [('top_logit', 'o'), ('seeded_graph_id', 'x')]:
        ee = [e for e in data['events'] if e['kind'] == kind]
        axes[1].scatter([e['delta']['primitive_action_count'] for e in ee],
                        [100 * e['delta']['ndtw'] for e in ee], marker=marker, label=kind)
    axes[1].axhline(0, color='gray', linewidth=.7)
    axes[1].axvline(0, color='gray', linewidth=.7)
    axes[1].set(xlabel='Additional primitives (alternative - baseline)', ylabel='nDTW change (pp)',
                title='All 32 interventions, including harms')
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(str(args.output_dir / 'full_return_diagnostics.png'), dpi=180)
    fig.savefig(str(args.output_dir / 'full_return_diagnostics.pdf'))
    print(json.dumps({'intervals': intervals,
                      'visited_then_failed': {k: [e['episode_id'] for e in v if e['left_neighborhood_and_failed']]
                                              for k, v in terminal.items()}}, indent=2))


if __name__ == '__main__':
    main()
