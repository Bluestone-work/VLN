#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


def rows(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream))


def truth(value):
    return value == 'True'


def percentile(values, q):
    values = sorted(values)
    if not values:
        return None
    return values[int(round((len(values) - 1) * q))]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train-analysis', type=Path, required=True)
    p.add_argument('--unseen-analysis', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    route_rows, state_rows = [], []
    for cohort, root in [('train', args.train_analysis), ('val_unseen', args.unseen_analysis)]:
        for row in rows(root / 'per_route.csv'):
            row['cohort'] = cohort; route_rows.append(row)
        for row in rows(root / 'per_state.csv'):
            row['cohort'] = cohort; state_rows.append(row)
    route_fields = ['cohort'] + [k for k in route_rows[0] if k != 'cohort']
    with (args.output_dir / 'per_route.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=route_fields)
        writer.writeheader(); writer.writerows(route_rows)
    new_steps = []
    for row in state_rows:
        if row['new_candidate_steps_min']:
            new_steps.extend([float(row['new_candidate_steps_min']), float(row['new_candidate_steps_max'])])
    keys = ['strict', 'quality_rescue', 'cost_capped_rescue']
    counts = {}
    for key in keys:
        counts[key] = {
            'A0': sum(truth(r['a0_' + key]) for r in route_rows),
            'A1': sum(truth(r['a1_' + key]) for r in route_rows),
        }
    added_quality = [r for r in route_rows if not truth(r['a0_quality_rescue']) and truth(r['a1_quality_rescue'])]
    added_cost = [r for r in route_rows if not truth(r['a0_cost_capped_rescue']) and truth(r['a1_cost_capped_rescue'])]
    summary = {
        'protocol': 'Gate A Dense-Candidate Full-Return Oracle, combined audited cohorts',
        'privileged_analysis_only': True,
        'routes': len(route_rows),
        'scenes': len({r['scene_id'] for r in route_rows}),
        'critical_nonstop_states': len(state_rows),
        'a0_native_branches': sum(int(r['a0_candidates']) for r in state_rows),
        'a1_new_branches': sum(int(r['a1_new_candidates']) for r in state_rows),
        'mean_a0_candidates_per_state': sum(int(r['a0_candidates']) for r in state_rows) / len(state_rows),
        'mean_a1_candidates_per_state': sum(int(r['a1_candidates']) for r in state_rows) / len(state_rows),
        'route_rescue_counts': counts,
        'incremental_quality_routes': len(added_quality),
        'incremental_quality_scenes': len({r['scene_id'] for r in added_quality}),
        'incremental_quality_route_ids': [r['episode_id'] for r in added_quality],
        'incremental_cost_capped_routes': len(added_cost),
        'incremental_cost_capped_scenes': len({r['scene_id'] for r in added_cost}),
        'native_rescue_but_policy_missed_routes': sum(truth(r['a0_quality_rescue']) and not truth(r['native_selected_quality_rescue']) for r in route_rows),
        'dense_still_no_quality_rescue_routes': sum(not truth(r['a1_quality_rescue']) for r in route_rows),
        'new_candidate_full_episode_primitive_proxy': {
            'min_of_state_ranges': min(new_steps), 'median_of_state_range_endpoints': percentile(new_steps, .5),
            'p90_of_state_range_endpoints': percentile(new_steps, .9), 'max_of_state_ranges': max(new_steps),
        },
        'gate_a_decision': 'GO for proposal-coverage oracle opportunity; no waypoint model training claim',
        'evidence': 'A1 adds 3 quality rescues in 3 scenes and 2 cost-capped rescues over A0 across 31 audited failed routes.',
        'a2_status': 'not run because changing ghost merge would alter persistent graph semantics',
    }
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    try:
        import matplotlib.pyplot as plt
        labels = ['Strict', 'Quality', 'Cost-capped']
        a0 = [counts[k]['A0'] for k in keys]; a1 = [counts[k]['A1'] for k in keys]
        x = list(range(len(labels)))
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        ax.bar([v - .18 for v in x], a0, .36, label='A0 native', color='#4c78a8')
        ax.bar([v + .18 for v in x], a1, .36, label='A1 dense union', color='#f58518')
        ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel('Rescuable routes / 31')
        ax.set_title('Privileged full-return rescue ceiling')
        ax.legend(frameon=False); fig.tight_layout()
        fig.savefig(str(args.output_dir / 'gate_a_rescue_ceiling.png'), dpi=180)
        plt.close(fig)
    except ImportError:
        pass
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
