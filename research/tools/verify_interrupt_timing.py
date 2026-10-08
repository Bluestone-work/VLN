#!/usr/bin/env python3
"""Independent schedule, physical-prefix, sensing and full-return verification."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
from fastdtw import fastdtw


def read(p):
    return json.loads(Path(p).read_text())


def main():
    p = argparse.ArgumentParser(); p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); a = p.parse_args()
    m = read(a.run/'manifest.json'); plan = m['plan']; source = Path(plan['config']['source_root'])
    assert read(a.run/'summary.json')['all_validity_checks_passed']
    for name, h in m['hashes'].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == h, name
    baseline = {}
    for line in (source/'capture_001/traces/worker_seed100.jsonl').open():
        r = json.loads(line)
        baseline[(str(r['episode_id']), r['high_level_step'])] = r
    original_data = json.load(gzip.open(m['source']['config']['dataset_path'], 'rt'))
    subset = json.load(gzip.open(m['overrides']['TASK_CONFIG.DATASET.DATA_PATH'], 'rt'))
    originals = {str(e['episode_id']): e for e in original_data['episodes']}
    assert {str(e['episode_id']) for e in subset['episodes']} == set(plan['replay_episodes'])
    assert all(e == originals[str(e['episode_id'])] for e in subset['episodes'])
    assert {k: v for k, v in original_data.items() if k != 'episodes'} == {k: v for k, v in subset.items() if k != 'episodes'}
    # The baseline nDTW measure uses the released train_gt reference paths.
    gt_path = Path('data/datasets/R2R_VLNCE_v1-2_preprocessed/train/train_gt.json.gz')
    refs = json.load(gzip.open(str(gt_path), 'rt'))
    outcomes = [json.loads(l) for l in (a.run/'results.jsonl').open()]
    modes = ['sense_only', 'interrupt_consume', 'interrupt_retain']
    events = read(a.run/'interrupt_plan.json')['events']
    expected = {'control_'+ep for ep in plan['replay_episodes']}
    expected.update('{}_{}_s{}_p{}'.format(mode, e['episode_id'], e['high_level_step'], e['cut_primitive']) for e in events for mode in modes)
    assert len(outcomes) == len(expected) and {r['case']['case_id'] for r in outcomes} == expected
    native_metrics = read(source/'noninterference_001/capture_episodes.json')
    checks = {'native_or_sensing_option_checks': 0, 'interrupted_prefix_checks': 0,
              'cut_sensor_checks': 0, 'return_components': 0, 'physical_continuity_checks': 0,
              'graph_retain_records': 0, 'graph_consume_records': 0}
    sensing = {}
    for result in outcomes:
        c = result['case']; ep = c['episode_id']; step = c.get('high_level_step')
        records = [json.loads(l) for l in (a.run/'cases'/c['case_id']/'trace.jsonl').open()]
        assert len(records) <= 15 and records[-1]['done'] and records[-1]['action']['act'] == 0
        points = []
        for i, r in enumerate(records):
            assert str(r['episode_id']) == ep and r['high_level_step'] == i
            old = baseline.get((ep, i))
            if c['mode'] in ['control', 'sense_only'] or i < step:
                assert old is not None
                for k in ['action', 'primitives', 'pre_pose', 'post_pose', 'pre_rng', 'post_rng', 'observation_hashes', 'distance_before', 'distance_after', 'done']:
                    assert r[k] == old[k], (c['case_id'], i, k)
                checks['native_or_sensing_option_checks'] += 1
            pp = [r['pre_pose']['position']]+[v['pose']['position'] for v in r['primitives']]
            assert pp[-1] == r['post_pose']['position']
            if points:
                assert points[-1] == pp[0]
            for point in pp:
                if not points or point != points[-1]:
                    points.append(point)
            checks['physical_continuity_checks'] += 1
        if 'cut_primitive' in c:
            cut = c['cut_primitive']; r = records[step]; old = baseline[(ep, step)]
            assert 0 < cut < len(old['primitives']) and r['phases'][cut-1] == 'ghost'
            key = (ep, step, cut)
            if c['mode'] == 'sense_only':
                sensing[key] = r['cut_sensor_hashes']
            else:
                assert r['interrupted'] and r['primitives'] == old['primitives'][:cut]
                assert r['pre_rng'] == old['pre_rng'] and r['post_rng'] == old['post_rng']
                assert r['action'] == old['action'] and r['pre_metrics'] == old['pre_metrics']
                assert r['post_metrics']['steps_taken']-r['pre_metrics']['steps_taken'] == cut
                assert r['cut_sensor_hashes'] == sensing[key]
                before = (r['pre_metrics'].get('collisions') or {}).get('count', 0)
                after = (r['post_metrics'].get('collisions') or {}).get('count', 0)
                assert after-before == sum(v['collided'] is True for v in r['primitives'])
                checks['interrupted_prefix_checks'] += 1; checks['cut_sensor_checks'] += 1
                graph = [json.loads(l) for l in (a.run/'cases'/c['case_id']/'graph/graph_options.jsonl').open()][step]
                assert graph['pending_ghost_restored'] == (c['mode'] == 'interrupt_retain')
                checks['graph_retain_records' if c['mode'] == 'interrupt_retain' else 'graph_consume_records'] += 1
        path = np.asarray(points, dtype=np.float32)
        reference = np.asarray(refs[ep]['locations'], dtype=np.float64)
        length = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
        error = records[-1]['distance_after']; success = float(error <= 3.0)
        ndtw = float(np.exp(-fastdtw(path, reference, dist=lambda x, y: np.linalg.norm(y-x))[0]/(3*len(reference))))
        shortest = records[0]['distance_before']
        recomputed = dict(distance_to_goal=error, success=success, spl=success*shortest/max(shortest, length),
                          ndtw=ndtw, sdtw=success*ndtw, path_length=length,
                          steps_taken=sum(len(r['primitives']) for r in records), high_level_steps=len(records))
        assert result['baseline_metrics'] == native_metrics[ep]
        for k, value in recomputed.items():
            assert abs(value-result['metrics'][k]) <= 1e-6, (c['case_id'], k, value, result['metrics'][k])
            checks['return_components'] += 1
        assert result['policy_calls'] == len(records)
        assert result['extra_sensing_calls'] == sum(r['sensed_at_cut'] for r in records)
        assert result['observation_at_calls'] == sum(r['observation_at_calls'] for r in records)
    a.output.mkdir(parents=True, exist_ok=False)
    report = {'all_checks_passed': True, 'rollouts': len(outcomes), 'events': len(events), 'checks': checks,
              'subset_identity_passed': True, 'reference_path': str(gt_path),
              'reference_sha256': hashlib.sha256(gt_path.read_bytes()).hexdigest(),
              'independence': 'Separately written prefix checks and metric reconstruction from primitive poses; same prescribed fastdtw metric. Goal geodesic distance is the recorded simulator measurement, not an independent navmesh computation.'}
    (a.output/'summary.json').write_text(json.dumps(report, indent=2)+'\n'); print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
