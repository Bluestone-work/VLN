#!/usr/bin/env python3
"""Enumerate native graph actions and save action-conditioned endpoint RGB-D."""
import argparse, collections, hashlib, itertools, json, os, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT)); os.chdir(str(ROOT))
import numpy as np
import yaml, habitat, habitat_extensions, vlnce_baselines
from vlnce_baselines.adaptive_action.option_calibration import OptionTraceEnv, unpack, restore_rng
from replay_option_calibration import compare


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--capture-dir', type=Path, required=True); ap.add_argument('--output-dir', type=Path, required=True); ap.add_argument('--max-episodes', type=int, default=2)
    a = ap.parse_args(); a.output_dir.mkdir(parents=True, exist_ok=False); endpoint = a.output_dir / 'endpoint_observations'; endpoint.mkdir()
    graph_rows = [json.loads(x) for x in (a.capture_dir/'graph_options/graph_options.jsonl').open() if x.strip()]
    trace_rows = [json.loads(x) for x in (a.capture_dir/'traces/worker_seed100.jsonl').open() if x.strip()]
    traces = {(str(x['episode_id']), int(x['high_level_step'])): x for x in trace_rows}
    graphs = {(str(x['episode_id']), int(x['high_level_step'])): x for x in graph_rows}
    episodes_order = list(collections.OrderedDict((k[0], None) for k in sorted(graphs)))[:a.max_episodes]
    source = next((a.capture_dir/'traces').glob('worker_seed*.yaml'))
    config = habitat.Config(yaml.safe_load(source.read_text())); config.freeze()
    dataset = habitat.make_dataset(config.TASK_CONFIG.DATASET.TYPE, config=config.TASK_CONFIG.DATASET)
    episode_map = {str(e.episode_id): e for e in dataset.episodes}; dataset.episodes = [episode_map[x] for x in episodes_order]
    env = OptionTraceEnv(config, dataset=dataset, trace_dir=False); rows=[]; sentinel=[]
    try:
        for episode_id in episodes_order:
            for step in sorted(k[1] for k in graphs if k[0] == episode_id):
                key=(episode_id, step); graph=graphs[key]; expected=traces[key]
                options=[o for o in graph.get('options',[]) if o.get('admissible')]
                for option in options:
                    env.habitat_env.episodes=[episode_map[episode_id]]; env.habitat_env.episode_iterator=itertools.cycle([episode_map[episode_id]]); env.reset()
                    pose=expected['pre_pose']; q=pose['rotation']; env.habitat_env.sim.set_agent_state(np.asarray(pose['position'],dtype=np.float32), np.quaternion(q[3],q[0],q[1],q[2])); env.habitat_env.task.measurements.reset_measures(episode=episode_map[episode_id], task=env.habitat_env.task); restore_rng(expected['pre_rng'])
                    env.step(unpack(option['action']), vis_info=None); actual=env.last_option_trace
                    is_selected=(option['index']==graph['effective_index']); check=None
                    if is_selected:
                        check=compare(expected, actual, {'endpoint_tolerance_m':1e-5,'rotation_tolerance_rad':1e-5,'progress_tolerance_m':1e-5})
                        sentinel.append(check)
                    obs=env.habitat_env.sim.get_sensor_observations(); files={}; stem='{}_step{:03d}_index{}'.format(episode_id,step,int(option['index']))
                    for sensor in ('rgb','depth'):
                        if sensor in obs:
                            path=endpoint/'{}_{}.npy'.format(stem,sensor); np.save(str(path),np.asarray(obs[sensor]),allow_pickle=False); files[sensor]=str(path)
                    rows.append({'episode_id':episode_id,'high_level_step':step,'index':option['index'],'graph_id':option.get('graph_id'),'action_type':option.get('action',{}).get('act'),'logit':option.get('logit'),'selected_native':is_selected,'endpoint_files':files,'replay_trace':actual,'sentinel':check})
    finally: env.close()
    with (a.output_dir/'native_endpoint_options.jsonl').open('w') as f:
        for r in rows: f.write(json.dumps(r,sort_keys=True,allow_nan=False)+'\n')
    summary={'episodes':len(episodes_order),'decisions':len(set((r['episode_id'],r['high_level_step']) for r in rows)),'action_branches':len(rows),'sentinel_count':len(sentinel),'sentinel_passed':sum(bool(x.get('passed')) for x in sentinel),'status':'PASS' if sentinel and all(x.get('passed') for x in sentinel) else 'FAIL'}
    (a.output_dir/'summary.json').write_text(json.dumps(summary,indent=2)+'\n'); print(json.dumps(summary,indent=2)); raise SystemExit(0 if summary['status']=='PASS' else 1)
if __name__=='__main__': main()
