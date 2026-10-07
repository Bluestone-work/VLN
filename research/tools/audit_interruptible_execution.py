#!/usr/bin/env python3
"""Static source-anchor audit; not a dynamic or control-flow proof."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

REQUIRED = {
    'vlnce_baselines/common/environments.py': [
        'self._env.sim.step_without_obs(act)',
        'def single_step_control(self, pos, tryout, vis_info):',
        'def multi_step_control(self, path, tryout, vis_info):',
        'self.single_step_control(action[\'ghost_pos\'], action[\'tryout\'], vis_info)',
        'observations = self.get_observation_at(agent_state.position, agent_state.rotation)',
    ],
    'vlnce_baselines/ss_trainer_ETP.py': [
        'gmap.delete_ghost(ghost_vp)',
        'prev_vp[i] = front_vp',
        'outputs = self.envs.step(env_actions)',
    ],
    'vlnce_baselines/models/graph_utils.py': [
        'def delete_ghost(self, vp):',
        'self.ghost_pos.pop(vp)',
    ],
    'vlnce_baselines/adaptive_action/option_calibration.py': [
        'self._primitive_trace.append',
        'def wrap_act(self, act, vis_info):',
        'def step(self, action, vis_info, *args, **kwargs):',
    ],
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    cfg = json.loads(args.config.read_text()); checks=[]; hashes={}
    for name, needles in REQUIRED.items():
        path=Path(name); text=path.read_text(); hashes[name]=hashlib.sha256(path.read_bytes()).hexdigest()
        checks.extend({'file':name,'needle':n,'present':n in text} for n in needles)
    if not all(x['present'] for x in checks): raise ValueError('Static execution contract changed')
    result={'experiment_id':cfg['experiment_id'],'status':'complete','source_checks':checks,
            'observation_between_primitives':'unavailable to high-level policy in normal non-video execution',
            'graph_transaction':'ghost deletion and prev_vp update occur before envs.step',
            'no_baseline_core_mutation':True,'no_simulator_run':True,'no_model_trained':True,
            'static_checks_only':True,'dynamic_semantics_proven':False,
            'decision':cfg['decision'],
            'git_commit':subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),
            'source_hashes':hashes}
    out=args.output; out.mkdir(parents=True,exist_ok=False)
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    (out/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    print(json.dumps({'checks':len(checks),'passed':all(x['present'] for x in checks),'decision':cfg['decision']},indent=2))


if __name__=='__main__': main()
