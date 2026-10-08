"""Research-only full-return enumeration and prefix-preserving interrupt oracle.

Imported by an explicit research runner only. Original trainer/env stay intact.
"""
import copy
import itertools
import json
import os
from pathlib import Path
import numpy as np
from habitat_baselines.common.baseline_registry import baseline_registry
from vlnce_baselines.ss_trainer_ETP import RLTrainer
from vlnce_baselines.adaptive_action.option_calibration import (
    OptionTraceEnv, pack, restore_rng, observation_hashes)
from vlnce_baselines.adaptive_action.graph_option_capture import GraphOptionCapture


class CutOption(Exception):
    pass


@baseline_registry.register_env(name='VLNCEInterruptConfirmationEnv')
class CriticalEnv(OptionTraceEnv):
    def __init__(self, config, dataset=None):
        super().__init__(config, dataset, trace_dir=False)
        self.case = None
        self.all_episodes = {str(e.episode_id): e for e in self._env.episodes}
        self.renders = 0
        self.phase = None

    def configure_case(self, case, output, initial_rng):
        self.case = case
        ep = self.all_episodes[str(case['episode_id'])]
        self._env.episode_iterator = itertools.cycle([ep])
        self.initial_rng = initial_rng
        self.case_output = Path(output)
        self.case_output.mkdir(parents=True, exist_ok=False)
        self.case_trace = self.case_output/'trace.jsonl'
        self.case_trace.touch(exist_ok=False)
        self.renders = 0

    def reset(self):
        obs = super().reset()
        if self.case is not None:
            restore_rng(self.initial_rng)
        return obs

    def get_observation_at(self, *args, **kwargs):
        self.renders += 1
        return super().get_observation_at(*args, **kwargs)

    def multi_step_control(self, path, tryout, vis_info):
        self.phase = 'backtrack'
        try:
            return super().multi_step_control(path, tryout, vis_info)
        finally:
            self.phase = 'ghost'

    def single_step_control(self, pos, tryout, vis_info):
        try:
            return super().single_step_control(pos, tryout, vis_info)
        except CutOption:
            if self.phase != 'ghost':
                raise ValueError('Backtracking interruption is outside this protocol')
            self.did_cut = True

    def wrap_act(self, act, vis_info):
        obs = super().wrap_act(act, vis_info)
        self.phases.append(self.phase)
        cut = self.case.get('cut_primitive') if self.case else None
        if (cut and self._option_step == self.case['high_level_step']
                and len(self._primitive_trace) == cut):
            if self.phase != 'ghost':
                raise ValueError('Cut must be inside final ghost segment')
            # Both sensing-only and replan arms render identical sensors here.
            state = self._env.sim.get_agent_state()
            sensed = self.get_observation_at(state.position, state.rotation)
            self.cut_sensor_hashes = observation_hashes(sensed)
            self.sensed_at_cut = True
            if self.case['mode'].startswith('interrupt'):
                raise CutOption()
        return obs

    def step(self, action, vis_info, *args, **kwargs):
        self.phases=[];self.did_cut=False;self.sensed_at_cut=False;self.cut_sensor_hashes=None
        self.phase = 'ghost' if action['act']==4 else 'stop'
        before = self.renders
        result = super().step(action, vis_info, *args, **kwargs)
        record = self.last_option_trace
        record.update(phases=self.phases, interrupted=self.did_cut,
                      sensed_at_cut=self.sensed_at_cut, cut_sensor_hashes=self.cut_sensor_hashes,
                      observation_at_calls=self.renders-before)
        with self.case_trace.open('a') as f:
            f.write(json.dumps(record, allow_nan=False)+'\n')
        return result


class CriticalHook(GraphOptionCapture):
    def __init__(self, case, baseline, output):
        super().__init__(output)
        self.case=case;self.baseline=baseline;self.applied=False;self.encountered=False
        self.saved_ghost=None

    def prepare(self, trainer, step, current, positions, nav_inputs, logits,
                chosen, policy_chosen, no_vp, embeddings):
        # Use the audited native option builder without copying the trainer loop.
        super().prepare(trainer,step,current,positions,nav_inputs,logits,chosen,policy_chosen,no_vp,None)
        if len(self.pending)!=1 or trainer.max_len!=15:
            raise ValueError('One worker and native horizon required')
        row=self.pending[0];ep=str(row['episode_id'])
        if ep!=str(self.case['episode_id']):raise ValueError('Wrong replay episode')
        if not self.applied:
            old=self.baseline[(ep,step)]
            for name in ['graph_ids','mask','visited','logits','policy_index','effective_index',
                         'graph_position','current_vp','budget_stop','no_vp_left']:
                if old[name]!=row[name]:raise ValueError('Prefix graph mismatch {} at {}:{}'.format(name,ep,step))
            for a,b in zip(old['options'],row['options']):
                if a['action']!=b['action']:raise ValueError('Full action identity mismatch')
        target = self.case['mode']!='control' and step==self.case['high_level_step']
        row['oracle_intervention']=False
        row['pending_ghost_restored']=False
        if target:
            self.encountered=True
            if self.case['mode']=='action':
                idx=self.case['action_index']
                option=next(o for o in row['options'] if o['index']==idx)
                if not option['admissible']:raise ValueError('Horizon/mask violation')
                if option['action']!=self.case['action']:raise ValueError('Planned action drift')
                if idx!=row['effective_index']:
                    chosen[0]=idx;row['effective_index']=idx;self.applied=True
                    row['oracle_intervention']=True
            elif self.case['mode'].startswith('interrupt'):
                if row['effective_index']==0:raise ValueError('Cannot interrupt STOP')
                self.applied=True;row['oracle_intervention']=True
                if self.case['mode']=='interrupt_retain':
                    gmap=trainer.gmaps[0];vp=row['graph_ids'][row['effective_index']]
                    names=['ghost_pos','ghost_mean_pos','ghost_embeds','ghost_fronts']
                    if gmap.has_real_pos:names.append('ghost_real_pos')
                    self.saved_ghost=(gmap,vp,{name:copy.deepcopy(getattr(gmap,name)[vp]) for name in names})
                    row['pending_ghost_restored']=True

    def commit(self, env_actions):
        super().commit(env_actions)
        if self.saved_ghost:
            gmap,vp,values=self.saved_ghost
            if vp in gmap.ghost_pos:raise ValueError('Native ghost consumption did not occur')
            for name,value in values.items():getattr(gmap,name)[vp]=value
            self.saved_ghost=None


@baseline_registry.register_trainer(name='SS-ETP-InterruptConfirmation')
class CriticalTrainer(RLTrainer):
    def __init__(self, config=None):
        super().__init__(config)
        if (config.NUM_ENVIRONMENTS!=1 or config.IL.back_algo!='control' or config.VIDEO_OPTION
                or config.RL_TOPO.ENABLED or self.action_abstraction!='default'
                or self.aaa_oracle_enabled or self.graph_selection_oracle_enabled
                or self.aaa_diagnostics_enabled
                or not config.TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING):
            raise ValueError('Requires calibrated single-worker native evaluation')
        self.manifest=json.loads(Path(os.environ['ETPNAV_CRITICAL_MANIFEST']).read_text())

    def rollout(self, mode, **kwargs):
        if mode!='eval':raise ValueError('Oracle is evaluation-only')
        # Import runner-local driver after registration; no default training path uses it.
        from critical_case_driver_confirmation import drive_cases
        drive_cases(self, super().rollout)
