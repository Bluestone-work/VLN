"""Opt-in action tracing and isolated replay support; no policy changes.

Registered only when explicitly imported by the research runner. The original
VLNCEDaggerEnv.step, turn and single/multi_step_control implementations are used.
"""
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
from habitat_baselines.common.baseline_registry import baseline_registry
from vlnce_baselines.common.environments import VLNCEDaggerEnv


def pack(value):
    """JSON-safe serialization retaining array/scalar precision and tuples."""
    if isinstance(value, np.ndarray):
        return {'__array__': value.tolist(), 'dtype': value.dtype.str}
    if isinstance(value, np.generic):
        return {'__scalar__': value.item(), 'dtype': value.dtype.str}
    if isinstance(value, tuple):
        return {'__tuple__': [pack(v) for v in value]}
    if isinstance(value, list):
        return [pack(v) for v in value]
    if isinstance(value, dict):
        return {str(k): pack(v) for k, v in value.items()}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError('Unsupported trace value {}'.format(type(value)))


def unpack(value):
    if isinstance(value, list):
        return [unpack(v) for v in value]
    if isinstance(value, dict):
        if '__array__' in value:
            return np.asarray(value['__array__'], dtype=np.dtype(value['dtype']))
        if '__scalar__' in value:
            return np.dtype(value['dtype']).type(value['__scalar__'])
        if '__tuple__' in value:
            return tuple(unpack(v) for v in value['__tuple__'])
        return {k: unpack(v) for k, v in value.items()}
    return value


def rng_snapshot():
    return pack({'python': random.getstate(), 'numpy': np.random.get_state()})


def restore_rng(state):
    state = unpack(state)
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])


def state_snapshot(sim):
    state = sim.get_agent_state()
    return {'position': state.position.tolist(),
            'rotation': list(state.rotation.imag) + [float(state.rotation.real)]}


def numeric_metrics(env):
    metrics = env.get_metrics()
    result = {}
    for key in ['steps_taken', 'collisions']:
        if key in metrics:
            result[key] = pack(metrics[key])
    return result


def observation_hashes(observations):
    return {key: hashlib.sha256(value.tobytes()).hexdigest()
            for key, value in observations.items() if isinstance(value, np.ndarray)}


@baseline_registry.register_env(name='VLNCEOptionTraceEnv')
class OptionTraceEnv(VLNCEDaggerEnv):
    def __init__(self, config, dataset=None, trace_dir=None):
        super().__init__(config, dataset)
        if self.video_option:
            raise ValueError('Calibration requires unchanged non-video evaluation')
        self.trace_path = None
        self.last_option_trace = None
        self._primitive_trace = None
        self._option_step = 0
        directory = os.environ.get('ETPNAV_OPTION_TRACE_DIR') if trace_dir is None else trace_dir
        if directory:
            directory = Path(directory)
            directory.mkdir(parents=True, exist_ok=True)
            prefix = 'worker_seed{}'.format(config.TASK_CONFIG.SEED)
            with (directory / (prefix + '.yaml')).open('x') as stream:
                stream.write(config.dump())
            self.trace_path = directory / (prefix + '.jsonl')
            with self.trace_path.open('x'):
                pass

    def reset(self):
        result = super().reset()
        self._option_step = 0
        return result

    def wrap_act(self, act, vis_info):
        result = super().wrap_act(act, vis_info)
        if self._primitive_trace is not None:
            self._primitive_trace.append({'action': int(act),
                                          'collided': bool(self._env.sim.previous_step_collided),
                                          'pose': state_snapshot(self._env.sim)})
        return result

    def step(self, action, vis_info, *args, **kwargs):
        # Capture typed action values before any execution or serialization.
        record = {'schema_version': 1, 'privileged_analysis_only': True,
                  'episode_id': str(self._env.current_episode.episode_id),
                  'scene_id': str(self._env.current_episode.scene_id),
                  'high_level_step': self._option_step,
                  'action': pack(action),
                  'pre_pose': state_snapshot(self._env.sim),
                  'pre_rng': rng_snapshot(),
                  'pre_metrics': numeric_metrics(self),
                  'distance_before': float(self.current_dist_to_goal())}
        self._primitive_trace = []
        try:
            result = super().step(action, vis_info, *args, **kwargs)
            if action['act'] == 0:
                # STOP uses Habitat Env.step directly, not wrap_act. No motion
                # collision is attributed to its cached collided flag.
                self._primitive_trace.append({'action': 0, 'collided': None,
                                              'pose': state_snapshot(self._env.sim)})
            record.update({'post_pose': state_snapshot(self._env.sim),
                           'post_rng': rng_snapshot(),
                           'post_metrics': numeric_metrics(self),
                           'distance_after': float(self.current_dist_to_goal()),
                           'done': bool(result[2]), 'primitives': self._primitive_trace,
                           'observation_hashes': observation_hashes(result[0])})
            self.last_option_trace = record
            if self.trace_path:
                with self.trace_path.open('a') as stream:
                    stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + '\n')
            self._option_step += 1
            return result
        finally:
            self._primitive_trace = None
