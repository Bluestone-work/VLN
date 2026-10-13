"""Exact single-action full-return hook for frozen native graph branches.

This module is opt-in and is deliberately separate from the existing
single-intervention trainer.  It permits STOP as an alternative so every
admissible native option can receive a full-return outcome.
"""
import json
import math
import os
import hashlib
import random
from pathlib import Path

import numpy as np
import torch
from habitat_baselines.common.baseline_registry import baseline_registry

from vlnce_baselines.ss_trainer_ETP import RLTrainer
from vlnce_baselines.adaptive_action.graph_option_capture import full_action
from vlnce_baselines.adaptive_action.option_calibration import pack


def decision_key(row):
    return row['scene_id'], str(row['episode_id']), row['high_level_step']


def canonical(value):
    """Hash all GraphMap contents, including tensors, without mutating them."""
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return {'shape': list(value.shape), 'dtype': value.dtype.str,
                'sha256': hashlib.sha256(value.tobytes()).hexdigest()}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        # Preserve insertion order: it can affect tie-breaking in native STOP.
        return [[canonical(k), canonical(v)] for k, v in value.items()]
    if isinstance(value, set):
        return sorted([canonical(x) for x in value], key=str)
    if isinstance(value, (tuple, list)):
        return [canonical(x) for x in value]
    if hasattr(value, 'nodes') and hasattr(value, 'edges'):
        return {'nodes': canonical(list(value.nodes(data=True))),
                'edges': canonical(list(value.edges(data=True)))}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError('Unhashed graph field: {}'.format(type(value)))


def state_digest(value):
    return hashlib.sha256(json.dumps(canonical(value), allow_nan=False,
                                     separators=(',', ':')).encode()).hexdigest()


class NativeFullReturnHook:
    def __init__(self, schedule, baseline, directory, enabled):
        self.events = {}
        for event in schedule['events']:
            episode = str(event['episode_id'])
            if episode in self.events:
                raise ValueError('Only one event per episode is allowed per run')
            if event['alternative_index'] == event['baseline_index']:
                raise ValueError('Full-return event must identify a distinct alternative')
            self.events[episode] = event
        self.baseline, self.enabled = baseline, bool(enabled)
        self.encountered, self.applied, self.seen = set(), set(), set()
        self.pending = None
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.log = self.directory / 'decisions.jsonl'
        self.log.touch()

    def prepare(self, trainer, step, current, positions, nav_inputs, logits,
                chosen, policy_chosen, no_vp, embeddings, waypoint_heatmap=None,
                candidate_metadata=None, isolated_native_embeds=None,
                isolated_native_logits=None, full_logits=None):
        if self.pending is not None or not np.array_equal(chosen, policy_chosen):
            raise ValueError('Unexpected upstream intervention')
        pending = []
        for i, episode in enumerate(trainer.envs.current_episodes()):
            episode_id = str(episode.episode_id)
            key = (str(episode.scene_id), episode_id, step)
            if key in self.seen:
                raise ValueError('Repeated decision key')
            ids = nav_inputs['gmap_vp_ids'][i]
            n = len(ids)
            mask = nav_inputs['gmap_masks'][i, :n].detach().cpu().numpy().astype(bool).tolist()
            visited = nav_inputs['gmap_visited_masks'][i, :n].detach().cpu().numpy().astype(bool).tolist()
            scores = logits[i, :n].detach().cpu().numpy()
            native_index = int(policy_chosen[i])
            budget_stop = step >= trainer.max_len - 1
            effective = 0 if native_index == 0 or budget_stop or no_vp[i] else native_index
            if not ids or ids[0] is not None or len(set(ids)) != n:
                raise ValueError('Invalid graph IDs')
            event = self.events.get(episode_id)
            # The native source capture is authoritative through the frozen
            # decision being intervened on.  An enabled alternative can lead
            # to a different graph after that endpoint, so post intervention
            # states must be recorded without pretending they are source
            # states.  Disabled runs remain fully baseline-checkable.
            post_intervention = bool(event and self.enabled and episode_id in self.encountered)
            reference = self.baseline.get(key)
            if reference is None and not post_intervention:
                raise ValueError('Missing baseline state {}'.format(key))
            observed = {'graph_ids': ids, 'mask': mask, 'visited': visited,
                        'logits': [float(x) if math.isfinite(float(x)) else None for x in scores],
                        'graph_position': pack(positions[i]), 'current_vp': current[i],
                        'policy_index': native_index, 'effective_index': effective,
                        'budget_stop': bool(budget_stop), 'no_vp_left': bool(no_vp[i])}
            if reference is not None and not post_intervention:
                for name, value in observed.items():
                    if value != reference[name]:
                        raise ValueError('Baseline prefix differs at {} field {}'.format(key, name))
            native_action = pack(full_action(trainer.gmaps[i], current[i], ids[effective]))
            if not post_intervention:
                old = next(o for o in reference['options'] if o['index'] == effective)
                if native_action != old['action']:
                    raise ValueError('Native action identity differs')
            scheduled = bool(event and step == event['high_level_step'])
            output_index, expected_action = effective, native_action
            if scheduled:
                if key != decision_key(event) or native_index != event['baseline_index']:
                    raise ValueError('Frozen event key/native index mismatch')
                if ids != event['expected_graph_ids'] or pack(positions[i]) != event['expected_graph_position']:
                    raise ValueError('Frozen graph state mismatch')
                alternate = int(event['alternative_index'])
                if not (0 <= alternate < n) or not mask[alternate] or visited[alternate]:
                    raise ValueError('Alternative is masked/visited/absent')
                alternate_action = pack(full_action(trainer.gmaps[i], current[i], ids[alternate]))
                if alternate_action != event['alternative_action']:
                    raise ValueError('Alternative action identity changed')
                if self.enabled:
                    chosen[i] = alternate
                    output_index, expected_action = alternate, alternate_action
            elif event and step > event['high_level_step'] and episode_id not in self.encountered:
                raise ValueError('Frozen event state was missed')
            pending.append({'scene_id': key[0], 'episode_id': episode_id,
                            'high_level_step': step, 'trajectory_id': str(episode.trajectory_id),
                            'native_index': native_index, 'effective_native_index': effective,
                            'output_index': output_index, 'scheduled': scheduled,
                            'intervened': scheduled and self.enabled,
                            'baseline_prefix_verified': not post_intervention,
                            'native_action': native_action, 'expected_action': expected_action,
                            'alternative_action': event['alternative_action'] if scheduled else None,
                            'graph_ids': ids, 'mask': mask, 'visited': visited,
                            'instruction': episode.instruction.instruction_text,
                            'graphmap_sha256': state_digest(vars(trainer.gmaps[i])),
                            'policy_rng_sha256': state_digest({
                                'python': random.getstate(), 'numpy': np.random.get_state(),
                                'torch_cpu': torch.get_rng_state(),
                                'torch_cuda': torch.cuda.get_rng_state_all()}),
                            'privileged_analysis_only': True})
        self.pending = pending

    def commit(self, env_actions):
        if self.pending is None or len(self.pending) != len(env_actions):
            raise ValueError('Missing prepared action batch')
        with self.log.open('a') as stream:
            for record, actual in zip(self.pending, env_actions):
                if pack(actual['action']) != record['expected_action']:
                    raise ValueError('Executed action differs from expected action')
                record['native_action_identity_verified'] = True
                self.seen.add(decision_key(record))
                if record['scheduled']:
                    self.encountered.add(record['episode_id'])
                if record['intervened']:
                    self.applied.add(record['episode_id'])
                stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + '\n')
        self.pending = None

    def override_prev_vp(self, env_index, default):
        return default

    def finish(self):
        if self.pending is not None or self.encountered != set(self.events):
            raise ValueError('Not every frozen event was encountered')
        expected = set(self.events) if self.enabled else set()
        if self.applied != expected:
            raise ValueError('Incorrect intervention coverage')
        (self.directory / 'summary.json').write_text(json.dumps({
            'hook_completed': True, 'enabled': self.enabled,
            'decisions': len(self.seen), 'encountered_episodes': sorted(self.encountered),
            'intervened_episodes': sorted(self.applied), 'privileged_analysis_only': True,
            'no_model_trained': True}, indent=2) + '\n')


@baseline_registry.register_trainer(name='SS-ETP-NativeFullReturn')
class NativeFullReturnTrainer(RLTrainer):
    def __init__(self, config=None):
        super().__init__(config)
        if (config.IL.back_algo != 'control' or config.VIDEO_OPTION or config.RL_TOPO.ENABLED
                or not config.TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING
                or self.aaa_oracle_enabled or self.graph_selection_oracle_enabled
                or self.aaa_diagnostics_enabled or self.action_abstraction != 'default'
                or config.NUM_ENVIRONMENTS != 1):
            raise ValueError('Native full-return requires the frozen single-worker baseline')
        manifest = json.loads(Path(os.environ['ETPNAV_NATIVE_FULL_RETURN_MANIFEST']).read_text())
        baseline = {}
        with (Path(manifest['source_capture']) / 'graph_options/graph_options.jsonl').open() as stream:
            for line in stream:
                row = json.loads(line)
                for option in row['options']:
                    option.pop('embedding', None)
                baseline[decision_key(row)] = row
        # RLTrainer's frozen evaluation loop calls this exact hook name.
        self._graph_option_capture = NativeFullReturnHook(
            manifest['schedule'], baseline, Path(manifest['output_dir']) / 'hook', manifest['enabled'])

    def eval(self):
        super().eval()
        self._graph_option_capture.finish()
