"""Privileged, opt-in one-action intervention; subsequent navigation is native.

Uses the existing pre-construction/after-construction research callbacks. No
default trainer imports this module. It never steps, rewinds or resets a simulator.
"""
import json
import math
import os
from pathlib import Path

import numpy as np
from habitat_baselines.common.baseline_registry import baseline_registry
from vlnce_baselines.ss_trainer_ETP import RLTrainer
from vlnce_baselines.adaptive_action.graph_option_capture import full_action
from vlnce_baselines.adaptive_action.option_calibration import pack


def decision_key(row):
    return row['scene_id'], str(row['episode_id']), row['high_level_step']


class SingleInterventionHook:
    def __init__(self, schedule, baseline, directory, enabled):
        self.events = {}
        for event in schedule['events']:
            episode = str(event['episode_id'])
            if episode in self.events:
                raise ValueError('Only one event per episode is allowed')
            if (event['baseline_index'] <= 0 or event['alternative_index'] <= 0
                    or event['baseline_index'] == event['alternative_index']
                    or any(event[k]['act'] != 4 for k in ['baseline_action', 'alternative_action'])):
                raise ValueError('Intervention must replace one distinct non-STOP action')
            self.events[episode] = event
        self.baseline, self.enabled = baseline, bool(enabled)
        self.encountered, self.applied, self.seen = set(), set(), set()
        self.pending = None
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.log = self.directory / 'decisions.jsonl'
        with self.log.open('x'):
            pass
        self.rows, self.prefix_checks = 0, 0

    def prepare(self, trainer, step, current, positions, nav_inputs, logits,
                chosen, policy_chosen, no_vp, embeddings, waypoint_heatmap=None):
        if self.pending is not None or not np.array_equal(chosen, policy_chosen):
            raise ValueError('Uncommitted step or unexpected upstream intervention')
        if trainer.max_len != 15:
            raise ValueError('The frozen decision horizon must stay 15')
        pending = []
        for i, episode in enumerate(trainer.envs.current_episodes()):
            episode_id = str(episode.episode_id)
            k = (str(episode.scene_id), episode_id, step)
            if k in self.seen:
                raise ValueError('Repeated navigation state key')
            ids = nav_inputs['gmap_vp_ids'][i]
            n = len(ids)
            mask = nav_inputs['gmap_masks'][i, :n].detach().cpu().numpy().astype(bool).tolist()
            visited = nav_inputs['gmap_visited_masks'][i, :n].detach().cpu().numpy().astype(bool).tolist()
            scores = logits[i, :n].detach().cpu().numpy()
            native_index = int(policy_chosen[i])
            budget_stop = step >= trainer.max_len - 1
            effective = 0 if native_index == 0 or budget_stop or no_vp[i] else native_index
            if (not ids or ids[0] is not None or len(set(ids)) != n
                    or not (0 <= native_index < n) or not mask[native_index]
                    or visited[native_index] or not math.isfinite(float(scores[native_index]))):
                raise ValueError('Invalid native graph action/mask')
            native_action = pack(full_action(trainer.gmaps[i], current[i], ids[effective]))
            observed = {'graph_ids': ids, 'mask': mask, 'visited': visited,
                        'logits': [float(x) if math.isfinite(float(x)) else None for x in scores],
                        'graph_position': pack(positions[i]), 'current_vp': current[i],
                        'policy_index': native_index, 'effective_index': effective,
                        'budget_stop': bool(budget_stop), 'no_vp_left': bool(no_vp[i])}
            in_prefix = episode_id not in self.applied
            if in_prefix:
                reference = self.baseline.get(k)
                if reference is None:
                    raise ValueError('Missing baseline prefix key: {}'.format(k))
                for name, value in observed.items():
                    if value != reference[name]:
                        raise ValueError('Baseline prefix {} differs at {}'.format(name, k))
                old = next(o for o in reference['options'] if o['index'] == effective)
                if native_action != old['action']:
                    raise ValueError('Baseline full-action identity differs')
            event = self.events.get(episode_id)
            scheduled = bool(event and step == event['high_level_step'])
            expected_action, output_index = native_action, effective
            if scheduled:
                if episode_id in self.encountered or effective == 0 or budget_stop or no_vp[i]:
                    raise ValueError('Repeated, STOP or horizon-invalid intervention')
                alternate = int(event['alternative_index'])
                if (k != decision_key(event) or native_index != event['baseline_index']
                        or ids != event['expected_graph_ids'] or pack(positions[i]) != event['expected_graph_position']
                        or native_action != event['baseline_action']):
                    raise ValueError('Frozen event no longer matches native state')
                if (not (0 < alternate < n) or not mask[alternate] or visited[alternate]
                        or not math.isfinite(float(scores[alternate]))):
                    raise ValueError('Alternative is masked, visited, STOP or absent')
                alternate_action = pack(full_action(trainer.gmaps[i], current[i], ids[alternate]))
                if alternate_action != event['alternative_action']:
                    raise ValueError('Alternative full-action identity changed')
                if self.enabled:
                    # The original action builder uses this same mutable array
                    # next, including front-node update and ghost consumption.
                    chosen[i] = alternate
                    output_index, expected_action = alternate, alternate_action
            elif event and step > event['high_level_step'] and episode_id not in self.encountered:
                raise ValueError('Frozen intervention state was missed')
            record = {'scene_id': k[0], 'episode_id': episode_id, 'high_level_step': step,
                      'trajectory_id': str(episode.trajectory_id), 'native_index': native_index,
                      'effective_native_index': effective, 'output_index': output_index,
                      'scheduled': scheduled, 'intervened': scheduled and self.enabled,
                      'baseline_prefix_verified': in_prefix, 'native_action': native_action,
                      'expected_action': expected_action, 'budget_stop': bool(budget_stop),
                      'native_stop_preserved': (effective == 0) == (output_index == 0),
                      'graph_ids': ids, 'mask': mask, 'visited': visited,
                      'privileged_analysis_only': True}
            pending.append((record, trainer.gmaps[i], bool(trainer.config.MODEL.consume_ghost)))
        self.pending = pending

    def commit(self, env_actions):
        if self.pending is None or len(self.pending) != len(env_actions):
            raise ValueError('Missing prepared native action batch')
        for (record, gmap, consume), actual in zip(self.pending, env_actions):
            if pack(actual['action']) != record['expected_action']:
                raise ValueError('Native builder executed the wrong full action')
            if record['expected_action']['act'] == 4 and consume:
                if record['expected_action']['ghost_vp'] in gmap.ghost_pos:
                    raise ValueError('Native builder failed to consume selected ghost')
            record['native_action_identity_verified'] = True
            record['native_ghost_consumption_verified'] = consume
        with self.log.open('a') as stream:
            for record, _, _ in self.pending:
                k = decision_key(record)
                self.seen.add(k)
                if record['scheduled']:
                    self.encountered.add(k[1])
                if record['intervened']:
                    self.applied.add(k[1])
                self.rows += 1
                self.prefix_checks += record['baseline_prefix_verified']
                stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + '\n')
        self.pending = None

    def finish(self):
        if self.pending is not None or self.encountered != set(self.events):
            raise ValueError('Not every frozen event was encountered exactly once')
        expected = set(self.events) if self.enabled else set()
        if self.applied != expected:
            raise ValueError('Incorrect intervention coverage')
        summary = {'hook_completed': True, 'enabled': self.enabled, 'decisions': self.rows,
                   'prefix_states_verified': self.prefix_checks,
                   'encountered_episodes': sorted(self.encountered), 'intervened_episodes': sorted(self.applied),
                   'privileged_analysis_only': True, 'no_model_trained': True}
        (self.directory / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')


@baseline_registry.register_trainer(name='SS-ETP-SingleIntervention')
class SingleInterventionTrainer(RLTrainer):
    def __init__(self, config=None):
        super().__init__(config)
        if (config.IL.back_algo != 'control' or config.VIDEO_OPTION or config.RL_TOPO.ENABLED
                or not config.TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING
                or self.aaa_oracle_enabled or self.graph_selection_oracle_enabled
                or self.aaa_diagnostics_enabled or self.action_abstraction != 'default'
                or config.NUM_ENVIRONMENTS != 1):
            raise ValueError('Intervention requires the frozen single-worker baseline')
        manifest = json.loads(Path(os.environ['ETPNAV_SINGLE_MANIFEST']).read_text())
        baseline = {}
        path = Path(manifest['source_capture']) / 'graph_options/graph_options.jsonl'
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                k = decision_key(row)
                if k in baseline:
                    raise ValueError('Duplicate source graph record')
                for option in row['options']:
                    option.pop('embedding', None)
                baseline[k] = row
        self._graph_option_capture = SingleInterventionHook(
            manifest['schedule'], baseline, Path(manifest['output_dir']) / 'hook', manifest['enabled'])

    def eval(self):
        super().eval()
        self._graph_option_capture.finish()
