"""Opt-in, read-only capture of actual graph actions before ghost consumption."""
import json
import math
import os
from pathlib import Path

import numpy as np
from habitat_baselines.common.baseline_registry import baseline_registry

from vlnce_baselines.ss_trainer_ETP import RLTrainer
from vlnce_baselines.adaptive_action.option_calibration import pack


def full_action(gmap, current, graph_id, tryout=False):
    """Mirror the existing action construction; every selected action is checked."""
    if graph_id is None:
        scores = list(gmap.node_stop_scores.items())
        stop = scores[int(np.argmax([item[1] for item in scores]))][0]
        path = [(vp, gmap.node_pos[vp]) for vp in gmap.shortest_path[current][stop]][1:]
        return {'act': 0, 'cur_vp': current, 'stop_vp': stop,
                'stop_pos': gmap.node_pos[stop], 'back_path': path, 'tryout': bool(tryout)}
    if graph_id not in gmap.ghost_pos:
        raise ValueError('Non-STOP graph action must be a live ghost: ' + str(graph_id))
    _, front = gmap.front_to_ghost_dist(graph_id)
    path = [(vp, gmap.node_pos[vp]) for vp in gmap.shortest_path[current][front]][1:]
    return {'act': 4, 'cur_vp': current, 'front_vp': front,
            'front_pos': gmap.node_pos[front], 'ghost_vp': graph_id,
            'ghost_pos': gmap.ghost_aug_pos[graph_id], 'back_path': path,
            'tryout': bool(tryout)}


def enumerate_options(gmap, current, ids, mask, visited, logits, budget_stop, no_vp, embeddings=None):
    n = len(ids)
    if not n or ids[0] is not None or any(len(x) != n for x in [mask, visited, logits]):
        raise ValueError('Graph ID, tensor length or STOP index mismatch')
    if len(set(ids)) != n:
        raise ValueError('Duplicate graph action IDs')
    if embeddings is not None and (embeddings.ndim != 2 or embeddings.shape != (n, 768)):
        raise ValueError('Expected one 768-dimensional frozen embedding per graph ID')
    options = []
    for i, graph_id in enumerate(ids):
        if not mask[i] or visited[i]:
            continue
        if not math.isfinite(float(logits[i])):
            raise ValueError('Valid unvisited action has nonfinite logit')
        if i > 0 and graph_id not in gmap.ghost_pos:
            raise ValueError('Visited or unknown graph ID exposed as a valid action')
        option = {'index': i, 'graph_id': graph_id, 'logit': float(logits[i]),
                  'current_proposal': graph_id in gmap.last_candidate_vps,
                  'admissible': i == 0 or not (budget_stop or no_vp),
                  'action': pack(full_action(gmap, current, graph_id))}
        if embeddings is not None:
            if not np.isfinite(embeddings[i]).all():
                raise ValueError('Nonfinite frozen embedding')
            option['embedding'] = pack(embeddings[i])
        options.append(option)
    if not options or options[0]['index'] != 0:
        raise ValueError('STOP must remain valid')
    return options


class GraphOptionCapture:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.path = self.directory / 'graph_options.jsonl'
        with self.path.open('x'):
            pass
        self.pending = None

    def prepare(self, trainer, step, current, positions, nav_inputs, logits,
                chosen, policy_chosen, no_vp, embeddings):
        if self.pending is not None:
            raise RuntimeError('Uncommitted previous decision')
        if not np.array_equal(chosen, policy_chosen):
            raise ValueError('Intervention changed the captured baseline action')
        self.pending = []
        for i, episode in enumerate(trainer.envs.current_episodes()):
            ids = nav_inputs['gmap_vp_ids'][i]
            n = len(ids)
            mask = nav_inputs['gmap_masks'][i, :n].detach().cpu().numpy().astype(bool)
            visited = nav_inputs['gmap_visited_masks'][i, :n].detach().cpu().numpy().astype(bool)
            scores = logits[i, :n].detach().cpu().numpy()
            features = embeddings[i, :n].detach().cpu().numpy() if embeddings is not None else None
            budget_stop = step >= trainer.max_len - 1
            policy_index = int(chosen[i])
            effective_index = 0 if policy_index == 0 or budget_stop or no_vp[i] else policy_index
            options = enumerate_options(trainer.gmaps[i], current[i], ids, mask, visited,
                                        scores, budget_stop, no_vp[i], features)
            valid = {o['index']: o for o in options}
            if policy_index not in valid or not valid[effective_index]['admissible']:
                raise ValueError('Selected action violates graph masks or episode budget')
            self.pending.append({'schema_version': 1, 'episode_id': str(episode.episode_id),
                                 'scene_id': str(episode.scene_id), 'trajectory_id': str(episode.trajectory_id),
                                 'high_level_step': step, 'current_vp': current[i],
                                 'graph_position': pack(positions[i]),
                                 'instruction': episode.instruction.instruction_text,
                                 'graph_ids': ids, 'mask': mask.tolist(), 'visited': visited.tolist(),
                                 'logits': [float(s) if math.isfinite(float(s)) else None for s in scores],
                                 'policy_index': policy_index, 'effective_index': effective_index,
                                 'budget_stop': bool(budget_stop), 'no_vp_left': bool(no_vp[i]),
                                 'forced_stop': policy_index != 0 and effective_index == 0,
                                 'options': options, 'privileged_labels_in_features': False})

    def commit(self, env_actions):
        if self.pending is None or len(env_actions) != len(self.pending):
            raise ValueError('Capture/native action batch mismatch')
        for row, native in zip(self.pending, env_actions):
            selected = next(o for o in row['options'] if o['index'] == row['effective_index'])
            if selected['action'] != pack(native['action']):
                raise ValueError('Reconstructed selected option differs from native action')
            row['native_selected_action_exact'] = True
        with self.path.open('a') as stream:
            for row in self.pending:
                stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')
        self.pending = None


@baseline_registry.register_trainer(name='SS-ETP-OptionCapture')
class GraphOptionTraceTrainer(RLTrainer):
    def __init__(self, config=None):
        super().__init__(config)
        if (config.IL.back_algo != 'control' or config.VIDEO_OPTION or config.RL_TOPO.ENABLED
                or not config.TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING
                or self.aaa_oracle_enabled or self.graph_selection_oracle_enabled
                or self.aaa_diagnostics_enabled or self.action_abstraction != 'default'):
            raise ValueError('Graph option capture requires the calibrated frozen baseline protocol')
        self._graph_option_capture = GraphOptionCapture(os.environ['ETPNAV_GRAPH_OPTION_DIR'])
