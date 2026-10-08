import gc
import os
import sys
import random
import warnings
from collections import defaultdict
from typing import Dict, List
import jsonlines

import lmdb
import msgpack_numpy
import numpy as np
import math
import time
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parallel import DistributedDataParallel as DDP

import tqdm
from gym import Space
from habitat import Config, logger
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
    apply_obs_transforms_obs_space,
    get_active_obs_transforms,
)
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.utils.common import batch_obs

from vlnce_baselines.common.aux_losses import AuxLosses
from vlnce_baselines.common.base_il_trainer import BaseVLNCETrainer
from vlnce_baselines.common.env_utils import construct_envs, construct_envs_for_rl, is_slurm_batch_job
from vlnce_baselines.common.utils import extract_instruction_tokens
from vlnce_baselines.models.graph_utils import GraphMap, MAX_DIST
from vlnce_baselines.utils import reduce_loss

from .utils import get_camera_orientations12
from .utils import (
    length2mask, dir_angle_feature_with_ele,
)
from vlnce_baselines.common.utils import dis_to_con, gather_list_and_concat
from habitat_extensions.measures import NDTW, StepsTaken
from fastdtw import fastdtw

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import tensorflow as tf  # noqa: F401

import torch.distributed as distr
import gzip
import json
from copy import deepcopy
from torch.cuda.amp import autocast, GradScaler
from vlnce_baselines.common.ops import pad_tensors_wgrad, gen_seq_masks
from vlnce_baselines.adaptive_action.diagnostics import DecisionLogger
from vlnce_baselines.adaptive_action.action_generators import (
    ACTION_ABSTRACTIONS,
    candidate_geometry_from_heatmap,
)
from vlnce_baselines.adaptive_action.oracle_selector import select_oracle_level
from vlnce_baselines.adaptive_action.graph_oracle_controls import controlled_graph_action
from torch.nn.utils.rnn import pad_sequence


@baseline_registry.register_trainer(name="SS-ETP")
class RLTrainer(BaseVLNCETrainer):
    def __init__(self, config=None):
        super().__init__(config)
        self.max_len = int(config.IL.max_traj_len) #  * 0.97 transfered gt path got 0.96 spl
        self._rl_topo_debug_transition_count = 0
        self._rl_topo_rollout_id = 0
        abstraction_cfg = getattr(config, "ACTION_ABSTRACTION", None)
        self.action_abstraction = (
            str(abstraction_cfg.LEVEL) if abstraction_cfg is not None else "default"
        )
        self.aaa_diagnostics_enabled = bool(
            abstraction_cfg is not None and abstraction_cfg.DIAGNOSTICS_ENABLED
        )
        self.aaa_diagnostic_max_records = int(
            abstraction_cfg.DIAGNOSTICS_MAX_RECORDS
        ) if abstraction_cfg is not None else -1
        self.aaa_logger = DecisionLogger(
            abstraction_cfg.DIAGNOSTICS_PATH
            if self.aaa_diagnostics_enabled else None,
            rank=getattr(config, "local_rank", 0),
        )
        self.aaa_oracle_enabled = bool(
            abstraction_cfg is not None and abstraction_cfg.ORACLE_ENABLED
        )
        self.aaa_oracle_max_records = int(
            abstraction_cfg.ORACLE_MAX_RECORDS
        ) if abstraction_cfg is not None else -1
        self.aaa_oracle_logger = DecisionLogger(
            abstraction_cfg.ORACLE_PATH
            if self.aaa_oracle_enabled else None,
            rank=getattr(config, "local_rank", 0),
        )
        self.graph_selection_oracle_enabled = bool(
            abstraction_cfg is not None and abstraction_cfg.GRAPH_SELECTION_ORACLE_ENABLED
        )
        self.graph_selection_oracle_mode = str(abstraction_cfg.GRAPH_SELECTION_ORACLE_MODE)
        self.aaa_include_ranker_embeddings = bool(
            abstraction_cfg is not None and abstraction_cfg.INCLUDE_RANKER_EMBEDDINGS
        )
        self.graph_selection_oracle_logger = DecisionLogger(
            abstraction_cfg.GRAPH_SELECTION_ORACLE_PATH
            if self.graph_selection_oracle_enabled else None,
            rank=getattr(config, "local_rank", 0),
        )

    def _make_dirs(self):
        if self.config.local_rank == 0:
            self._make_ckpt_dir()
            # os.makedirs(self.lmdb_features_dir, exist_ok=True)
            if self.config.EVAL.SAVE_RESULTS:
                self._make_results_dir()

    def save_checkpoint(self, iteration: int):
        torch.save(
            obj={
                "state_dict": self.policy.state_dict(),
                "config": self.config,
                "optim_state": self.optimizer.state_dict(),
                "critic_optim_state": (
                    self.critic_optimizer.state_dict()
                    if getattr(self, "critic_optimizer", None) is not None
                    else None
                ),
                "ppo_optim_state": (
                    self.critic_optimizer.state_dict()
                    if getattr(self, "critic_optimizer", None) is not None
                    else None
                ),
                "iteration": iteration,
            },
            f=os.path.join(self.config.CHECKPOINT_FOLDER, f"ckpt.iter{iteration}.pth"),
        )

    def _set_config(self):
        self.split = self.config.TASK_CONFIG.DATASET.SPLIT
        self.config.defrost()
        self.config.TASK_CONFIG.TASK.NDTW.SPLIT = self.split
        self.config.TASK_CONFIG.TASK.SDTW.SPLIT = self.split
        self.config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = -1
        self.config.SIMULATOR_GPU_IDS = self.config.SIMULATOR_GPU_IDS[self.config.local_rank]
        self.config.use_pbar = not is_slurm_batch_job()
        ''' if choosing image '''
        resize_config = self.config.RL.POLICY.OBS_TRANSFORMS.RESIZER_PER_SENSOR.SIZES
        crop_config = self.config.RL.POLICY.OBS_TRANSFORMS.CENTER_CROPPER_PER_SENSOR.SENSOR_CROPS
        task_config = self.config.TASK_CONFIG
        camera_orientations = get_camera_orientations12()
        for sensor_type in ["RGB", "DEPTH"]:
            resizer_size = dict(resize_config)[sensor_type.lower()]
            cropper_size = dict(crop_config)[sensor_type.lower()]
            sensor = getattr(task_config.SIMULATOR, f"{sensor_type}_SENSOR")
            for action, orient in camera_orientations.items():
                camera_template = f"{sensor_type}_{action}"
                camera_config = deepcopy(sensor)
                camera_config.ORIENTATION = camera_orientations[action]
                camera_config.UUID = camera_template.lower()
                setattr(task_config.SIMULATOR, camera_template, camera_config)
                task_config.SIMULATOR.AGENT_0.SENSORS.append(camera_template)
                resize_config.append((camera_template.lower(), resizer_size))
                crop_config.append((camera_template.lower(), cropper_size))
        self.config.RL.POLICY.OBS_TRANSFORMS.RESIZER_PER_SENSOR.SIZES = resize_config
        self.config.RL.POLICY.OBS_TRANSFORMS.CENTER_CROPPER_PER_SENSOR.SENSOR_CROPS = crop_config
        self.config.TASK_CONFIG = task_config
        self.config.SENSORS = task_config.SIMULATOR.AGENT_0.SENSORS
        if self.config.VIDEO_OPTION:
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP_VLNCE")
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("DISTANCE_TO_GOAL")
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("SUCCESS")
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("SPL")
            os.makedirs(self.config.VIDEO_DIR, exist_ok=True)
            shift = 0.
            orient_dict = {
                'Back': [0, math.pi + shift, 0],            # Back
                'Down': [-math.pi / 2, 0 + shift, 0],       # Down
                'Front':[0, 0 + shift, 0],                  # Front
                'Right':[0, math.pi / 2 + shift, 0],        # Right
                'Left': [0, 3 / 2 * math.pi + shift, 0],    # Left
                'Up':   [math.pi / 2, 0 + shift, 0],        # Up
            }
            sensor_uuids = []
            H = 224
            for sensor_type in ["RGB"]:
                sensor = getattr(self.config.TASK_CONFIG.SIMULATOR, f"{sensor_type}_SENSOR")
                for camera_id, orient in orient_dict.items():
                    camera_template = f"{sensor_type}{camera_id}"
                    camera_config = deepcopy(sensor)
                    camera_config.WIDTH = H
                    camera_config.HEIGHT = H
                    camera_config.ORIENTATION = orient
                    camera_config.UUID = camera_template.lower()
                    camera_config.HFOV = 90
                    sensor_uuids.append(camera_config.UUID)
                    setattr(self.config.TASK_CONFIG.SIMULATOR, camera_template, camera_config)
                    self.config.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS.append(camera_template)
        self.config.freeze()

        self.world_size = self.config.GPU_NUMBERS
        self.local_rank = self.config.local_rank
        self.batch_size = self.config.IL.batch_size
        torch.cuda.set_device(self.device)
        if self.world_size > 1:
            distr.init_process_group(backend='nccl', init_method='env://')
            self.device = self.config.TORCH_GPU_IDS[self.local_rank]
            self.config.defrost()
            self.config.TORCH_GPU_ID = self.config.TORCH_GPU_IDS[self.local_rank]
            self.config.freeze()
            torch.cuda.set_device(self.device)

    def _init_envs(self):
        # for DDP to load different data
        self.config.defrost()
        self.config.TASK_CONFIG.SEED = self.config.TASK_CONFIG.SEED + self.local_rank
        self.config.freeze()

        self.envs = construct_envs(
            self.config, 
            get_env_class(self.config.ENV_NAME),
            auto_reset_done=False
        )
        env_num = self.envs.num_envs
        dataset_len = sum(self.envs.number_of_episodes)
        logger.info(f'LOCAL RANK: {self.local_rank}, ENV NUM: {env_num}, DATASET LEN: {dataset_len}')
        observation_space = self.envs.observation_spaces[0]
        action_space = self.envs.action_spaces[0]
        self.obs_transforms = get_active_obs_transforms(self.config)
        observation_space = apply_obs_transforms_obs_space(
            observation_space, self.obs_transforms
        )

        return observation_space, action_space

    def _critic_module(self):
        """Return the critic independently of whether ETP is DDP-wrapped."""
        net = self.policy.net
        if isinstance(net, DDP):
            net = net.module
        return net.critic

    def _actor_head_module(self):
        """Return the trainable residual actor head."""
        net = self.policy.net
        if isinstance(net, DDP):
            net = net.module
        return net.residual_sap_head

    def _il_actor_head_module(self):
        """Return the frozen IL reference SAP head."""
        net = self.policy.net
        if isinstance(net, DDP):
            net = net.module
        return net.vln_bert.global_sap_head

    def _initialize_policy(
        self,
        config: Config,
        load_from_ckpt: bool,
        observation_space: Space,
        action_space: Space,
        load_optimizer: bool = True,
    ):
        start_iter = 0
        policy = baseline_registry.get_policy(self.config.MODEL.policy_name)
        self.policy = policy.from_config(
            config=config,
            observation_space=observation_space,
            action_space=action_space,
        )
        ''' initialize the waypoint predictor here '''
        from vlnce_baselines.waypoint_pred.TRM_net import BinaryDistPredictor_TRM
        self.waypoint_predictor = BinaryDistPredictor_TRM(device=self.device)
        cwp_fn = 'data/wp_pred/check_cwp_bestdist_hfov63' if self.config.MODEL.task_type == 'rxr' else 'data/wp_pred/check_cwp_bestdist_hfov90'
        self.waypoint_predictor.load_state_dict(torch.load(cwp_fn, map_location = torch.device('cpu'))['predictor']['state_dict'])
        for param in self.waypoint_predictor.parameters():
            param.requires_grad_(False)

        self.policy.to(self.device)
        self.waypoint_predictor.to(self.device)
        self.num_recurrent_layers = self.policy.net.num_recurrent_layers

        rl_topo_enabled = bool(self.config.RL_TOPO.ENABLED)
        if rl_topo_enabled:
            # Keep the ETP encoder and IL SAP head frozen so replayed graph
            # embeddings remain exact. PPO learns only a zero-initialized
            # residual actor and the critic.
            for name, param in self.policy.named_parameters():
                param.requires_grad_(
                    "net.critic." in name
                    or "net.residual_sap_head." in name
                )
        else:
            # The optional heads are not part of the original IL loss. Keep
            # them frozen so released IL checkpoints retain their optimizer
            # parameter set.
            for name, param in self.policy.named_parameters():
                if "net.critic." in name or "net.residual_sap_head." in name:
                    param.requires_grad_(False)

        if self.config.GPU_NUMBERS > 1:
            print('Using', self.config.GPU_NUMBERS,'GPU!')
            # PPO replays each snapshot through both trainable heads, so all
            # parameters that require gradients participate in the update.
            self.policy.net = DDP(self.policy.net.to(self.device), device_ids=[self.device],
                output_device=self.device,
                find_unused_parameters=False,
                broadcast_buffers=False)

        if rl_topo_enabled:
            critic = self._critic_module()
            actor_head = self._actor_head_module()
            ppo_parameters = [
                {
                    "params": list(actor_head.parameters()),
                    "lr": float(self.config.RL_TOPO.ACTOR_LR),
                },
                {
                    "params": list(critic.parameters()),
                    "lr": float(self.config.RL_TOPO.CRITIC_LR),
                },
            ]
            self.critic_optimizer = torch.optim.AdamW(
                ppo_parameters
            )
            # Keep the legacy attribute for checkpoint compatibility.  The
            # training loop selects this PPO optimizer explicitly.
            self.optimizer = self.critic_optimizer
            logger.info(
                "RL_TOPO residual PPO parameters: critic=%d, residual_head=%d, "
                "IL_head=frozen, alpha=%.3f",
                sum(p.numel() for p in critic.parameters()),
                sum(p.numel() for p in actor_head.parameters()),
                float(self.config.RL_TOPO.RESIDUAL_ALPHA),
            )
        else:
            self.critic_optimizer = None
            self.optimizer = torch.optim.AdamW(
                [p for p in self.policy.parameters() if p.requires_grad],
                lr=self.config.IL.lr,
            )

        if load_from_ckpt:
            if config.IL.is_requeue:
                import glob
                ckpt_list = list(filter(os.path.isfile, glob.glob(config.CHECKPOINT_FOLDER + "/*")) )
                ckpt_list.sort(key=os.path.getmtime)
                ckpt_path = ckpt_list[-1]
            else:
                ckpt_path = config.IL.ckpt_to_load
            ckpt_dict = self.load_checkpoint(ckpt_path, map_location="cpu")
            start_iter = ckpt_dict["iteration"]

            if 'module' in list(ckpt_dict['state_dict'].keys())[0] and self.config.GPU_NUMBERS == 1:
                self.policy.net = torch.nn.DataParallel(self.policy.net.to(self.device),
                    device_ids=[self.device], output_device=self.device)
                self.policy.load_state_dict(ckpt_dict["state_dict"], strict=False)
                self.policy.net = self.policy.net.module
                self.waypoint_predictor = torch.nn.DataParallel(self.waypoint_predictor.to(self.device),
                    device_ids=[self.device], output_device=self.device)
            else:
                self.policy.load_state_dict(ckpt_dict["state_dict"], strict=False)
            if rl_topo_enabled:
                if config.IL.is_requeue:
                    ppo_optim_state = ckpt_dict.get("ppo_optim_state")
                    if ppo_optim_state is None:
                        ppo_optim_state = ckpt_dict.get("critic_optim_state")
                    if ppo_optim_state is not None:
                        try:
                            self.critic_optimizer.load_state_dict(ppo_optim_state)
                        except (ValueError, RuntimeError) as exc:
                            logger.warning(
                                "Skipping incompatible PPO optimizer state: %s",
                                exc,
                            )
            elif load_optimizer and not config.IL.is_requeue:
                self.optimizer.load_state_dict(ckpt_dict["optim_state"])
        if rl_topo_enabled:
            # Snapshot the frozen IL reference for diagnostics.  This is
            # intentionally taken after checkpoint loading.
            self._initial_il_actor_params = {
                name: param.detach().float().cpu().clone()
                for name, param in self._il_actor_head_module().named_parameters()
            }
            logger.info(f"Loaded weights from checkpoint: {ckpt_path}, iteration: {start_iter}")
			
        params = sum(param.numel() for param in self.policy.parameters())
        params_t = sum(
            p.numel() for p in self.policy.parameters() if p.requires_grad
        )
        logger.info(f"Agent parameters: {params/1e6:.2f} MB. Trainable: {params_t/1e6:.2f} MB.")
        logger.info("Finished setting up policy.")

        return start_iter

    def _teacher_action(self, batch_angles, batch_distances, candidate_lengths):
        if self.config.MODEL.task_type == 'r2r':
            cand_dists_to_goal = [[] for _ in range(len(batch_angles))]
            oracle_cand_idx = []
            for j in range(len(batch_angles)):
                for k in range(len(batch_angles[j])):
                    angle_k = batch_angles[j][k]
                    forward_k = batch_distances[j][k]
                    dist_k = self.envs.call_at(j, "cand_dist_to_goal", {"angle": angle_k, "forward": forward_k})
                    cand_dists_to_goal[j].append(dist_k)
                curr_dist_to_goal = self.envs.call_at(j, "current_dist_to_goal")
                # if within target range (which def as 3.0)
                if curr_dist_to_goal < 1.5:
                    oracle_cand_idx.append(candidate_lengths[j] - 1)
                else:
                    oracle_cand_idx.append(np.argmin(cand_dists_to_goal[j]))
            return oracle_cand_idx
        elif self.config.MODEL.task_type == 'rxr':
            kargs = []
            current_episodes = self.envs.current_episodes()
            for i in range(self.envs.num_envs):
                kargs.append({
                    'ref_path':self.gt_data[str(current_episodes[i].episode_id)]['locations'],
                    'angles':batch_angles[i],
                    'distances':batch_distances[i],
                    'candidate_length':candidate_lengths[i]
                })
            oracle_cand_idx = self.envs.call(["get_cand_idx"]*self.envs.num_envs, kargs)
            return oracle_cand_idx

    def _teacher_action_new(self, batch_gmap_vp_ids, batch_no_vp_left):
        teacher_actions = []
        cur_episodes = self.envs.current_episodes()
        for i, (gmap_vp_ids, gmap, no_vp_left) in enumerate(zip(batch_gmap_vp_ids, self.gmaps, batch_no_vp_left)):
            curr_dis_to_goal = self.envs.call_at(i, "current_dist_to_goal")
            if curr_dis_to_goal < 1.5:
                teacher_actions.append(0)
            else:
                if no_vp_left:
                    teacher_actions.append(-100)
                elif self.config.IL.expert_policy == 'spl':
                    ghost_vp_pos = [(vp, random.choice(pos)) for vp, pos in gmap.ghost_real_pos.items()]
                    ghost_dis_to_goal = [
                        self.envs.call_at(i, "point_dist_to_goal", {"pos": p[1]})
                        for p in ghost_vp_pos
                    ]
                    target_ghost_vp = ghost_vp_pos[np.argmin(ghost_dis_to_goal)][0]
                    teacher_actions.append(gmap_vp_ids.index(target_ghost_vp))
                elif self.config.IL.expert_policy == 'ndtw':
                    ghost_vp_pos = [(vp, random.choice(pos)) for vp, pos in gmap.ghost_real_pos.items()]
                    target_ghost_vp = self.envs.call_at(i, "ghost_dist_to_ref", {
                        "ghost_vp_pos": ghost_vp_pos,
                        "ref_path": self.gt_data[str(cur_episodes[i].episode_id)]['locations'],
                    })
                    teacher_actions.append(gmap_vp_ids.index(target_ghost_vp))
                else:
                    raise NotImplementedError
       
        return torch.tensor(teacher_actions).cuda()

    def _vp_feature_variable(self, obs):
        batch_rgb_fts, batch_dep_fts, batch_loc_fts = [], [], []
        batch_nav_types, batch_view_lens = [], []

        for i in range(self.envs.num_envs):
            rgb_fts, dep_fts, loc_fts , nav_types = [], [], [], []
            cand_idxes = np.zeros(12, dtype=np.bool)
            cand_idxes[obs['cand_img_idxes'][i]] = True
            # cand
            rgb_fts.append(obs['cand_rgb'][i])
            dep_fts.append(obs['cand_depth'][i])
            loc_fts.append(obs['cand_angle_fts'][i])
            nav_types += [1] * len(obs['cand_angles'][i])
            # non-cand
            rgb_fts.append(obs['pano_rgb'][i][~cand_idxes])
            dep_fts.append(obs['pano_depth'][i][~cand_idxes])
            loc_fts.append(obs['pano_angle_fts'][~cand_idxes])
            nav_types += [0] * (12-np.sum(cand_idxes))
            
            batch_rgb_fts.append(torch.cat(rgb_fts, dim=0))
            batch_dep_fts.append(torch.cat(dep_fts, dim=0))
            batch_loc_fts.append(torch.cat(loc_fts, dim=0))
            batch_nav_types.append(torch.LongTensor(nav_types))
            batch_view_lens.append(len(nav_types))
        # collate
        batch_rgb_fts = pad_tensors_wgrad(batch_rgb_fts)
        batch_dep_fts = pad_tensors_wgrad(batch_dep_fts)
        batch_loc_fts = pad_tensors_wgrad(batch_loc_fts).cuda()
        batch_nav_types = pad_sequence(batch_nav_types, batch_first=True).cuda()
        batch_view_lens = torch.LongTensor(batch_view_lens).cuda()

        return {
            'rgb_fts': batch_rgb_fts, 'dep_fts': batch_dep_fts, 'loc_fts': batch_loc_fts,
            'nav_types': batch_nav_types, 'view_lens': batch_view_lens,
        }
        
    def _nav_gmap_variable(self, cur_vp, cur_pos, cur_ori):
        batch_gmap_vp_ids, batch_gmap_step_ids, batch_gmap_lens = [], [], []
        batch_gmap_img_fts, batch_gmap_pos_fts = [], []
        batch_gmap_pair_dists, batch_gmap_visited_masks = [], []
        batch_no_vp_left = []

        for i, gmap in enumerate(self.gmaps):
            node_vp_ids = list(gmap.node_pos.keys())
            ghost_vp_ids = list(gmap.ghost_pos.keys())
            if len(ghost_vp_ids) == 0:
                batch_no_vp_left.append(True)
            else:
                batch_no_vp_left.append(False)

            gmap_vp_ids = [None] + node_vp_ids + ghost_vp_ids
            gmap_step_ids = [0] + [gmap.node_stepId[vp] for vp in node_vp_ids] + [0]*len(ghost_vp_ids)
            gmap_visited_masks = [0] + [1] * len(node_vp_ids) + [0] * len(ghost_vp_ids)

            gmap_img_fts = [gmap.get_node_embeds(vp) for vp in node_vp_ids] + \
                           [gmap.get_node_embeds(vp) for vp in ghost_vp_ids]
            gmap_img_fts = torch.stack(
                [torch.zeros_like(gmap_img_fts[0])] + gmap_img_fts, dim=0
            )

            gmap_pos_fts = gmap.get_pos_fts(
                cur_vp[i], cur_pos[i], cur_ori[i], gmap_vp_ids
            )
            gmap_pair_dists = np.zeros((len(gmap_vp_ids), len(gmap_vp_ids)), dtype=np.float32)
            for j in range(1, len(gmap_vp_ids)):
                for k in range(j+1, len(gmap_vp_ids)):
                    vp1 = gmap_vp_ids[j]
                    vp2 = gmap_vp_ids[k]
                    if not vp1.startswith('g') and not vp2.startswith('g'):
                        dist = gmap.shortest_dist[vp1][vp2]
                    elif not vp1.startswith('g') and vp2.startswith('g'):
                        front_dis2, front_vp2 = gmap.front_to_ghost_dist(vp2)
                        dist = gmap.shortest_dist[vp1][front_vp2] + front_dis2
                    elif vp1.startswith('g') and vp2.startswith('g'):
                        front_dis1, front_vp1 = gmap.front_to_ghost_dist(vp1)
                        front_dis2, front_vp2 = gmap.front_to_ghost_dist(vp2)
                        dist = front_dis1 + gmap.shortest_dist[front_vp1][front_vp2] + front_dis2
                    else:
                        raise NotImplementedError
                    gmap_pair_dists[j, k] = gmap_pair_dists[k, j] = dist / MAX_DIST
            
            batch_gmap_vp_ids.append(gmap_vp_ids)
            batch_gmap_step_ids.append(torch.LongTensor(gmap_step_ids))
            batch_gmap_lens.append(len(gmap_vp_ids))
            batch_gmap_img_fts.append(gmap_img_fts)
            batch_gmap_pos_fts.append(torch.from_numpy(gmap_pos_fts))
            batch_gmap_pair_dists.append(torch.from_numpy(gmap_pair_dists))
            batch_gmap_visited_masks.append(torch.BoolTensor(gmap_visited_masks))
        
        # collate
        batch_gmap_step_ids = pad_sequence(batch_gmap_step_ids, batch_first=True).cuda()
        batch_gmap_img_fts = pad_tensors_wgrad(batch_gmap_img_fts)
        batch_gmap_pos_fts = pad_tensors_wgrad(batch_gmap_pos_fts).cuda()
        batch_gmap_lens = torch.LongTensor(batch_gmap_lens)
        batch_gmap_masks = gen_seq_masks(batch_gmap_lens).cuda()
        batch_gmap_visited_masks = pad_sequence(batch_gmap_visited_masks, batch_first=True).cuda()

        bs = self.envs.num_envs
        max_gmap_len = max(batch_gmap_lens)
        gmap_pair_dists = torch.zeros(bs, max_gmap_len, max_gmap_len).float()
        for i in range(bs):
            gmap_pair_dists[i, :batch_gmap_lens[i], :batch_gmap_lens[i]] = batch_gmap_pair_dists[i]
        gmap_pair_dists = gmap_pair_dists.cuda()

        return {
            'gmap_vp_ids': batch_gmap_vp_ids, 'gmap_step_ids': batch_gmap_step_ids,
            'gmap_img_fts': batch_gmap_img_fts, 'gmap_pos_fts': batch_gmap_pos_fts, 
            'gmap_masks': batch_gmap_masks, 'gmap_visited_masks': batch_gmap_visited_masks, 'gmap_pair_dists': gmap_pair_dists,
            'no_vp_left': batch_no_vp_left,
        }

    def _history_variable(self, obs):
        batch_size = obs['pano_rgb'].shape[0]
        hist_rgb_fts = obs['pano_rgb'][:, 0, ...].cuda()
        hist_pano_rgb_fts = obs['pano_rgb'].cuda()
        hist_pano_ang_fts = obs['pano_angle_fts'].unsqueeze(0).expand(batch_size, -1, -1).cuda()

        return hist_rgb_fts, hist_pano_rgb_fts, hist_pano_ang_fts

    @staticmethod
    def _pause_envs(envs, batch, envs_to_pause):
        if len(envs_to_pause) > 0:
            state_index = list(range(envs.num_envs))
            for idx in reversed(envs_to_pause):
                state_index.pop(idx)
                envs.pause_at(idx)
            
            for k, v in batch.items():
                batch[k] = v[state_index]

        return envs, batch

    def train(self):
        self._set_config()
        if self.config.MODEL.task_type == 'rxr':
            self.gt_data = {}
            for role in self.config.TASK_CONFIG.DATASET.ROLES:
                with gzip.open(
                    self.config.TASK_CONFIG.TASK.NDTW.GT_PATH.format(
                        split=self.split, role=role
                    ), "rt") as f:
                    self.gt_data.update(json.load(f))

        observation_space, action_space = self._init_envs()
        start_iter = self._initialize_policy(
            self.config,
            self.config.IL.load_from_ckpt,
            observation_space=observation_space,
            action_space=action_space,
        )

        total_iter = self.config.IL.iters
        log_every  = self.config.IL.log_every
        writer     = TensorboardWriter(self.config.TENSORBOARD_DIR if self.local_rank < 1 else None)

        self.scaler = GradScaler()
        logger.info('Traning Starts... GOOD LUCK!')
        for idx in range(start_iter, total_iter, log_every):
            interval = min(log_every, max(total_iter-idx, 0))
            cur_iter = idx + interval

            sample_ratio = self.config.IL.sample_ratio ** (idx // self.config.IL.decay_interval + 1)
            # sample_ratio = self.config.IL.sample_ratio ** (idx // self.config.IL.decay_interval)
            logs = self._train_interval(interval, self.config.IL.ml_weight, sample_ratio)

            if self.local_rank < 1:
                loss_str = f'iter {cur_iter}: '
                for k, v in logs.items():
                    logs[k] = np.mean(v)
                    loss_str += f'{k}: {logs[k]:.3f}, '
                    writer.add_scalar(f'loss/{k}', logs[k], cur_iter)
                logger.info(loss_str)
                self.save_checkpoint(cur_iter)
        
    def _train_interval(self, interval, ml_weight, sample_ratio):
        self.policy.train()
        if bool(self.config.RL_TOPO.ENABLED):
            # Keep the frozen actor deterministic while leaving the critic in
            # training mode.
            self.policy.eval()
            self._critic_module().train()
            # The exact old distribution was collected with dropout disabled.
            self._actor_head_module().eval()
        if self.world_size > 1:
            self.policy.net.module.rgb_encoder.eval()
            self.policy.net.module.depth_encoder.eval()
        else:
            self.policy.net.rgb_encoder.eval()
            self.policy.net.depth_encoder.eval()
        self.waypoint_predictor.eval()

        if self.local_rank < 1:
            pbar = tqdm.trange(interval, leave=False, dynamic_ncols=True)
        else:
            pbar = range(interval)
        self.logs = defaultdict(list)

        for idx in pbar:
            if bool(self.config.RL_TOPO.ENABLED):
                with autocast():
                    self.rollout('train', ml_weight, sample_ratio)

                ppo_epochs = int(self.config.RL_TOPO.PPO_EPOCHS)
                if ppo_epochs < 1:
                    raise ValueError("RL_TOPO.PPO_EPOCHS must be >= 1")
                for _ in range(ppo_epochs):
                    self.critic_optimizer.zero_grad()
                    # PPO replay is intentionally kept in FP32.  The critic
                    # can otherwise produce non-finite gradients under the
                    # legacy CUDA AMP stack even when the scalar loss is
                    # finite.
                    with autocast(enabled=False):
                        ppo_loss = self._compute_ppo_loss()
                    self.scaler.scale(ppo_loss).backward()
                    self.scaler.unscale_(self.critic_optimizer)
                    actor_grad_norm = torch.sqrt(sum(
                        param.grad.detach().float().pow(2).sum()
                        for param in self._actor_head_module().parameters()
                        if param.grad is not None
                    )).item()
                    if not np.isfinite(actor_grad_norm):
                        for name, param in self._actor_head_module().named_parameters():
                            if param.grad is not None:
                                logger.warning(
                                    "RL_TOPO actor grad %s finite=%s min=%s max=%s",
                                    name,
                                    bool(torch.isfinite(param.grad).all()),
                                    float(torch.nan_to_num(param.grad.detach()).min()),
                                    float(torch.nan_to_num(param.grad.detach()).max()),
                                )
                    critic_grad_norm = torch.sqrt(sum(
                        param.grad.detach().float().pow(2).sum()
                        for param in self._critic_module().parameters()
                        if param.grad is not None
                    )).item()
                    self.logs['ppo/actor_grad_norm'].append(float(actor_grad_norm))
                    self.logs['ppo/critic_grad_norm'].append(float(critic_grad_norm))
                    torch.nn.utils.clip_grad_norm_(
                        self.critic_optimizer.param_groups[0]['params']
                        + self.critic_optimizer.param_groups[1]['params'],
                        float(self.config.RL_TOPO.MAX_GRAD_NORM),
                    )
                    self.scaler.step(self.critic_optimizer)
                    self.scaler.update()
                    self._record_ppo_parameter_diagnostics()
            else:
                self.optimizer.zero_grad()
                self.loss = 0.
                with autocast():
                    self.rollout('train', ml_weight, sample_ratio)
                self.scaler.scale(self.loss).backward() # self.loss.backward()
                self.scaler.step(self.optimizer)         # optimizer.step()
                self.scaler.update()

            if self.local_rank < 1:
                pbar.set_postfix({'iter': f'{idx+1}/{interval}'})
            
        return deepcopy(self.logs)

    @torch.no_grad()
    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = 0,
    ):
        if self.local_rank < 1:
            logger.info(f"checkpoint_path: {checkpoint_path}")
        self.config.defrost()
        self.config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
        self.config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = -1
        self.config.IL.ckpt_to_load = checkpoint_path
        if self.config.VIDEO_OPTION:
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP_VLNCE")
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("DISTANCE_TO_GOAL")
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("SUCCESS")
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("SPL")
            os.makedirs(self.config.VIDEO_DIR, exist_ok=True)
            shift = 0.
            orient_dict = {
                'Back': [0, math.pi + shift, 0],            # Back
                'Down': [-math.pi / 2, 0 + shift, 0],       # Down
                'Front':[0, 0 + shift, 0],                  # Front
                'Right':[0, math.pi / 2 + shift, 0],        # Right
                'Left': [0, 3 / 2 * math.pi + shift, 0],    # Left
                'Up':   [math.pi / 2, 0 + shift, 0],        # Up
            }
            sensor_uuids = []
            H = 224
            for sensor_type in ["RGB"]:
                sensor = getattr(self.config.TASK_CONFIG.SIMULATOR, f"{sensor_type}_SENSOR")
                for camera_id, orient in orient_dict.items():
                    camera_template = f"{sensor_type}{camera_id}"
                    camera_config = deepcopy(sensor)
                    camera_config.WIDTH = H
                    camera_config.HEIGHT = H
                    camera_config.ORIENTATION = orient
                    camera_config.UUID = camera_template.lower()
                    camera_config.HFOV = 90
                    sensor_uuids.append(camera_config.UUID)
                    setattr(self.config.TASK_CONFIG.SIMULATOR, camera_template, camera_config)
                    self.config.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS.append(camera_template)
        self.config.freeze()

        if self.config.EVAL.SAVE_RESULTS:
            fname = os.path.join(
                self.config.RESULTS_DIR,
                f"stats_ckpt_{checkpoint_index}_{self.config.TASK_CONFIG.DATASET.SPLIT}.json",
            )
            if os.path.exists(fname) and not os.path.isfile(self.config.EVAL.CKPT_PATH_DIR):
                print("skipping -- evaluation exists.")
                return
        self.envs = construct_envs(
            self.config, 
            get_env_class(self.config.ENV_NAME),
            episodes_allowed=self.traj[::5] if self.config.EVAL.fast_eval else self.traj,
            auto_reset_done=False, # unseen: 11006 
        )
        dataset_length = sum(self.envs.number_of_episodes)
        print('local rank:', self.local_rank, '|', 'dataset length:', dataset_length)

        obs_transforms = get_active_obs_transforms(self.config)
        observation_space = apply_obs_transforms_obs_space(
            self.envs.observation_spaces[0], obs_transforms
        )
        self._initialize_policy(
            self.config,
            load_from_ckpt=True,
            observation_space=observation_space,
            action_space=self.envs.action_spaces[0],
            load_optimizer=False,
        )
        self.policy.eval()
        self.waypoint_predictor.eval()

        if self.config.EVAL.EPISODE_COUNT == -1:
            eps_to_eval = sum(self.envs.number_of_episodes)
        else:
            eps_to_eval = min(self.config.EVAL.EPISODE_COUNT, sum(self.envs.number_of_episodes))
        self.stat_eps = {}
        self.pbar = tqdm.tqdm(total=eps_to_eval) if self.config.use_pbar else None

        while len(self.stat_eps) < eps_to_eval:
            self.rollout('eval')
        self.envs.close()

        if self.world_size > 1:
            distr.barrier()
        aggregated_states = {}
        num_episodes = len(self.stat_eps)
        for stat_key in next(iter(self.stat_eps.values())).keys():
            aggregated_states[stat_key] = (
                sum(v[stat_key] for v in self.stat_eps.values()) / num_episodes
            )
        total = torch.tensor(num_episodes).cuda()
        if self.world_size > 1:
            distr.reduce(total,dst=0)
        total = total.item()

        if self.world_size > 1:
            logger.info(f"rank {self.local_rank}'s {num_episodes}-episode results: {aggregated_states}")
            for k,v in aggregated_states.items():
                v = torch.tensor(v*num_episodes).cuda()
                cat_v = gather_list_and_concat(v,self.world_size)
                v = (sum(cat_v)/total).item()
                aggregated_states[k] = v
        
        split = self.config.TASK_CONFIG.DATASET.SPLIT
        fname = os.path.join(
            self.config.RESULTS_DIR,
            f"stats_ep_ckpt_{checkpoint_index}_{split}_r{self.local_rank}_w{self.world_size}.json",
        )
        with open(fname, "w") as f:
            json.dump(self.stat_eps, f, indent=2)

        if self.local_rank < 1:
            if self.config.EVAL.SAVE_RESULTS:
                fname = os.path.join(
                    self.config.RESULTS_DIR,
                    f"stats_ckpt_{checkpoint_index}_{split}.json",
                )
                with open(fname, "w") as f:
                    json.dump(aggregated_states, f, indent=2)

            logger.info(f"Episodes evaluated: {total}")
            checkpoint_num = checkpoint_index + 1
            for k, v in aggregated_states.items():
                logger.info(f"Average episode {k}: {v:.6f}")
                writer.add_scalar(f"eval_{k}/{split}", v, checkpoint_num)

    @torch.no_grad()
    def inference(self):
        checkpoint_path = self.config.INFERENCE.CKPT_PATH
        logger.info(f"checkpoint_path: {checkpoint_path}")
        self.config.defrost()
        self.config.IL.ckpt_to_load = checkpoint_path
        self.config.TASK_CONFIG.DATASET.SPLIT = self.config.INFERENCE.SPLIT
        self.config.TASK_CONFIG.DATASET.ROLES = ["guide"]
        self.config.TASK_CONFIG.DATASET.LANGUAGES = self.config.INFERENCE.LANGUAGES
        self.config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
        self.config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = -1
        self.config.TASK_CONFIG.TASK.MEASUREMENTS = ['POSITION_INFER']
        self.config.TASK_CONFIG.TASK.SENSORS = [s for s in self.config.TASK_CONFIG.TASK.SENSORS if "INSTRUCTION" in s]
        self.config.SIMULATOR_GPU_IDS = [self.config.SIMULATOR_GPU_IDS[self.config.local_rank]]
        # if choosing image
        resize_config = self.config.RL.POLICY.OBS_TRANSFORMS.RESIZER_PER_SENSOR.SIZES
        crop_config = self.config.RL.POLICY.OBS_TRANSFORMS.CENTER_CROPPER_PER_SENSOR.SENSOR_CROPS
        task_config = self.config.TASK_CONFIG
        camera_orientations = get_camera_orientations12()
        for sensor_type in ["RGB", "DEPTH"]:
            resizer_size = dict(resize_config)[sensor_type.lower()]
            cropper_size = dict(crop_config)[sensor_type.lower()]
            sensor = getattr(task_config.SIMULATOR, f"{sensor_type}_SENSOR")
            for action, orient in camera_orientations.items():
                camera_template = f"{sensor_type}_{action}"
                camera_config = deepcopy(sensor)
                camera_config.ORIENTATION = camera_orientations[action]
                camera_config.UUID = camera_template.lower()
                setattr(task_config.SIMULATOR, camera_template, camera_config)
                task_config.SIMULATOR.AGENT_0.SENSORS.append(camera_template)
                resize_config.append((camera_template.lower(), resizer_size))
                crop_config.append((camera_template.lower(), cropper_size))
        self.config.RL.POLICY.OBS_TRANSFORMS.RESIZER_PER_SENSOR.SIZES = resize_config
        self.config.RL.POLICY.OBS_TRANSFORMS.CENTER_CROPPER_PER_SENSOR.SENSOR_CROPS = crop_config
        self.config.TASK_CONFIG = task_config
        self.config.SENSORS = task_config.SIMULATOR.AGENT_0.SENSORS
        self.config.freeze()

        torch.cuda.set_device(self.device)
        self.world_size = self.config.GPU_NUMBERS
        self.local_rank = self.config.local_rank
        if self.world_size > 1:
            distr.init_process_group(backend='nccl', init_method='env://')
            self.device = self.config.TORCH_GPU_IDS[self.local_rank]
            torch.cuda.set_device(self.device)
            self.config.defrost()
            self.config.TORCH_GPU_ID = self.config.TORCH_GPU_IDS[self.local_rank]
            self.config.freeze()
        self.traj = self.collect_infer_traj()

        self.envs = construct_envs(
            self.config, 
            get_env_class(self.config.ENV_NAME),
            episodes_allowed=self.traj,
            auto_reset_done=False,
        )

        obs_transforms = get_active_obs_transforms(self.config)
        observation_space = apply_obs_transforms_obs_space(
            self.envs.observation_spaces[0], obs_transforms
        )
        self._initialize_policy(
            self.config,
            load_from_ckpt=True,
            observation_space=observation_space,
            action_space=self.envs.action_spaces[0],
            load_optimizer=False,
        )
        self.policy.eval()
        self.waypoint_predictor.eval()

        if self.config.INFERENCE.EPISODE_COUNT == -1:
            eps_to_infer = sum(self.envs.number_of_episodes)
        else:
            eps_to_infer = min(self.config.INFERENCE.EPISODE_COUNT, sum(self.envs.number_of_episodes))
        self.path_eps = defaultdict(list)
        self.inst_ids: Dict[str, int] = {}   # transfer submit format
        self.pbar = tqdm.tqdm(total=eps_to_infer)

        while len(self.path_eps) < eps_to_infer:
            self.rollout('infer')
        self.envs.close()

        if self.world_size > 1:
            aggregated_path_eps = [None for _ in range(self.world_size)]
            distr.all_gather_object(aggregated_path_eps, self.path_eps)
            tmp_eps_dict = {}
            for x in aggregated_path_eps:
                tmp_eps_dict.update(x)
            self.path_eps = tmp_eps_dict

            aggregated_inst_ids = [None for _ in range(self.world_size)]
            distr.all_gather_object(aggregated_inst_ids, self.inst_ids)
            tmp_inst_dict = {}
            for x in aggregated_inst_ids:
                tmp_inst_dict.update(x)
            self.inst_ids = tmp_inst_dict


        if self.config.MODEL.task_type == "r2r":
            with open(self.config.INFERENCE.PREDICTIONS_FILE, "w") as f:
                json.dump(self.path_eps, f, indent=2)
            logger.info(f"Predictions saved to: {self.config.INFERENCE.PREDICTIONS_FILE}")
        else:  # use 'rxr' format for rxr-habitat leaderboard
            preds = []
            for k,v in self.path_eps.items():
                # save only positions that changed
                path = [v[0]["position"]]
                for p in v[1:]:
                    if p["position"] != path[-1]: path.append(p["position"])
                preds.append({"instruction_id": self.inst_ids[k], "path": path})
            preds.sort(key=lambda x: x["instruction_id"])
            with jsonlines.open(self.config.INFERENCE.PREDICTIONS_FILE, mode="w") as writer:
                writer.write_all(preds)
            logger.info(f"Predictions saved to: {self.config.INFERENCE.PREDICTIONS_FILE}")

    def get_pos_ori(self):
        pos_ori = self.envs.call(['get_pos_ori']*self.envs.num_envs)
        pos = [x[0] for x in pos_ori]
        ori = [x[1] for x in pos_ori]
        return pos, ori

    def _record_rl_topo_transition(
        self,
        env_index,
        trajectory_id,
        stepk,
        action_index,
        old_log_prob,
        entropy,
        value_t,
        gmap_embeds,
        reward,
        done,
        gmap_vp_ids,
        gmap_masks,
        gmap_visited_masks,
        dist_before,
        dist_after,
        il_rl_kl=0.0,
        argmax_action_changed=False,
        forced_stop=False,
    ):
        graph_length = int(gmap_masks.sum().item())
        candidate_ids = [
            "STOP" if vp_id is None else vp_id
            for vp_id in gmap_vp_ids[:graph_length]
        ]
        valid_action_mask = (
            gmap_masks & gmap_visited_masks.logical_not()
        ).detach().cpu().tolist()[:graph_length]
        graph_embed_snapshot = gmap_embeds[:graph_length].detach().float().clone()
        selected_graph_id = candidate_ids[action_index]

        transition = {
            "environment_index": int(env_index),
            "trajectory_id": trajectory_id,
            "high_level_step": int(stepk),
            "action_index": int(action_index),
            "selected_graph_id": selected_graph_id,
            "old_log_prob": float(old_log_prob),
            "entropy": float(entropy),
            "value_t": float(value_t),
            "reward_t": float(reward),
            "done_t": bool(done),
            "reward": float(reward),
            "done": bool(done),
            # Exact replay snapshot.  The candidate order is the first
            # dimension of this tensor and is paired with valid_action_mask.
            "gmap_embeds": graph_embed_snapshot,
            "state_embed": graph_embed_snapshot[0].clone(),
            "candidate_ids": candidate_ids,
            "valid_action_mask": valid_action_mask,
            "graph_length": graph_length,
            "dist_before": float(dist_before),
            "dist_after": float(dist_after),
            "il_rl_kl": float(il_rl_kl),
            "argmax_action_changed": bool(argmax_action_changed),
            "policy_stop": bool(int(action_index) == 0),
            "forced_stop": bool(forced_stop),
        }
        self.rl_topo_transitions.append(transition)

        if (
            self.config.RL_TOPO.DEBUG_TRANSITIONS
            and self._rl_topo_debug_transition_count < 20
        ):
            print(
                f"Env {env_index} | Step {stepk}\n"
                f"Candidates: {candidate_ids}\n"
                f"Action index: {action_index}\n"
                f"Selected: {selected_graph_id}\n"
                f"log_prob: {old_log_prob:.6f}\n"
                f"distance_before: {dist_before:.6f}\n"
                f"distance_after: {dist_after:.6f}\n"
                f"reward: {reward:.6f}\n"
                f"done: {bool(done)}"
            )
            self._rl_topo_debug_transition_count += 1

        return len(self.rl_topo_transitions) - 1

    def _make_aaa_diagnostic(self, env_index, stepk, cur_pos, cur_ori,
                             wp_outputs, nav_inputs, nav_logits, action_index,
                             selected_target, dist_before,
                             policy_action_index=None,
                             graph_oracle_record=None,
                             gmap_embeds=None):
        """Build one pre-execution decision record.

        Scenario labels are deliberately geometric proxies. They are useful
        for the first diagnostic cycle and do not claim semantic perception.
        """
        episode = self.envs.current_episodes()[env_index]
        episode_id = str(episode.episode_id)
        angles = list(wp_outputs["cand_angles"][env_index])
        distances = list(wp_outputs["cand_distances"][env_index])
        scores = list(wp_outputs.get("cand_scores", [[]])[env_index])
        cand_count = len(angles)
        if cand_count:
            unit_x = [math.cos(float(a)) for a in angles]
            unit_y = [math.sin(float(a)) for a in angles]
            mean_x = sum(unit_x) / cand_count
            mean_y = sum(unit_y) / cand_count
            angular_dispersion = 1.0 - math.sqrt(mean_x * mean_x + mean_y * mean_y)
        else:
            angular_dispersion = 0.0
        if cand_count <= 2:
            geometry_proxy = "low_proposal_count"
        elif angular_dispersion >= 0.55:
            geometry_proxy = "dispersed_proposals"
        else:
            geometry_proxy = "concentrated_proposals"

        reference_distance = None
        try:
            reference_distance = self.envs.call_at(
                env_index, "current_episode_ref_distance"
            )
        except (TypeError, ValueError):
            reference_distance = None

        graph_ids = nav_inputs["gmap_vp_ids"][env_index]
        graph_scores = nav_logits[env_index].detach().float().cpu().tolist()
        graph_visited = nav_inputs["gmap_visited_masks"][env_index].detach().cpu().tolist()
        graph_valid = nav_inputs["gmap_masks"][env_index].detach().cpu().tolist()
        graph_candidates = []
        for idx, vp_id in enumerate(graph_ids):
            if idx >= len(graph_scores) or not math.isfinite(graph_scores[idx]):
                continue
            graph_candidates.append({
                "id": "STOP" if vp_id is None else vp_id,
                "index": idx,
                "logit": graph_scores[idx],
                "visited": bool(graph_visited[idx]) if idx < len(graph_visited) else None,
            })
        graph_index_by_id = {vp_id: idx for idx, vp_id in enumerate(graph_ids)}
        candidate_graph_mapping = []
        candidate_vps = list(getattr(self.gmaps[env_index], "last_candidate_vps", []))
        for candidate_index, (angle, distance) in enumerate(zip(angles, distances)):
            graph_id = candidate_vps[candidate_index] if candidate_index < len(candidate_vps) else None
            graph_index = graph_index_by_id.get(graph_id)
            logit = graph_scores[graph_index] if graph_index is not None and graph_index < len(graph_scores) else None
            candidate_graph_mapping.append({
                "candidate_index": candidate_index,
                "graph_id": graph_id,
                "graph_index": graph_index,
                "graph_logit": logit,
                "graph_valid": bool(graph_valid[graph_index]) if graph_index is not None and graph_index < len(graph_valid) else False,
                "graph_visited": bool(graph_visited[graph_index]) if graph_index is not None and graph_index < len(graph_visited) else None,
                "selected_by_policy": graph_id == selected_target.get("id"),
            })
            if (
                self.aaa_include_ranker_embeddings
                and graph_index is not None
                and gmap_embeds is not None
            ):
                candidate_graph_mapping[-1]["graph_embedding"] = (
                    gmap_embeds[env_index, graph_index]
                    .detach().float().cpu().tolist()
                )
        instruction = getattr(episode, "instruction", None)
        if isinstance(instruction, dict):
            instruction = instruction.get("instruction_text", instruction)
        else:
            instruction = getattr(instruction, "instruction_text", str(instruction))
        return {
            "episode_id": episode_id,
            "scene_id": str(episode.scene_id),
            "schema_version": 2,
            "privileged_analysis_only": ["distance_to_goal_before", "distance_to_goal_after", "distance_to_reference_path", "progress", "near_goal"],
            "near_goal": float(dist_before) <= 3.0,
            "high_level_step": int(stepk),
            "abstraction": self.action_abstraction,
            "pose": {"position": list(cur_pos[env_index]), "rotation": list(cur_ori[env_index])},
            "instruction": instruction,
            "waypoint_candidates": [
                {"angle": float(a), "distance": float(d),
                 "score": float(scores[k]) if k < len(scores) else None}
                for k, (a, d) in enumerate(zip(angles, distances))
            ],
            "waypoint_candidate_count": cand_count,
            "candidate_angular_dispersion": float(angular_dispersion),
            "proposal_distribution_label": geometry_proxy,
            "graph_candidates": graph_candidates,
            "candidate_graph_mapping": candidate_graph_mapping,
            "selected_action_index": int(action_index),
            "policy_action_index": (
                int(policy_action_index) if policy_action_index is not None else int(action_index)
            ),
            "graph_selection_oracle": graph_oracle_record,
            "selected_target": selected_target,
            "policy_entropy": float(torch.distributions.Categorical(
                logits=nav_logits[env_index]
            ).entropy().detach().cpu().item()),
            "distance_to_goal_before": float(dist_before),
            "distance_to_reference_path": (
                float(reference_distance) if reference_distance is not None else None
            ),
        }

    def _write_aaa_diagnostic(self, record):
        if not self.aaa_diagnostics_enabled:
            return
        if self.aaa_diagnostic_max_records >= 0 and self.aaa_logger.count >= self.aaa_diagnostic_max_records:
            return
        self.aaa_logger.write(record)

    def _probe_aaa_oracle(self, env_index, stepk, wp_outputs):
        """Probe all fixed waypoint abstractions at the current simulator state.

        ``cand_dist_to_goal`` restores the worker simulator after each probe,
        so every level sees the identical pose and predicted heatmap.  This is
        privileged analysis only; the selected oracle level never changes the
        action executed by the rollout.
        """
        if not self.aaa_oracle_enabled:
            return
        if (self.aaa_oracle_max_records >= 0 and
                self.aaa_oracle_logger.count >= self.aaa_oracle_max_records):
            return
        heatmap = wp_outputs.get("waypoint_heatmap_probs")
        if heatmap is None:
            return
        heatmap = heatmap[env_index].detach().cpu()
        before = float(self.envs.call_at(env_index, "current_dist_to_goal"))
        level_outcomes = {}
        for level in ACTION_ABSTRACTIONS:
            geometry = candidate_geometry_from_heatmap(heatmap, level)
            candidates = []
            for angle, distance, score in zip(
                    geometry["angles"], geometry["distances"], geometry["scores"]):
                after = float(self.envs.call_at(
                    env_index, "cand_dist_to_goal",
                    {"angle": angle, "forward": distance},
                ))
                candidates.append({
                    "angle": float(angle),
                    "distance": float(distance),
                    "score": float(score),
                    "distance_to_goal_after": after,
                    "progress": float(before - after),
                })
            level_outcomes[level] = candidates
        oracle_level, oracle_value = select_oracle_level(level_outcomes)
        best_by_level = {
            level: (max(items, key=lambda item: item["progress"])
                    if items else None)
            for level, items in level_outcomes.items()
        }
        episode = self.envs.current_episodes()[env_index]
        self.aaa_oracle_logger.write({
            "schema_version": 1,
            "privileged_analysis_only": True,
            "episode_id": str(episode.episode_id),
            "scene_id": str(episode.scene_id),
            "high_level_step": int(stepk),
            "fixed_execution_abstraction": self.action_abstraction,
            "distance_to_goal_before": before,
            "levels": level_outcomes,
            "best_by_level": best_by_level,
            "oracle_level": oracle_level,
            "oracle_best_progress": oracle_value,
            "fixed_level_best_progress": (
                best_by_level.get(self.action_abstraction, {}).get("progress")
                if best_by_level.get(self.action_abstraction) else None
            ),
        })

    def _graph_selection_oracle_action(self, env_index, stepk, wp_outputs,
                                       nav_inputs, policy_action_index, nav_logits):
        """Choose a current graph candidate by privileged one-step progress.

        This is an analysis-only intervention. It keeps the default waypoint
        generator and all graph construction unchanged, but replaces the
        learned graph ranking for one evaluation rollout. Goal distance is
        privileged and the resulting trajectory must never be treated as a
        deployable method.
        """
        gmap = self.gmaps[env_index]
        graph_ids = nav_inputs["gmap_vp_ids"][env_index]
        graph_valid = nav_inputs["gmap_masks"][env_index].detach().cpu().tolist()
        graph_visited = nav_inputs["gmap_visited_masks"][env_index].detach().cpu().tolist()
        graph_index_by_id = {vp_id: idx for idx, vp_id in enumerate(graph_ids)}
        angles = wp_outputs["cand_angles"][env_index]
        distances = wp_outputs["cand_distances"][env_index]
        current_distance = float(self.envs.call_at(env_index, "current_dist_to_goal"))
        outcomes = []
        by_graph_index = {}
        for candidate_index, (angle, distance) in enumerate(zip(angles, distances)):
            if candidate_index >= len(gmap.last_candidate_vps):
                continue
            graph_id = gmap.last_candidate_vps[candidate_index]
            graph_index = graph_index_by_id.get(graph_id)
            if graph_index is None or graph_index >= len(graph_valid):
                continue
            if not graph_valid[graph_index] or graph_visited[graph_index] or not str(graph_id).startswith("g"):
                continue
            after = float(self.envs.call_at(
                env_index, "cand_dist_to_goal",
                {"angle": angle, "forward": distance},
            ))
            outcome = {
                "candidate_index": int(candidate_index),
                "graph_id": graph_id,
                "graph_index": int(graph_index),
                "angle": float(angle),
                "distance": float(distance),
                "progress": float(current_distance - after),
            }
            outcomes.append(outcome)
            previous = by_graph_index.get(graph_index)
            if previous is None or outcome["progress"] > previous["progress"]:
                by_graph_index[graph_index] = outcome
        best_candidate_action = None
        if by_graph_index:
            best_candidate_action = max(
                by_graph_index,
                key=lambda index: by_graph_index[index]["progress"],
            )
        valid_nonstop = [idx for idx, vp in enumerate(graph_ids)
                         if idx > 0 and graph_valid[idx] and not graph_visited[idx]
                         and math.isfinite(float(nav_logits[env_index, idx].item()))]
        best_nonstop = max(valid_nonstop,
                           key=lambda idx: float(nav_logits[env_index, idx].item())) if valid_nonstop else 0
        oracle_action = controlled_graph_action(
            self.graph_selection_oracle_mode, int(policy_action_index),
            best_nonstop, best_candidate_action, current_distance,
        )
        record = {
            "schema_version": 1,
            "privileged_analysis_only": True,
            "intervention_mode": self.graph_selection_oracle_mode,
            "best_learned_nonstop_action": int(best_nonstop),
            "episode_id": str(self.envs.current_episodes()[env_index].episode_id),
            "high_level_step": int(stepk),
            "policy_action_index": int(policy_action_index),
            "oracle_action_index": int(oracle_action),
            "distance_to_goal_before": current_distance,
            "candidate_outcomes": outcomes,
            "oracle_selected_candidate": by_graph_index.get(oracle_action),
        }
        self.graph_selection_oracle_logger.write(record)
        return oracle_action, record

    def _link_rl_topo_next_values(self, trajectory_ids, values):
        """Attach V(s_{t+1}) to the preceding transition in each env slot."""
        for env_index, trajectory_id in enumerate(trajectory_ids):
            previous_index = self._rl_topo_last_transition.get(trajectory_id)
            if previous_index is None:
                continue
            previous = self.rl_topo_transitions[previous_index]
            if not previous["done_t"]:
                previous["next_value_t"] = float(values[env_index].detach().item())

    def _compute_rl_topo_gae(self):
        """Compute episode-safe GAE targets for the PPO replay snapshot.

        Transitions are interleaved by vectorized environment.  Grouping by a
        stable trajectory id prevents a paused environment slot from joining
        another episode's bootstrapping chain.
        """
        transitions = self.rl_topo_transitions
        if not transitions:
            self._rl_topo_gae_ready = False
            return

        gamma = float(self.config.RL_TOPO.GAMMA)
        gae_lambda = float(self.config.RL_TOPO.GAE_LAMBDA)
        grouped = defaultdict(list)
        for index, transition in enumerate(transitions):
            grouped[transition["trajectory_id"]].append(index)

        advantages = np.zeros(len(transitions), dtype=np.float32)
        returns = np.zeros(len(transitions), dtype=np.float32)
        missing_bootstrap_count = 0

        for indices in grouped.values():
            gae = 0.0
            for position in reversed(range(len(indices))):
                index = indices[position]
                transition = transitions[index]
                done = bool(transition["done_t"])

                if done:
                    next_value = 0.0
                else:
                    next_value = transition.get("next_value_t")
                    if next_value is None:
                        # The extra value-only pass should provide this value
                        # for every non-terminal tail. Keep an explicit
                        # diagnostic if an environment disappears before it
                        # can be bootstrapped.
                        if not done:
                            missing_bootstrap_count += 1
                        next_value = 0.0

                delta = (
                    float(transition["reward_t"])
                    + gamma * float(next_value) * (1.0 - float(done))
                    - float(transition["value_t"])
                )
                has_next_transition = position + 1 < len(indices)
                continuation = (1.0 - float(done)) * float(has_next_transition)
                gae = delta + gamma * gae_lambda * continuation * gae
                advantages[index] = gae
                returns[index] = gae + float(transition["value_t"])

        adv_mean = float(advantages.mean())
        adv_std = float(advantages.std())
        normalized_advantages = (advantages - adv_mean) / max(adv_std, 1e-8)
        for index, transition in enumerate(transitions):
            transition["advantage_t"] = float(normalized_advantages[index])
            transition["return_t"] = float(returns[index])
            transition["returns_t"] = float(returns[index])

        value_mean = float(np.mean([t["value_t"] for t in transitions]))
        return_mean = float(np.mean(returns))
        self.logs["ppo/value_mean"].append(value_mean)
        self.logs["ppo/return_mean"].append(return_mean)
        self.logs["ppo/advantage_mean"].append(float(normalized_advantages.mean()))
        self.logs["ppo/advantage_std"].append(float(normalized_advantages.std()))
        self.logs["ppo/mean_reward"].append(
            float(np.mean([t["reward_t"] for t in transitions]))
        )
        self.logs["ppo/mean_return"].append(return_mean)
        self.logs["ppo/missing_bootstrap_count"].append(
            float(missing_bootstrap_count)
        )
        self.logs["ppo/stop_selection_rate"].append(
            float(np.mean([t["policy_stop"] for t in transitions]))
        )
        self.logs["ppo/forced_stop_rate"].append(
            float(np.mean([t["forced_stop"] for t in transitions]))
        )
        self.logs["ppo/mean_high_level_actions"].append(
            float(np.mean([len(indices) for indices in grouped.values()]))
        )
        self.logs["ppo/il_rl_kl"].append(
            float(np.mean([t["il_rl_kl"] for t in transitions]))
        )
        self.logs["ppo/argmax_action_change_rate"].append(
            float(np.mean([t["argmax_action_changed"] for t in transitions]))
        )
        self._rl_topo_gae_ready = True
        if self.config.RL_TOPO.DEBUG:
            logger.info(
                "RL_TOPO GAE: value mean %.6f, return mean %.6f, "
                "advantage mean %.6f/std %.6f",
                value_mean,
                return_mean,
                float(normalized_advantages.mean()),
                float(normalized_advantages.std()),
            )

    def _compute_ppo_loss(self):
        """Recompute PPO on exact graph snapshots from one rollout.

        Only ETP's existing global_sap_head and the critic are trainable in
        this stage.  Replaying the saved embeddings and masks preserves the
        candidate ordering used to collect each old log probability.
        """
        if not getattr(self, "_rl_topo_gae_ready", False):
            return self._actor_head_module().proj.weight.sum() * 0.0

        transitions = self.rl_topo_transitions
        critic = self._critic_module()
        device = next(critic.parameters()).device

        replay_outputs = []
        for transition in transitions:
            graph_embeds = transition["gmap_embeds"].to(device)
            replay_outputs.append(
                self.policy.net(
                    mode="ppo_snapshot",
                    gmap_embeds=graph_embeds.unsqueeze(0),
                    residual_alpha=float(self.config.RL_TOPO.RESIDUAL_ALPHA),
                    valid_action_mask=torch.as_tensor(
                        transition["valid_action_mask"],
                        dtype=torch.bool,
                        device=device,
                    ).unsqueeze(0),
                )
            )

        values = torch.cat(
            [output["value"].reshape(-1) for output in replay_outputs]
        ).float()
        return_targets = torch.as_tensor(
            [transition["returns_t"] for transition in transitions],
            dtype=values.dtype,
            device=device,
        )
        advantages = torch.as_tensor(
            [transition["advantage_t"] for transition in transitions],
            dtype=values.dtype,
            device=device,
        )
        value_loss = F.mse_loss(values, return_targets)

        new_log_probs = []
        entropies = []
        old_log_probs = []
        for transition, output in zip(transitions, replay_outputs):
            logits = output["policy_logits"].squeeze(0)
            valid_mask = torch.as_tensor(
                transition["valid_action_mask"], dtype=torch.bool, device=device
            )
            action = torch.tensor(
                transition["action_index"], dtype=torch.long, device=device
            )
            masked_logits = logits.masked_fill(valid_mask.logical_not(), -float("inf"))
            log_probs = F.log_softmax(masked_logits.float(), dim=-1)
            probs = log_probs.exp()
            new_log_probs.append(log_probs[action])
            safe_log_probs = torch.where(
                torch.isfinite(log_probs), log_probs, torch.zeros_like(log_probs)
            )
            entropies.append(-(probs * safe_log_probs).sum())
            old_log_probs.append(transition["old_log_prob"])

        new_log_probs = torch.stack(new_log_probs)
        entropies = torch.stack(entropies)
        old_log_probs = torch.as_tensor(
            old_log_probs, dtype=new_log_probs.dtype, device=device
        )
        if self.config.RL_TOPO.DEBUG:
            logger.info(
                "RL_TOPO PPO tensors: old_lp_finite=%s new_lp_finite=%s "
                "ratio_finite=%s adv_finite=%s",
                bool(torch.isfinite(old_log_probs).all()),
                bool(torch.isfinite(new_log_probs).all()),
                bool(torch.isfinite(new_log_probs - old_log_probs).all()),
                bool(torch.isfinite(advantages).all()),
            )
        if self.config.RL_TOPO.DEBUG and (
            not torch.isfinite(new_log_probs).all()
            or not torch.isfinite(entropies).all()
            or not torch.isfinite(advantages).all()
        ):
            logger.warning(
                "RL_TOPO non-finite PPO inputs: new_log_prob=%s entropy=%s advantage=%s",
                bool(torch.isfinite(new_log_probs).all()),
                bool(torch.isfinite(entropies).all()),
                bool(torch.isfinite(advantages).all()),
            )
        ratios = torch.exp(new_log_probs - old_log_probs)
        clip_eps = float(self.config.RL_TOPO.PPO_CLIP)
        unclipped = ratios * advantages
        clipped = torch.clamp(ratios, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
        policy_loss = -torch.minimum(unclipped, clipped).mean()
        entropy = entropies.mean()
        total_loss = (
            policy_loss
            + float(self.config.RL_TOPO.VALUE_COEF) * value_loss
            - float(self.config.RL_TOPO.ENTROPY_COEF) * entropy
        )

        approx_kl = (old_log_probs - new_log_probs).mean().detach().item()
        clip_fraction = (
            (torch.abs(ratios.detach() - 1.0) > clip_eps).float().mean().item()
        )
        self.logs["ppo/policy_loss"].append(float(policy_loss.detach().item()))
        self.logs["ppo/value_loss"].append(float(value_loss.detach().item()))
        self.logs["ppo/entropy"].append(float(entropy.detach().item()))
        self.logs["ppo/approx_kl"].append(float(approx_kl))
        self.logs["ppo/clip_fraction"].append(float(clip_fraction))
        # These are pre-update diagnostics. The post-update values are logged
        # by _record_ppo_parameter_diagnostics after optimizer.step().
        self.logs["ppo/residual_l2_before"].append(
            float(self._residual_l2())
        )
        self.logs["ppo/il_head_l2_change_before"].append(
            float(self._il_head_l2_change())
        )
        if self.config.RL_TOPO.DEBUG:
            logger.info(
                "RL_TOPO PPO: policy_loss %.6f, value_loss %.6f, entropy %.6f, "
                "approx_kl %.6f, clip_fraction %.6f",
                float(policy_loss.detach().item()),
                float(value_loss.detach().item()),
                float(entropy.detach().item()),
                float(approx_kl),
                float(clip_fraction),
            )
        return total_loss

    def _residual_l2(self):
        return float(torch.sqrt(sum(
            param.detach().float().pow(2).sum()
            for param in self._actor_head_module().parameters()
        )).item())

    def _il_head_l2_change(self):
        return float(torch.sqrt(sum(
            (param.detach().float().cpu() - self._initial_il_actor_params[name]).pow(2).sum()
            for name, param in self._il_actor_head_module().named_parameters()
        )).item())

    def _record_ppo_parameter_diagnostics(self):
        self.logs["ppo/residual_l2"].append(self._residual_l2())
        self.logs["ppo/il_head_l2_change"].append(self._il_head_l2_change())

    def rollout(self, mode, ml_weight=None, sample_ratio=None):
        if self.graph_selection_oracle_enabled and mode != 'eval':
            raise ValueError('Privileged graph interventions are evaluation-only')
        if mode == 'train':
            feedback = 'sample'
        elif mode == 'eval' or mode == 'infer':
            feedback = 'argmax'
        else:
            raise NotImplementedError

        rl_topo_enabled = bool(self.config.RL_TOPO.ENABLED)
        collect_rl_topo = rl_topo_enabled and mode == 'train'
        if collect_rl_topo:
            self.rl_topo_transitions = []
            self._rl_topo_last_transition = {}
            self._rl_topo_rollout_id += 1

        self.envs.resume_all()
        observations = self.envs.reset()
        if collect_rl_topo:
            trajectory_ids = [
                f"{self.local_rank}:{self._rl_topo_rollout_id}:{slot}:"
                f"{episode.episode_id}"
                for slot, episode in enumerate(self.envs.current_episodes())
            ]
        else:
            trajectory_ids = []
        instr_max_len = self.config.IL.max_text_len # r2r 80, rxr 200
        instr_pad_id = 1 if self.config.MODEL.task_type == 'rxr' else 0
        observations = extract_instruction_tokens(observations, self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID,
                                                  max_length=instr_max_len, pad_id=instr_pad_id)
        batch = batch_obs(observations, self.device)
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)
        
        if mode == 'eval':
            env_to_pause = [i for i, ep in enumerate(self.envs.current_episodes()) 
                            if ep.episode_id in self.stat_eps]    
            self.envs, batch = self._pause_envs(self.envs, batch, env_to_pause)
            if self.envs.num_envs == 0: return
        if mode == 'infer':
            env_to_pause = [i for i, ep in enumerate(self.envs.current_episodes()) 
                            if ep.episode_id in self.path_eps]    
            self.envs, batch = self._pause_envs(self.envs, batch, env_to_pause)
            if self.envs.num_envs == 0: return
            curr_eps = self.envs.current_episodes()
            for i in range(self.envs.num_envs):
                if self.config.MODEL.task_type == 'rxr':
                    ep_id = curr_eps[i].episode_id
                    k = curr_eps[i].instruction.instruction_id
                    self.inst_ids[ep_id] = int(k)

        # encode instructions
        all_txt_ids = batch['instruction']
        all_txt_masks = (all_txt_ids != instr_pad_id)
        all_txt_embeds = self.policy.net(
            mode='language',
            txt_ids=all_txt_ids,
            txt_masks=all_txt_masks,
        )

        loss = 0.
        total_actions = 0.
        not_done_index = list(range(self.envs.num_envs))

        have_real_pos = (mode == 'train' or self.config.VIDEO_OPTION)
        ghost_aug = self.config.IL.ghost_aug if mode == 'train' else 0
        self.gmaps = [GraphMap(have_real_pos, 
                               self.config.IL.loc_noise, 
                               self.config.MODEL.merge_ghost,
                               ghost_aug) for _ in range(self.envs.num_envs)]
        prev_vp = [None] * self.envs.num_envs
        eval_action_stats = {}
        if mode == 'eval':
            eval_action_stats = {
                episode.episode_id: {
                    'high_level_steps': 0,
                    'policy_stop_count': 0,
                    'forced_stop_count': 0,
                    'forced_stop_max_len_count': 0,
                    'forced_stop_no_vp_count': 0,
                }
                for episode in self.envs.current_episodes()
            }

        # One extra value-only pass bootstraps a non-terminal rollout tail.
        for stepk in range(self.max_len + 1):
            total_actions += self.envs.num_envs
            txt_masks = all_txt_masks[not_done_index]
            txt_embeds = all_txt_embeds[not_done_index]
            
            # cand waypoint prediction
            wp_outputs = self.policy.net(
                mode = "waypoint",
                waypoint_predictor = self.waypoint_predictor,
                observations = batch,
                in_train = (mode == 'train' and self.config.IL.waypoint_aug),
                action_abstraction=self.action_abstraction,
                return_waypoint_heatmap=(
                    self.aaa_oracle_enabled
                    or bool(os.environ.get('ETPNAV_DENSE_CANDIDATE_CAPTURE'))
                ),
            )

            # The oracle is evaluated before graph updates and before the
            # actual action is stepped, ensuring a matched navigation state.
            if self.aaa_oracle_enabled:
                for i in range(self.envs.num_envs):
                    self._probe_aaa_oracle(i, stepk, wp_outputs)

            # pano encoder
            vp_inputs = self._vp_feature_variable(wp_outputs)
            vp_inputs.update({
                'mode': 'panorama',
            })
            pano_embeds, pano_masks = self.policy.net(**vp_inputs)
            avg_pano_embeds = torch.sum(pano_embeds * pano_masks.unsqueeze(2), 1) / \
                              torch.sum(pano_masks, 1, keepdim=True)

            # get vp_id, vp_pos of cur_node and cand_ndoe
            cur_pos, cur_ori = self.get_pos_ori()
            cur_vp, cand_vp, cand_pos = [], [], []
            for i in range(self.envs.num_envs):
                cur_vp_i, cand_vp_i, cand_pos_i = self.gmaps[i].identify_node(
                    cur_pos[i], cur_ori[i], wp_outputs['cand_angles'][i], wp_outputs['cand_distances'][i]
                )
                cur_vp.append(cur_vp_i)
                cand_vp.append(cand_vp_i)
                cand_pos.append(cand_pos_i)
            
            if mode == 'train' or self.config.VIDEO_OPTION:
                cand_real_pos = []
                for i in range(self.envs.num_envs):
                    cand_real_pos_i = [
                        self.envs.call_at(i, "get_cand_real_pos", {"angle": ang, "forward": dis})
                        for ang, dis in zip(wp_outputs['cand_angles'][i], wp_outputs['cand_distances'][i])
                    ]
                    cand_real_pos.append(cand_real_pos_i)
            else:
                cand_real_pos = [None] * self.envs.num_envs

            for i in range(self.envs.num_envs):
                cur_embeds = avg_pano_embeds[i]
                cand_embeds = pano_embeds[i][vp_inputs['nav_types'][i]==1]
                self.gmaps[i].update_graph(prev_vp[i], stepk+1,
                                           cur_vp[i], cur_pos[i], cur_embeds,
                                           cand_vp[i], cand_pos[i], cand_embeds,
                                           cand_real_pos[i])

            nav_inputs = self._nav_gmap_variable(cur_vp, cur_pos, cur_ori)
            nav_inputs.update({
                'mode': 'navigation',
                'txt_embeds': txt_embeds,
                'txt_masks': txt_masks,
                'rl_topo_debug': bool(
                    self.config.RL_TOPO.ENABLED and self.config.RL_TOPO.DEBUG
                ),
                'residual_alpha': float(self.config.RL_TOPO.RESIDUAL_ALPHA)
                if self.config.RL_TOPO.ENABLED else 0.0,
                'action_abstraction': self.action_abstraction,
            })
            no_vp_left = nav_inputs.pop('no_vp_left')
            nav_outs = self.policy.net(**nav_inputs)
            il_logits = nav_outs['global_logits']
            nav_logits = (
                nav_outs['policy_logits']
                if rl_topo_enabled else il_logits
            )
            nav_values = nav_outs['value']
            nav_probs = F.softmax(nav_logits, 1)
            il_rl_kl = None
            argmax_action_changed = None
            if rl_topo_enabled:
                il_dist = torch.distributions.Categorical(logits=il_logits)
                rl_dist = torch.distributions.Categorical(logits=nav_logits)
                il_rl_kl = torch.distributions.kl_divergence(il_dist, rl_dist)
                argmax_action_changed = il_logits.argmax(dim=-1) != nav_logits.argmax(dim=-1)
            if collect_rl_topo:
                self._link_rl_topo_next_values(trajectory_ids, nav_values)
            if collect_rl_topo and stepk == self.max_len:
                break
            for i, gmap in enumerate(self.gmaps):
                gmap.node_stop_scores[cur_vp[i]] = nav_probs[i, 0].data.item()

            # random sample demo
            # logits = torch.randn(nav_inputs['gmap_masks'].shape).cuda()
            # logits.masked_fill_(~nav_inputs['gmap_masks'], -float('inf'))
            # logits.masked_fill_(nav_inputs['gmap_visited_masks'], -float('inf'))

            if mode == 'train' or self.config.VIDEO_OPTION:
                teacher_actions = self._teacher_action_new(nav_inputs['gmap_vp_ids'], no_vp_left)
            if mode == 'train' and not rl_topo_enabled:
                loss += F.cross_entropy(nav_logits, teacher_actions, reduction='sum', ignore_index=-100)

            # determine action
            old_log_probs = None
            entropies = None
            if feedback == 'sample':
                if rl_topo_enabled:
                    dist = torch.distributions.Categorical(logits=nav_logits)
                    a_t = dist.sample()
                    old_log_probs = dist.log_prob(a_t)
                    entropies = dist.entropy()
                else:
                    c = torch.distributions.Categorical(nav_probs)
                    a_t = c.sample().detach()
                    a_t = torch.where(torch.rand_like(a_t, dtype=torch.float)<=sample_ratio, teacher_actions, a_t)
            elif feedback == 'argmax':
                a_t = nav_logits.argmax(dim=-1)
            else:
                raise NotImplementedError
            cpu_a_t = a_t.cpu().numpy()
            policy_cpu_a_t = cpu_a_t.copy()
            graph_oracle_records = [None] * self.envs.num_envs
            if self.graph_selection_oracle_enabled and mode == 'eval':
                for i in range(self.envs.num_envs):
                    cpu_a_t[i], graph_oracle_records[i] = self._graph_selection_oracle_action(
                        i, stepk, wp_outputs, nav_inputs, policy_cpu_a_t[i], nav_logits
                    )

            # make equiv action
            env_actions = []
            diagnostic_records = []
            use_tryout = (self.config.IL.tryout and not self.config.TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING)
            graph_option_capture = getattr(self, '_graph_option_capture', None)
            if mode == 'eval' and graph_option_capture is not None:
                graph_option_capture.prepare(
                    self, stepk, cur_vp, cur_pos, nav_inputs, nav_logits,
                    cpu_a_t, policy_cpu_a_t, no_vp_left, nav_outs.get('gmap_embeds'),
                    waypoint_heatmap=wp_outputs.get('waypoint_heatmap_probs'))
            for i, gmap in enumerate(self.gmaps):
                policy_stop = int(cpu_a_t[i]) == 0
                forced_by_max_len = (not policy_stop) and stepk == self.max_len - 1
                forced_by_no_vp = (not policy_stop) and bool(no_vp_left[i])
                forced_stop = forced_by_max_len or forced_by_no_vp
                if mode == 'eval':
                    episode_id = self.envs.current_episodes()[i].episode_id
                    action_stats = eval_action_stats[episode_id]
                    action_stats['high_level_steps'] += 1
                    action_stats['policy_stop_count'] += int(policy_stop)
                    action_stats['forced_stop_count'] += int(forced_stop)
                    action_stats['forced_stop_max_len_count'] += int(forced_by_max_len)
                    action_stats['forced_stop_no_vp_count'] += int(forced_by_no_vp)

                if policy_stop or forced_stop:
                    # stop at node with max stop_prob
                    vp_stop_scores = [(vp, stop_score) for vp, stop_score in gmap.node_stop_scores.items()]
                    stop_scores = [s[1] for s in vp_stop_scores]
                    stop_vp = vp_stop_scores[np.argmax(stop_scores)][0]
                    stop_pos = gmap.node_pos[stop_vp]
                    selected_target = {
                        "type": "stop_node",
                        "id": stop_vp,
                        "position": list(stop_pos),
                    }
                    if self.config.IL.back_algo == 'control':
                        back_path = [(vp, gmap.node_pos[vp]) for vp in gmap.shortest_path[cur_vp[i]][stop_vp]]
                        back_path = back_path[1:]
                    else:
                        back_path = None
                    vis_info = {
                            'nodes': list(gmap.node_pos.values()),
                            'ghosts': list(gmap.ghost_aug_pos.values()),
                            'predict_ghost': stop_pos,
                    }
                    env_actions.append(
                        {
                            'action': {
                                'act': 0,
                                'cur_vp': cur_vp[i],
                                'stop_vp': stop_vp, 'stop_pos': stop_pos,
                                'back_path': back_path,
                                'tryout': use_tryout,
                            },
                            'vis_info': vis_info,
                        }
                    )
                else:
                    ghost_vp = nav_inputs['gmap_vp_ids'][i][cpu_a_t[i]]
                    ghost_pos = gmap.ghost_aug_pos[ghost_vp]
                    selected_target = {
                        "type": "ghost",
                        "id": ghost_vp,
                        "position": list(ghost_pos),
                    }
                    _, front_vp = gmap.front_to_ghost_dist(ghost_vp)
                    front_pos = gmap.node_pos[front_vp]
                    if self.config.VIDEO_OPTION:
                        teacher_action_cpu = teacher_actions[i].cpu().item()
                        if teacher_action_cpu in [0, -100]:
                            teacher_ghost = None
                        else:
                            teacher_ghost = gmap.ghost_aug_pos[nav_inputs['gmap_vp_ids'][i][teacher_action_cpu]]
                        vis_info = {
                            'nodes': list(gmap.node_pos.values()),
                            'ghosts': list(gmap.ghost_aug_pos.values()),
                            'predict_ghost': ghost_pos,
                            'teacher_ghost': teacher_ghost,
                        }
                    else:
                        vis_info = None
                    # teleport to front, then forward to ghost
                    if self.config.IL.back_algo == 'control':
                        back_path = [(vp, gmap.node_pos[vp]) for vp in gmap.shortest_path[cur_vp[i]][front_vp]]
                        back_path = back_path[1:]
                    else:
                        back_path = None
                    env_actions.append(
                        {
                            'action': {
                                'act': 4,
                                'cur_vp': cur_vp[i],
                                'front_vp': front_vp, 'front_pos': front_pos,
                                'ghost_vp': ghost_vp, 'ghost_pos': ghost_pos,
                                'back_path': back_path,
                                'tryout': use_tryout,
                            },
                            'vis_info': vis_info,
                        }
                    )
                    prev_vp[i] = front_vp
                    if self.config.MODEL.consume_ghost:
                        gmap.delete_ghost(ghost_vp)

                if self.aaa_diagnostics_enabled:
                    diagnostic_records.append(self._make_aaa_diagnostic(
                        env_index=i,
                        stepk=stepk,
                        cur_pos=cur_pos,
                        cur_ori=cur_ori,
                        wp_outputs=wp_outputs,
                        nav_inputs=nav_inputs,
                        nav_logits=nav_logits,
                        action_index=cpu_a_t[i],
                        policy_action_index=policy_cpu_a_t[i],
                        graph_oracle_record=graph_oracle_records[i],
                        gmap_embeds=nav_outs.get('gmap_embeds'),
                        selected_target=selected_target,
                        dist_before=self.envs.call_at(i, "current_dist_to_goal"),
                    ))

                if mode == 'eval' and graph_option_capture is not None:
                    graph_option_capture.commit(env_actions)
                    for i in range(self.envs.num_envs):
                        prev_vp[i] = graph_option_capture.override_prev_vp(i, prev_vp[i])
                outputs = self.envs.step(env_actions)
            observations, _, dones, infos = [list(x) for x in zip(*outputs)]

            if self.aaa_diagnostics_enabled:
                for i, record in enumerate(diagnostic_records):
                    after = self.envs.call_at(i, "current_dist_to_goal")
                    record["distance_to_goal_after"] = float(after)
                    record["progress"] = float(
                        record["distance_to_goal_before"] - after
                    )
                    record["done"] = bool(dones[i])
                    record["selected_action_succeeded_progress"] = bool(
                        record["progress"] > 0.0
                    )
                    collision_info = infos[i].get("collisions") if isinstance(infos[i], dict) else None
                    if isinstance(collision_info, dict):
                        record["collision_count_after"] = collision_info.get("count")
                    self._write_aaa_diagnostic(record)

            if collect_rl_topo:
                # Match RLTrainer's existing evaluation criterion exactly.
                success_distance = 3.0
                for i in range(self.envs.num_envs):
                    high_level_info = infos[i]["rl_high_level"]
                    dist_before = float(high_level_info["dist_before"])
                    dist_after = float(high_level_info["dist_after"])
                    action_index = int(cpu_a_t[i])
                    reward = float(self.config.RL_TOPO.PROGRESS_WEIGHT) * (
                        dist_before - dist_after
                    )
                    if dones[i] and dist_after <= success_distance:
                        reward += float(self.config.RL_TOPO.SUCCESS_REWARD)
                    if action_index == 0 and dist_after > success_distance:
                        reward -= float(self.config.RL_TOPO.WRONG_STOP_PENALTY)

                    self._record_rl_topo_transition(
                        env_index=i,
                        trajectory_id=trajectory_ids[i],
                        stepk=stepk,
                        action_index=action_index,
                        old_log_prob=old_log_probs[i].detach().item(),
                        entropy=entropies[i].detach().item(),
                        value_t=nav_values[i].detach().item(),
                        gmap_embeds=nav_outs['gmap_embeds'][i],
                        reward=reward,
                        done=dones[i],
                        gmap_vp_ids=nav_inputs['gmap_vp_ids'][i],
                        gmap_masks=nav_inputs['gmap_masks'][i],
                        gmap_visited_masks=nav_inputs['gmap_visited_masks'][i],
                        dist_before=dist_before,
                        dist_after=dist_after,
                        il_rl_kl=(
                            il_rl_kl[i].detach().item()
                            if il_rl_kl is not None else 0.0
                        ),
                        argmax_action_changed=(
                            argmax_action_changed[i].detach().item()
                            if argmax_action_changed is not None else False
                        ),
                        forced_stop=(
                            int(cpu_a_t[i]) != 0
                            and (stepk == self.max_len - 1 or no_vp_left[i])
                        ),
                    )
                    current_index = len(self.rl_topo_transitions) - 1
                    if dones[i]:
                        self.rl_topo_transitions[current_index]["next_value_t"] = 0.0
                        self._rl_topo_last_transition.pop(trajectory_ids[i], None)
                    else:
                        self._rl_topo_last_transition[trajectory_ids[i]] = current_index

            # calculate metric
            if mode == 'eval':
                curr_eps = self.envs.current_episodes()
                for i in range(self.envs.num_envs):
                    if not dones[i]:
                        continue
                    info = infos[i]
                    ep_id = curr_eps[i].episode_id
                    gt_path = np.array(self.gt_data[str(ep_id)]['locations']).astype(np.float)
                    pred_path = np.array(info['position']['position'])
                    distances = np.array(info['position']['distance'])
                    metric = {}
                    metric['steps_taken'] = info['steps_taken']
                    metric['distance_to_goal'] = distances[-1]
                    metric['success'] = 1. if distances[-1] <= 3. else 0.
                    metric['oracle_success'] = 1. if (distances <= 3.).any() else 0.
                    metric['path_length'] = float(np.linalg.norm(pred_path[1:] - pred_path[:-1],axis=1).sum())
                    metric['collisions'] = info['collisions']['count'] / len(pred_path)
                    gt_length = distances[0]
                    metric['spl'] = metric['success'] * gt_length / max(gt_length, metric['path_length'])
                    dtw_distance = fastdtw(pred_path, gt_path, dist=NDTW.euclidean_distance)[0]
                    metric['ndtw'] = np.exp(-dtw_distance / (len(gt_path) * 3.))
                    metric['sdtw'] = metric['ndtw'] * metric['success']
                    metric['ghost_cnt'] = self.gmaps[i].ghost_cnt
                    action_stats = eval_action_stats[ep_id]
                    high_level_steps = max(1, action_stats['high_level_steps'])
                    metric['high_level_steps'] = action_stats['high_level_steps']
                    metric['policy_stop_count'] = action_stats['policy_stop_count']
                    metric['forced_stop_count'] = action_stats['forced_stop_count']
                    metric['forced_stop_max_len_count'] = action_stats['forced_stop_max_len_count']
                    metric['forced_stop_no_vp_count'] = action_stats['forced_stop_no_vp_count']
                    metric['policy_stop_rate'] = (
                        action_stats['policy_stop_count'] / high_level_steps
                    )
                    metric['forced_stop_rate'] = (
                        action_stats['forced_stop_count'] / high_level_steps
                    )
                    metric['actual_stop_rate'] = (
                        (action_stats['policy_stop_count'] + action_stats['forced_stop_count'])
                        / high_level_steps
                    )
                    self.stat_eps[ep_id] = metric
                    self.pbar.update()

            # record path
            if mode == 'infer':
                curr_eps = self.envs.current_episodes()
                for i in range(self.envs.num_envs):
                    if not dones[i]:
                        continue
                    info = infos[i]
                    ep_id = curr_eps[i].episode_id
                    self.path_eps[ep_id] = [
                        {
                            'position': info['position_infer']['position'][0],
                            'heading': info['position_infer']['heading'][0],
                            'stop': False
                        }
                    ]
                    for p, h in zip(info['position_infer']['position'][1:], info['position_infer']['heading'][1:]):
                        if p != self.path_eps[ep_id][-1]['position']:
                            self.path_eps[ep_id].append({
                                'position': p,
                                'heading': h,
                                'stop': False
                            })
                    self.path_eps[ep_id] = self.path_eps[ep_id][:500]
                    self.path_eps[ep_id][-1]['stop'] = True
                    self.pbar.update()

            # pause env
            if sum(dones) > 0:
                for i in reversed(list(range(self.envs.num_envs))):
                    if dones[i]:
                        not_done_index.pop(i)
                        self.envs.pause_at(i)
                        observations.pop(i)
                        # graph stop
                        self.gmaps.pop(i)
                        prev_vp.pop(i)
                        if collect_rl_topo:
                            trajectory_ids.pop(i)

            if self.envs.num_envs == 0:
                break

            # obs for next step
            observations = extract_instruction_tokens(observations,self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID)
            batch = batch_obs(observations, self.device)
            batch = apply_obs_transforms_batch(batch, self.obs_transforms)

        if mode == 'train' and collect_rl_topo:
            # Any transition without a following state belongs to a rollout
            # segment boundary.  Its GAE recursion is therefore terminated
            # without crossing into the next vectorized episode/rollout.
            for transition in self.rl_topo_transitions:
                transition.setdefault("next_value_t", None)
            self._compute_rl_topo_gae()
        elif mode == 'train':
            loss = ml_weight * loss / total_actions
            self.loss += loss
            self.logs['IL_loss'].append(loss.item())
