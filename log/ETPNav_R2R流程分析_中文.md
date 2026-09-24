# ETPNav R2R 训练与导航流程分析

> 目的：为后续在 ETPNav 上增加“高层拓扑候选节点强化学习规划器”提供代码级参考。
>
> 当前结论基于仓库源代码追踪得到。本次只新增本说明文档，没有修改源代码。
>
> 重要约定：RL 动作是选择一个全局拓扑候选节点（ghost node），不是 TURN/FORWARD 等低层动作。选中后仍由现有低层控制器执行到该节点。

## 1. 运行入口与完整调用链

`run_r2r/main.bash` 启动：

```text
python -m torch.distributed.launch ... run.py ...
```

随后调用链为：

```text
run.py:main()
  -> run_exp()
  -> get_config()
  -> baseline_registry.get_trainer("SS-ETP")
  -> RLTrainer.train()
  -> RLTrainer._set_config()
  -> RLTrainer._init_envs()
  -> RLTrainer._initialize_policy()
  -> RLTrainer._train_interval()
  -> RLTrainer.rollout("train")
```

评估路径是：

```text
RLTrainer.eval()
  -> RLTrainer._eval_checkpoint()
  -> RLTrainer.rollout("eval")
```

推理路径是：

```text
RLTrainer.inference()
  -> RLTrainer.rollout("infer")
```

训练、评估、推理都通过 `rollout()` 执行 waypoint 预测、GraphMap 更新、全局动作选择和环境控制。

## 2. Waypoint 候选生成

主要函数：

- `vlnce_baselines/models/Policy_ViewSelection_ETP.py::ETP.forward()` 的 `mode == "waypoint"` 分支
- `vlnce_baselines/waypoint_pred/TRM_net.py::BinaryDistPredictor_TRM.forward()`
- `vlnce_baselines/waypoint_pred/utils.py::nms()`

输入是当前环境观测中的 RGB 和 depth 全方向图像。代码固定使用 12 个方向：

```text
NUM_ANGLES = 120
NUM_IMGS = 12
NUM_CLASSES = 12
```

由此得到：

```text
RGB CLIP 特征：      [12B, 512]
Depth 特征：         [12B, 128, 4, 4]
Waypoint heatmap：   [B, 120, 12]
```

`BinaryDistPredictor_TRM` 内部将输出 reshape 成 `[B, 120, 12]`。120 个角度 bin 对应每 3 度一个方向，12 个类别对应不同距离。

`nms(..., max_predictions=5)` 最多保留 5 个 waypoint 候选。每个候选包含：

```text
角度：       一个弧度值
距离：       (distance_index + 1) * 0.25
角度特征：   [4]
RGB 特征：   [512]
Depth 特征： [128]
```

训练时，如果 `in_train=True`，NMS 得到的区域还会从 heatmap 概率中采样距离和角度；评估/推理时直接使用 NMS 非零位置。

`RLTrainer._vp_feature_variable()` 将候选视图和非候选全景视图拼接。对于环境 `i`：

```text
V_i = K_i + 12 - U_i
```

其中 `K_i` 是候选数量，`U_i` 是候选使用的不同 panorama 图像索引数量。因此视图张量在 batch 后大致为：

```text
RGB 特征：   [B, Vmax, 512]
Depth 特征： [B, Vmax, 128]
位置特征：   [B, Vmax, 4]
nav_types：  [B, Vmax]
```

全景编码器将其编码为：

```text
pano_embeds： [B, Vmax, 768]
pano_masks：  [B, Vmax]
```

当前节点特征是有效全景特征的平均值 `[B, 768]`。候选 ghost 的局部特征是 `[K_i, 768]`。

注意：`Policy_ViewSelection_ETP.py` 中有旧注释写成 2048 维 RGB，但实际使用的 `CLIPEncoder` 输出是 512 维，且 ETP 配置中的 `image_feat_size` 也是 512。

## 3. GraphMap 的创建与更新

GraphMap 类位于：

```text
vlnce_baselines/models/graph_utils.py::GraphMap
```

核心函数：

- `GraphMap.identify_node()`：生成当前 node ID、临时候选 ID 和估计位置
- `GraphMap.update_graph()`：写入 node、ghost、边和最短路
- `GraphMap.get_node_embeds()`：取得 node/ghost 的 768 维特征
- `GraphMap.get_pos_fts()`：生成每个图节点的 7 维空间特征
- `GraphMap.front_to_ghost_dist()`：确定 ghost 对应的最近 front node

每个环境开始 rollout 时创建一个 GraphMap：

```python
self.gmaps = [
    GraphMap(have_real_pos, loc_noise, merge_ghost, ghost_aug)
    for _ in range(self.envs.num_envs)
]
```

每个拓扑决策周期中，调用顺序是：

```text
waypoint prediction
  -> pano encoding
  -> get_pos_ori()
  -> GraphMap.identify_node()
  -> get_cand_real_pos()       # 训练或视频模式
  -> GraphMap.update_graph()
```

`update_graph()` 的行为：

1. 加入当前 node。
2. 如果存在 `prev_vp`，连接前一 node 和当前 node。
3. 候选与已有 node 在 `loc_noise` 范围内重合时，连接到已有 node。
4. 其他候选创建或合并为 ghost。
5. ghost 特征按多次观测累加后求平均。
6. 保存 ghost 的 front node 列表。
7. 重新计算 NetworkX 的所有节点最短路径和最短距离。

主要字典：

```text
node_pos, node_embeds, node_stepId
ghost_pos, ghost_mean_pos, ghost_embeds, ghost_fronts
ghost_real_pos
shortest_path, shortest_dist
node_stop_scores
```

ghost ID 形式为 `g0`、`g1` 等；已访问 node ID 通常是 `"0"`、`"1"` 等。

## 4. 全局图张量与 visited mask

主要函数：

```text
vlnce_baselines/ss_trainer_ETP.py::RLTrainer._nav_gmap_variable()
```

每个环境的 graph 节点顺序严格是：

```text
[None, 已访问 node..., ghost...]
```

其中位置 0 是 stop token。

如果环境 `i` 有 `N_i` 个 node 和 `G_i` 个 ghost，则：

```text
L_i = 1 + N_i + G_i
```

单环境张量为：

```text
gmap_vp_ids：        Python list，长度 L_i
gmap_step_ids：      [L_i]
gmap_img_fts：       [L_i, 768]
gmap_pos_fts：       [L_i, 7]
gmap_pair_dists：    [L_i, L_i]
gmap_visited_masks： [L_i]
```

batch padding 后：

```text
gmap_step_ids：      [B, Lmax]
gmap_img_fts：       [B, Lmax, 768]
gmap_pos_fts：       [B, Lmax, 7]
gmap_masks：         [B, Lmax]
gmap_visited_masks： [B, Lmax]
gmap_pair_dists：    [B, Lmax, Lmax]
```

`gmap_visited_masks` 的构造是：

```text
[0] + [1] * 已访问 node 数量 + [0] * ghost 数量
```

因此：

- stop token 不被 visited mask 屏蔽；
- 已访问 node 会被屏蔽；
- ghost 可以作为全局拓扑动作；
- padding 由 `gmap_masks` 屏蔽。

`gmap_pos_fts` 的 7 个维度是：

```text
sin(relative_heading)
cos(relative_heading)
sin(relative_elevation)
cos(relative_elevation)
line_distance / MAX_DIST
shortest_graph_distance / MAX_DIST
shortest_graph_step / MAX_STEP
```

`MAX_DIST = 30`，`MAX_STEP = 10`。

指令编码张量为：

```text
txt_ids：    [B, T]
txt_masks：  [B, T]
txt_embeds： [B, T, 768]
```

R2R 中 `T` 的上限来自 `IL.max_text_len = 80`。

## 5. Navigation global logits

调用链：

```text
ETP.forward(mode="navigation")
  -> GlocalTextPathNavCMT.forward_navigation()
  -> GlobalMapEncoder 的 cross-modal encoder
  -> global_sap_head
```

`forward_navigation()` 首先将：

```text
gmap_img_fts
+ graph step embedding
+ graph positional embedding
```

相加，随后使用 pairwise graph distance 作为 graph spatial relation，运行跨模态编码器。

输出形状：

```text
gmap_embeds：   [B, Lmax, 768]
global_logits： [B, Lmax]
```

代码随后执行：

```python
global_logits.masked_fill_(gmap_visited_masks, -inf)
global_logits.masked_fill_(gmap_masks.logical_not(), -inf)
```

因此 `global_logits[:, 0]` 是 stop logit，`global_logits[:, 1:]` 中未被屏蔽的位置对应全局 ghost/node 候选。

## 6. 全局动作采样与 argmax

相关代码位于：

```text
vlnce_baselines/ss_trainer_ETP.py::RLTrainer.rollout()
```

先计算：

```python
nav_probs = F.softmax(nav_logits, 1)
```

训练模式：

```python
c = torch.distributions.Categorical(nav_probs)
a_t = c.sample().detach()
a_t = torch.where(
    torch.rand_like(a_t, dtype=torch.float) <= sample_ratio,
    teacher_actions,
    a_t,
)
```

评估和推理模式：

```python
a_t = nav_logits.argmax(dim=-1)
```

这里的 `a_t` 是全局 graph 序列中的索引，不是 Habitat 的 TURN/FORWARD 动作。

## 7. Teacher action 与 imitation loss

Teacher action 函数：

```text
RLTrainer._teacher_action_new()
```

规则：

```text
当前 goal 距离 < 1.5：      action = 0
没有 ghost：                 action = -100
expert_policy = spl：        选真实 ghost 位置到 goal 距离最小者
expert_policy = ndtw：       通过 ghost_dist_to_ref() 选择 ghost
```

`ghost_real_pos` 中的位置来自 `get_cand_real_pos()`，该函数会临时执行候选方向的 forward，再恢复 simulator 状态。

IL loss：

```python
loss += F.cross_entropy(
    nav_logits,
    teacher_actions,
    reduction="sum",
    ignore_index=-100,
)
```

最终：

```text
loss = ml_weight * loss / total_actions
self.loss += loss
```

当前模型是以全局 graph action 为监督目标的 imitation learning，而不是 PPO。

## 8. 选中 ghost 后如何转换为低层导航

在 `rollout()` 中：

- action `0`：返回 stop node；
- 其他 action：通过 `gmap_vp_ids[i][a_t[i]]` 得到 ghost ID。

选中 ghost 后：

```text
ghost_vp
  -> ghost_pos
  -> front_to_ghost_dist()
  -> front_vp
  -> front_pos
```

然后构造：

```python
{
    "action": {
        "act": 4,
        "cur_vp": cur_vp,
        "front_vp": front_vp,
        "front_pos": front_pos,
        "ghost_vp": ghost_vp,
        "ghost_pos": ghost_pos,
        "back_path": back_path,
        "tryout": tryout,
    }
}
```

实际执行函数是：

```text
vlnce_baselines/common/environments.py::VLNCEDaggerEnv.step()
```

`act == 4` 时：

1. 如果 `back_algo == teleport`，直接 teleport 到 `front_pos`。
2. 如果 `back_algo == control`，沿 graph path 使用 `multi_step_control()` 返回。
3. 使用 `single_step_control()` 朝 `ghost_pos` 转向并前进。
4. 每个低层 simulator step 通过 `wrap_act()` 执行。

`single_step_control()` 使用：

```text
TURN_LEFT / TURN_RIGHT
MOVE_FORWARD
sim.previous_step_collided
```

注意：`habitat_extensions/nav.py` 中的 `MoveHighToLowAction`、`MoveHighToLowActionEval` 和 `MoveHighToLowActionInference` 是 Habitat 注册动作。当前 ETP rollout 的 `act: 4` 已经被 `VLNCEDaggerEnv.step()` 拦截，因此这条 ETP 路径并不直接调用 `nav.py` 中的动作类。

## 9. Habitat 状态、奖励信号和指标来源

### Episode done

```text
VLNCEDaggerEnv.get_done()
  -> self._env.episode_over
```

`rollout()` 从 `envs.step()` 得到 `dones`。

### Distance-to-goal

可以直接调用：

```text
VLNCEDaggerEnv.current_dist_to_goal()
```

它通过 simulator 的 geodesic distance 计算当前 agent 到 episode goal 的距离。

也可以读取 `Position` measure：

```text
info["position"]["distance"]
```

### Success

当前 ETP evaluator 手工计算：

```python
success = 1.0 if distances[-1] <= 3.0 else 0.0
```

任务配置中的成功距离是 3.0 米。

### Collision

低层控制器可直接读取：

```text
sim.previous_step_collided
```

它是每个 simulator 子步的布尔值。`COLLISIONS` 是 Habitat measure，启用后可以通过 `info` 获取累计碰撞统计。

### Travelled distance

`Position` measure 保存位置序列：

```text
info["position"]["position"]
```

ETP evaluator 使用相邻位置的欧氏距离之和作为 path length。

仓库还定义了独立的 `PathLength` measure，用 `_previous_position` 累积欧氏位移。

### 当前配置风险

`run_r2r/r2r_vlnce.yaml` 的 `TASK.MEASUREMENTS` 当前基本是注释状态：

```yaml
MEASUREMENTS: []
```

因此 `position`、`steps_taken`、`collisions` 等字段不一定会自动出现在 `info` 中。后续 PPO 若需要这些信号，应显式启用对应 measures，或者在 `VLNCEDaggerEnv.step()` 中聚合并返回。

## 10. 未来高层 PPO 的最小修改范围

如果继续使用现有 ETP controller，最小的算法修改范围是：

1. `vlnce_baselines/ss_trainer_ETP.py`
   - 在 `rollout()` 中保存高层 action、log probability、value、reward、done 和 action mask。
   - 在 `envs.step()` 前后形成一个高层 semi-MDP transition。
   - 在 `_train_interval()` 中增加 GAE、PPO clipped objective、value loss 和 entropy loss。

2. `vlnce_baselines/models/Policy_ViewSelection_ETP.py`
   - 增加 value head，或复用其中目前未使用的 `Critic` 类。
   - actor 可以继续使用 `global_logits`。
   - value 输入可以使用 `gmap_embeds[:, 0, :]` 或图表示的 pooled feature。

3. `run_r2r/iter_train.yaml`
   - 增加 PPO 学习率、clip range、gamma、GAE lambda、entropy coefficient、value coefficient、rollout 长度等配置。
   - 配置高层 reward 选项。

4. `run_r2r/r2r_vlnce.yaml`
   - 显式启用 PPO 所需的 `POSITION`、`STEPS_TAKEN`、`COLLISIONS`、`SUCCESS` 等 measures。

如果需要让环境显式返回每个高层动作的低层步数、碰撞数和 travelled distance，还需要修改：

```text
vlnce_baselines/common/environments.py
```

`habitat_extensions/nav.py` 只有在未来把动作重新改为 Habitat 原生的 HIGHTOLOW dispatch 时才是必需修改文件。

`graph_utils.py` 对基本 PPO 不是必需修改项，因为当前图状态和动作 mask 已经完整构造；但如果需要稳定保存 ghost ID、动作候选快照或跨时间步对齐候选，则可能需要修改。

## 11. PPO 实现风险

- 图长度 `L_i` 动态变化，PPO 旧策略和新策略必须使用相同的候选顺序与 mask。
- 只保存整数 action index 可能不够，最好同时保存当步的 ghost ID 或候选快照。
- 一个高层动作可能执行多个 TURN/FORWARD、backtracking 或 teleport，是一个 variable-duration 的 semi-MDP transition。
- `back_algo=teleport` 会跳过正常物理运动，可能导致碰撞和 path length 统计不一致。
- 当前训练动作会按 `sample_ratio` 被 teacher action 替换；纯 PPO 需要取消或明确记录这种行为。
- 当前 `global_logits` 已经参与 imitation cross-entropy，PPO actor loss 需要和 IL loss 协调。
- `Policy.Critic` 当前没有被实例化或调用。
- `ILPolicy` 中通用的 PPO 接口仍然是未实现状态。
- teacher stop 使用 1.5 米，而任务 success 使用 3.0 米，两个阈值不同。
- `stepk == max_len - 1` 会强制进入 stop 行为。
- waypoint augmentation 还会引入候选生成随机性，PPO 的概率账本需要明确是否包含这一级随机过程。
- 训练时 `ghost_real_pos` 可用；评估和推理时通常不可用，不能直接复用 SPL teacher。
