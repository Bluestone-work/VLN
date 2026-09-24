# STAGE 1：高层 RL transition 采集

日期：2026-09-23

本阶段只采集高层 transition，不实现 PPO 优化。

没有新增 actor、residual actor、显式 topology feature、semantic feature 或 history，也没有修改 GraphMap、waypoint prediction、reward API 或低层 controller 的运动逻辑。

## 高层 transition 边界

当前边界位于：

```text
RLTrainer.rollout()
  -> 构造当前 GraphMap 张量
  -> 根据 global_logits 采样 STOP/ghost graph action
  -> 构造 env_actions
  -> VectorEnv.step()
  -> VLNCEDaggerEnv.step()
  -> 完成全部 teleport / backtracking / TURN / FORWARD
  -> 返回 next observation 和 info
  -> 计算一个高层 reward
  -> 写入 self.rl_topo_transitions
```

一次 `VLNCEDaggerEnv.step()` 对应一个高层 transition。低层循环不会产生额外 RL transition。

## RL_TOPO 开启时的动作采样

训练 rollout 且 `RL_TOPO.ENABLED=true` 时，动作分布改为：

```python
dist = torch.distributions.Categorical(logits=nav_logits)
a_t = dist.sample()
old_log_probs = dist.log_prob(a_t)
entropies = dist.entropy()
```

此分支不执行：

```python
torch.where(..., teacher_actions, a_t)
```

也就是说，RL 模式下 teacher action 仍可用于现有 IL loss，但不再替换采样动作。

`RL_TOPO.ENABLED=false` 时仍使用原始 ETPNav 训练逻辑：

```python
Categorical(nav_probs).sample()
-> sample_ratio teacher replacement
```

评估和推理仍然使用原始 argmax。

## dist_before 和 dist_after

修改位置：

```text
vlnce_baselines/common/environments.py::VLNCEDaggerEnv.step()
```

在现有 movement logic 之前：

```python
dist_before = self.current_dist_to_goal()
```

随后原样执行：

- `act == 4` 的返回 front node；
- teleport 或 `multi_step_control()`；
- `single_step_control()`；
- 所有 TURN/FORWARD 和 tryout collision handling；
- `act == 0` 的 stop node 返回和 Habitat stop。

现有 movement logic 完成后：

```python
dist_after = self.current_dist_to_goal()
```

并附加：

```python
info["rl_high_level"] = {
    "dist_before": float(dist_before),
    "dist_after": float(dist_after),
}
```

为保持 `RL_TOPO.ENABLED=false` 时的原始路径，环境只有在配置启用 RL_TOPO 时才进行这两个距离测量和 info 扩展；Habitat 自带 reward 没有修改。

## 高层 reward

训练 rollout 在 `envs.step()` 返回后计算：

```python
reward = PROGRESS_WEIGHT * (dist_before - dist_after)
```

如果本次环境 transition 完成且 `dist_after <= 3.0`（与现有 RLTrainer evaluator 的判断一致）：

```python
reward += SUCCESS_REWARD
```

如果采样的 graph action index 是 0（STOP），且 `dist_after > 3.0`：

```python
reward -= WRONG_STOP_PENALTY
```

没有加入其他 reward 项。该 reward 只写入 transition buffer，不进入 Habitat `get_reward()`，也没有 PPO optimizer step。

## Transition buffer

每次启用 RL_TOPO 的训练 rollout 开始时创建：

```python
self.rl_topo_transitions = []
```

每个 transition 保存：

```text
environment_index
high_level_step
action_index
selected_graph_id
old_log_prob
entropy
reward
done
candidate_ids
valid_action_mask
graph_length
dist_before
dist_after
```

`candidate_ids` 保存当时完整的候选顺序：

```text
["STOP", visited node IDs..., ghost IDs...]
```

其中 stop token 的 `None` 被保存为字符串 `"STOP"`。

`valid_action_mask` 是：

```text
gmap_masks & logical_not(gmap_visited_masks)
```

并裁剪到当前 graph length。这样同时保存 `action_index`、候选顺序、selected ID 和 mask，不依赖 action index 在后续 graph step 中保持稳定。

## Debug 输出

当：

```yaml
RL_TOPO:
  ENABLED: true
  DEBUG_TRANSITIONS: true
```

只打印全局前 20 个 transition，格式为：

```text
Env 0 | Step 3
Candidates: ['STOP', 'g4', 'g7', 'g9']
Action index: 2
Selected: g7
log_prob: ...
distance_before: ...
distance_after: ...
reward: ...
done: ...
```

debug 计数器跨 rollout 保持，因此不会每个 episode 重新打印 20 条。

## 修改文件

本阶段修改：

```text
vlnce_baselines/common/environments.py
vlnce_baselines/ss_trainer_ETP.py
```

Stage 0 已修改并继续生效的文件：

```text
vlnce_baselines/config/default.py
run_r2r/iter_train.yaml
vlnce_baselines/models/Policy_ViewSelection_ETP.py
```

本阶段没有修改：

```text
vlnce_baselines/models/graph_utils.py
vlnce_baselines/waypoint_pred/*
habitat_extensions/nav.py
```

## 验证

通过：

```text
python -m py_compile \
  vlnce_baselines/common/environments.py \
  vlnce_baselines/ss_trainer_ETP.py \
  vlnce_baselines/models/Policy_ViewSelection_ETP.py \
  vlnce_baselines/config/default.py
```

通过：

```text
AST parse
静态 transition contract 检查
git -c core.whitespace=cr-at-eol diff --check
```

直接 import 仍受运行环境依赖缺失影响：

```text
ModuleNotFoundError: No module named 'jsonlines'
```

独立 PyTorch 分布检查也无法运行，因为当前环境没有安装 `torch`。源文件语法检查不依赖这些运行时包，并已通过。
