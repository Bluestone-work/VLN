# STAGE 0：RL_TOPO 代码准备记录

日期：2026-09-23

目标：为后续高层拓扑 PPO 做接口准备，同时保持 ETPNav 原有行为。

本阶段没有实现 PPO，没有新增 actor，没有修改 reward、GraphMap、waypoint prediction、动作选择或低层 controller。

## 已修改文件

### `vlnce_baselines/config/default.py`

增加顶层 `_C.RL_TOPO` 配置默认值：

```yaml
RL_TOPO:
  ENABLED: false
  DEBUG: false
  DEBUG_TRANSITIONS: false
  PROGRESS_WEIGHT: 1.0
  SUCCESS_REWARD: 5.0
  WRONG_STOP_PENALTY: 2.0
  GAMMA: 0.99
  GAE_LAMBDA: 0.95
  PPO_CLIP: 0.2
  VALUE_COEF: 0.5
  ENTROPY_COEF: 0.01
  PPO_EPOCHS: 4
```

`ENABLED` 默认关闭。这些字段目前只作为未来阶段的配置入口，不会改变当前训练逻辑。

### `run_r2r/iter_train.yaml`

加入同名 `RL_TOPO` 配置段，所有值与默认配置一致，确保 R2R 实验配置中明确记录该功能为关闭状态。

该文件在本次工作开始前已经存在一处与本任务无关的工作树修改：`MODEL.pretrained_path` 使用 `data/pretrained/...`。本次没有修改或回退这处已有改动。

### `vlnce_baselines/models/Policy_ViewSelection_ETP.py`

给 `ETP.forward()` 增加可选参数：

```python
rl_topo_debug=False
```

在 `mode == "navigation"` 分支中保留原始 `outs` 字典和 `global_logits`，并在 `rl_topo_debug=True` 时验证：

```text
global_logits.ndim == 2
gmap_embeds.ndim == 3
gmap_embeds.shape[:2] == global_logits.shape
gmap_embeds.shape[-1] == 768
```

当前 `GlocalTextPathNavCMT.forward_navigation()` 已经返回：

```python
{
    "gmap_embeds": gmap_embeds,
    "global_logits": global_logits,
}
```

因此本阶段只是把这个接口显式保留下来并增加 debug 合约检查，没有改变 logits 数值或返回数据结构。

### `vlnce_baselines/ss_trainer_ETP.py`

在导航输入中传入：

```python
'rl_topo_debug': bool(
    self.config.RL_TOPO.ENABLED and self.config.RL_TOPO.DEBUG
)
```

当 `RL_TOPO.ENABLED=false` 时，即使误将 `DEBUG` 打开，forward 也不执行额外 shape assertion，不改变现有动作选择、teacher replacement、IL loss 或环境动作。

## 当前导航输出形状

ETP navigation forward 的最终输出为：

```text
global_logits: [B, Lmax]
gmap_embeds:   [B, Lmax, 768]
```

其中：

- `B` 是当前并行环境数量；
- `Lmax` 是 batch 中最大 GraphMap 长度；
- graph action 顺序仍然是 `[STOP, visited nodes..., ghost nodes...]`；
- visited node 和 padding mask 的处理完全保持不变。

## 未修改的行为

在 `RL_TOPO.ENABLED=false` 下，以下路径保持原样：

- waypoint predictor；
- GraphMap 创建和更新；
- teacher action 计算；
- `sample_ratio` 控制的 teacher replacement；
- imitation cross-entropy；
- 训练采样；
- evaluation argmax；
- inference argmax；
- ghost 到低层 controller 的转换；
- `VLNCEDaggerEnv.step()`；
- reward 和 done 处理。

本阶段没有任何新的 `nn.Parameter`。

## 检查结果

通过：

```text
python -m py_compile \
  vlnce_baselines/config/default.py \
  vlnce_baselines/models/Policy_ViewSelection_ETP.py \
  vlnce_baselines/ss_trainer_ETP.py
```

通过：

```text
git -c core.whitespace=cr-at-eol diff --check
```

Python AST/compile 检查也通过。

仓库当前 Python 运行环境缺少 `jsonlines`，因此直接导入以下模块无法完成：

```text
vlnce_baselines.config.default
vlnce_baselines.models.Policy_ViewSelection_ETP
vlnce_baselines.ss_trainer_ETP
```

错误为：

```text
ModuleNotFoundError: No module named 'jsonlines'
```

这属于运行环境依赖缺失；源文件语法检查已通过。

## 原始 ETPNav baseline 复现命令

在仓库根目录执行：

```bash
bash run_r2r/main.bash train 29500
```

该命令使用 `run_r2r/iter_train.yaml`，其中 `RL_TOPO.ENABLED=false`。它仍然走原来的 `SS-ETP` imitation-learning 训练路径；`main.bash` 当前默认使用 2 个分布式进程和 8 个环境。
