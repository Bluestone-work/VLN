# ETP-R1 / CBR-GRPO 迁移与实验记录（2026-11-19）

## 已完成

- 从官方 `ETP-R1` 仓库迁移 R1 policy、R1 VLN-BERT、R1 waypoint predictor、R1 SS/GRPO trainer，以及 `R1Env` 注册层。
- 共享配置增加官方 GRPO 参数和 CBR 参数；`GRPO.cf_enabled` 默认 `False`。
- 新增 `branch_state.py`：保存/恢复 Habitat agent state、GraphMap、`prev_vp`、Python/NumPy/PyTorch/CUDA RNG，并用 fingerprint 检查回滚结果。
- 新增 `counterfactual_rollout.py`：候选 top-k、分支 return、GRPO branch-relative advantage 和逐分支恢复。
- 新增 `CBR-GRPO-R1` registry entry。CBR 打开但没有通过 probe 时会显式报错，不会悄悄执行伪反事实更新。

## 真实验证

### CBR 纯状态 smoke

命令：

```bash
PYTHONPATH=/home/wj/habitat-lab:/home/wj/VLN-CE/ETPNav \
  /home/wj/miniconda3/envs/etpnav_legacy/bin/python \
  research/tools/test_cbr_branch_state.py
```

结果：`CBR_BRANCH_STATE_SMOKE_PASS`。两个候选从同一初始状态运行；环境位置、GraphMap、`prev_vp` 回滚；随机数序列也恢复一致；分支优势为 `1.0/-1.0`。

### 官方 R1 checkpoint 兼容性

对本地 `data/pretrained/ETP/mlm.sap_r2r/ckpts/model_step_82500.pt` 运行了真实初始化。失败是预期的结构不兼容：

- 词表：`30522` vs R1 的 `250002`；
- position embedding：`512` vs `514`；
- R1 fused SAP head：`1536` hidden vs 旧 checkpoint 的 `768`。

因此旧 checkpoint 没有被宣称为官方 R1 权重，CBR/R1 长训练保持关闭。

### 可靠 fallback continuation

由于官方 joint XLM-R/R1 release checkpoint 尚未下载到本机，启动了独立目录的旧 ETPNav R2R-CE RL_TOPO continuation：

- 起点：`data/logs/checkpoints/release_r2r/ckpt.iter12000.pth`；
- 真实训练：1 iteration，`iter 12001`；
- rollout：10 个 high-level actions；
- mean reward `1.474`，mean return `7.432`；
- policy loss `0.000770`，value loss `60.794758`；
- actor/critic grad norm `0.025/43.595`；
- 输出 checkpoint 中新增 6 个 RL_TOPO critic/residual 参数键（旧 release checkpoint 没有这些键），说明 optimizer 更新和保存路径实际执行；
- 输出 checkpoint：`data/logs/checkpoints/continuation_r2r_20261119/ckpt.iter12001.pth`。

在同一 `val_unseen` 4-episode cohort 上做了更新前后配对评估：

| checkpoint | SR | SPL | nDTW | distance-to-goal |
|---|---:|---:|---:|---:|
| release iter12000 | 1.000 | 0.961482 | 0.893008 | 1.703980 |
| continuation iter12001 | 1.000 | 0.961482 | 0.893008 | 1.703980 |

这是一轮 smoke continuation，指标未变化；不能据此声称 SR 提升。

## 当前结论

CBR 的关键状态恢复机制已经有可重复的单元级验证，但还没有在 Habitat R2R 真实候选上完成 branch-return census，也没有接入 GRPO loss。官方 R1 训练的阻断原因是官方 joint checkpoint/extra files 缺失，而不是代码编译错误。当前可复现、可继续运行的是旧 ETPNav RL_TOPO continuation。

