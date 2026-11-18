# TETP：事务式证据拓扑规划（候选创新点）

日期：2026-11-18

## 核心问题

当前 ETPNav 的 waypoint candidate 一旦进入 GraphMap，就会同时成为：

1. 可执行 graph action；
2. persistent graph memory 的输入；
3. cross-modal Graph Transformer 的上下文。

已有实验已经分别观察到两个问题：dense candidate 会污染 native memory；新
candidate 的 logit 又会在 expanded action set 中出现明显的 OOD score excursion。
因此单独做 attention mask、native logit isolation 或普通 ranker 都不能保证安全。

## 提议的方法

提出 **TETP（Transactional Evidence Topological Planning）**。它把图分成两个
状态：

- **committed evidence graph**：只包含已经观测并执行过的 native node/edge，作为
  唯一 persistent memory；
- **proposal buffer**：当前 heatmap 生成的未执行候选，只能作为 query 读取
  committed graph，不能写回 native memory，也不能改变 native ghost 的均值、位置
 关系或 STOP history。

每个 candidate 进入执行集合前必须通过三个无需 outcome 的准入条件：

### 1. Native noninterference certificate

用 native-only memory 和 proposal-query pass 分别计算 native logits。若

\[
\max_{a\in A_{native}} |l_a^{query}-l_a^{native}| > \epsilon,
\]

则整批 proposal abstain，执行原生 action。当前原型默认
`epsilon=1e-4`，这是表示不变性门槛，不是可调的性能阈值。

### 2. Local native support

对每个 proposal query embedding \(z_p\)，计算它与当前 native action embedding
集合的最大 cosine support：

\[
s(p)=\max_{a\in A_{native}} \cos(z_p,z_a).
\]

只有 `s(p) >= 0.50` 的 proposal 才能进入候选集合。这个条件把“新增 action”
限制在当前 native representation 的局部支持域中，防止仅凭一个大 logit 进入。

### 3. Query stability and score envelope

在两个或多个等价 query pass（例如 panorama view dropout、固定角度子采样）中
得到 proposal scores \(l_p^{(k)}\)。要求：

\[
\operatorname{std}_k(l_p^{(k)}) \le 0.25,
\quad
l_p-\max_{a\in A_{native}}l_a \le 0.50.
\]

第一项过滤 epistemic 不稳定，第二项过滤 expanded action set 中不受支持的
异常高分。候选只有在通过三项检查且最终分数高于 native top action 时才替换
native action；否则保持 native action（abstention）。

执行完成并得到新的真实观测后，候选才可以从 proposal buffer **commit** 到
evidence graph。失败、未执行或仅由 heatmap 假设产生的节点不进入 persistent
memory。

## 为什么这是当前最值得做的创新点

它不是再增加一个 ranker，也不是把 dense candidate 无条件交给原 argmax，而是把
当前实验暴露的两个失败机制统一成一个可证伪的设计：

- H1（representation contamination）由 committed/proposal memory transaction
  和 native noninterference certificate 处理；
- H2（dense OOD score）由 local support、multi-pass stability 和 native score
  envelope 处理；
- 不需要 goal distance、reference path、完整回报或 outcome label 才能决定是否
  abstain；
- native action、native STOP、controller 和 primitive budget 都可以保持不变。

与已有工作的差异需要谨慎表述：Perceiver-style memory/query separation、稀疏
语义记忆、trajectory waypoint 和 graph macro-action RL 都已有先例。当前可主张
的候选贡献是 **面向 VLN-CE 动态 graph action set 的“已观测证据提交”和“新增
候选准入证书”联合机制**，而不是宣称 memory/query attention 或 conformal
calibration 本身新颖。完整方法和代码审查后才可确定论文级 novelty。

## 最小可行实验

冻结 R2R checkpoint、seed、controller、STOP 和 15-step budget，注册四个 arm：

| Arm | Graph memory | Action set | 目的 |
|---|---|---|---|
| A0 | native committed graph | native | 原始 baseline |
| A1 | native+dense shared graph | native+dense | 已知失败对照 |
| A2 | committed native graph + dense query | native+dense | 只验证 memory transaction |
| TETP | committed native graph + dense query | certificate-admitted union | 完整方法 |

所有阈值在读取 route outcomes 前固定为：`1e-4 / 0.50 / 0.25 / 0.50`。先做
native-success control replication，再做包含 native failures 的 route cohort。首要
门槛是：

1. native-success route 的 action/STOP/trace/primitive 完整保留；
2. TETP 不得出现 A1 的系统性 success destruction；
3. 在保留 native success 的前提下，才统计 unresolved route 的 rescue、SPL、nDTW
   和 primitive cost；
4. 若全部 proposal 都被拒绝，结果依然是有价值的“安全但无增益”边界，不能改阈值
   追求 coverage。

## 当前可验证的先验

现有结果已经提供了直接动机，但还没有验证 TETP 的闭环收益：

- 真实 dense graph 在 87 个状态上让 native embedding 平均漂移 4.45，logit 平均
  漂移 1.18；
- memory/query probe 已经在 87/87 状态保持 native argmax，native logit 最大误差
  `3.81e-6`；
- 99-route dense control 中 A1 破坏 18/99 native successes；
- 44-route A2-MASK 即使不执行 dense action，仍破坏 10/44 native successes，说明
  memory contamination 必须在 action admission 之前处理。

因此下一步应该先实现 TETP 的闭环 trace/audit runner，在新冻结 cohort 上验证
native-success retention；在这个 gate 通过前，不训练 ranker、PPO、VLM，也不使用
full-return outcome 调阈值。

## 代码状态

`vlnce_baselines/adaptive_action/native_anchor_gate.py` 是独立的纯 Python 原型，
不改变 baseline 和 simulator。它实现 native drift、native support、query stability
和 score envelope 四项检查；对应单元测试位于
`vlnce_baselines/adaptive_action/test_native_anchor_gate.py`。下一步只需把它接到
已有的 `graph_option_capture` / memory-query probe，先做 trace-level certificate
审计，再决定是否进入闭环 rollout。
