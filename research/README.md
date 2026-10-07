# Adaptive Action Abstraction 研究审阅入口

本分支保存截至 2026-10-07 的研究报告、配置、诊断脚本和轻量结果。当前没有得到可部署的导航提升；原始 NMS 粒度切换、短窗口标签蒸馏和当前数据上的 selector 训练均未通过开发门槛。具体证据与各项结论的适用范围见下列报告。

建议阅读顺序：

1. [中文进展](RESEARCH_STATUS_CN.md)：已完成工作及当前判断。
2. [主报告](REPORT.md)与[实验总表](EXPERIMENT_TABLE.md)：基线、固定粒度、oracle 和后续诊断。
3. [代码审计](AAA_CODE_AUDIT.md)与[基线复现](BASELINE_REPORT.md)。
4. [失败分析](FAILURE_ANALYSIS.md)：历史代理指标的纠正，以及尚不能因果归类的问题。
5. [完整回报实验](FULL_RETURN_LABEL_REPORT.md)、[原生 STOP 诊断](NATIVE_STOP_FEASIBILITY_REPORT.md)及[中途执行审计](INTERRUPTIBLE_EXECUTION_AUDIT.md)。
6. [研究日志](RESEARCH_LOG.md)：实验过程、失败尝试和下一步条件。

## 本分支包含什么

- `configs/`、`tools/` 和 `../vlnce_baselines/adaptive_action/` 保存实验配置、工具与模块。
- `results/` 保存小于 5 MiB 的 JSON 摘要、指标、清单及部分历史快照；这是审阅用选集，不是完整实验归档。
- [实验工作树补丁](results/interruptible_execution_audit/archive_001/git_diff.patch)保存相对于 `1c1a794148a774940d59a12dc601ea51febf3d7e` 的七个已跟踪文件改动，包含核心诊断接入及配置改动。本次提交没有将这些改动直接应用到核心文件；阅读代码时应结合该补丁。它是当时已有工作树的完整差异，不意味着其中每项改动都由最近一轮审计新增。

## 复核与复现边界

这个分支用于审阅，直接 checkout 后不是已经接好全部诊断钩子的可运行实验版本。需要重放时，应在独立干净 checkout 中先检查并应用上述补丁，再准备报告指定的 legacy 环境、数据和 checkpoint。

原始 JSONL 轨迹、数据集、模型权重、缓存论文 HTML、大部分 stdout/stderr、图件及部分源快照未上传，仍在原实验机器。报告和 manifest 中指向这些文件的路径、哈希保持原样；路径出现在清单中不代表文件已经上传。原始实验记录中的 commit 是运行时基准 commit，不能替代工作树补丁和源文件哈希。

本次发布不新增导航评测，不重新解释旧结果为模型增益。静态源字符串检查也不等于动态仿真测试。
