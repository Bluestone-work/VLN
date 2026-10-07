# Method-level literature check — 2026-10-07

Scope: version-pinned primary method sections, extending abstract screening.
This is not an exhaustive search or completed code/supplement audit. Source HTML
and SHA256 records are in `results/literature_methods_20261007/`.

## Important overlap

[Beyond Waypoints, 2606.07244v1](https://arxiv.org/html/2606.07244v1), Section 4.1,
predicts trajectory lengths from image tokens and truncates generated trajectories
before extracting candidate endpoints. Section 4.2 uses trajectory features in
navigation. State-dependent action horizon therefore already appears in VLN-CE.

Our interpretation: the universal fixed-granularity premise is untenable. This
does not establish an identical coarse/default/fine representation switch, but
generic adaptive granularity or controller/proposal consistency cannot establish
our distinction. Label calibration remains an experimental control.

## Other method comparisons

| Primary method sections | Relevant design | Implication |
| --- | --- | --- |
| [ETPNav, 2304.03047v3](https://arxiv.org/html/2304.03047v3), III-B–D | Online topology built from predicted proposals; graph-goal planning; obstacle-aware local execution and retries. | Variable graph size, movement through historical nodes and obstacle recovery already exist. Candidate count alone is insufficient differentiation. |
| [Macro-action graph RL, 2609.03906v1](https://arxiv.org/html/2609.03906v1), III-B–F | Frontier actions execute through backtracking and local control; action-aware values and graph PPO train macro-action decisions. | A generic graph-ranking/PPO pivot has substantial overlap. No separate learned contribution has been shown here. |
| [DifNav, 2508.09444v1](https://arxiv.org/html/2508.09444v1), Section 3 | Observation/instruction-conditioned trajectory generation, scoring and progressive refinement. | Trajectory candidates are established; these sections do not rule out every form of state-dependent horizon. |
| [AgenticNav, 2606.10577v3](https://arxiv.org/html/2606.10577v3), Section 3 | VLM decisions invoke perception, memory and pixel-target navigation tools. | Tool selection and on-demand sensing form an existing alternative architecture with different information/computation budgets. |
| [SparseNav, 2609.26408v1](https://arxiv.org/html/2609.26408v1), Section 3.2 | Instruction-aware sparse sensing with a hybrid frontier/local-direction candidate set. | Mixed candidates and instruction-dependent computation have precedents. A hybrid union is not necessarily a learned representation switch. |

## Remaining checks and research decision

These overlaps do not prove that an identical representation switch has been
published, nor that one helps this checkpoint. Associated code, supplements and
broader hierarchical/dynamic-action-space literature remain to be reviewed.
Absence from six papers cannot establish novelty.

Complete the frozen graph-option diagnostic and report cost-controlled local
opportunity. Require a specific observed failure mechanism and held-out deployable
gains before proposing another model. A positive diagnostic would establish an
opportunity, not novelty. Current NMS-density selector development remains NO-GO.

## Interruptible execution distinction (2026-10-07)

The current code audit separates two ideas that should not be conflated. Beyond
Waypoints Section 4.1 predicts a trajectory length and truncates candidate
trajectories **before execution**. ETPNav currently executes a selected ghost
through its control segment and only then returns a high-level observation. A
hypothetical ETPNav mid-option interrupt would acquire a new observation during
execution and issue a new graph action, which is not established as identical by
the cited paper.

This distinction does not create a novelty claim. The one interior-arrival
route is insufficient motivation, and a safe prototype would require a graph
transaction for consumed ghosts, predecessor updates, partial costs and pending
back-path state. The current cycle therefore keeps interruptible execution
NO-GO pending a separate mechanism and information-budget study.
