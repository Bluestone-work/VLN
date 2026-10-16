# Explicit graph/history waypoint-value probe

HEAD: `715d134` (`research: clarify transition view visualization`). The
remote tracking ref was still `50f56c7` when this probe started; the SSH push
was attempted but did not complete before the connection timeout.

This is an offline, outcome-blind diagnostic. It does not modify ETPNav, its
Graph Transformer, controller, STOP rule, or waypoint predictor. No endpoint
observation, goal distance, reference path, final outcome, PPO/GRPO/RL, or
end-to-end policy training is used as a feature.

## Question

After current-node RGB-D and one-step endpoint-latent probes failed to show a
stable residual signal, does the explicit graph/history state available before
an action predict the long-term BENEFIT/HARM of a native candidate waypoint?

## Dataset and representation

The source is the exact native single-action full-return table
`native_single_action_full_return_44_002/directional_pairs_002/pairs.jsonl`.
It contains 3,877 causal alternatives from 369 decision states, 44 episodes
and 11 scenes: 430 BENEFIT and 3,447 HARM labels; 182 states contain both
classes. Rows with UNCERTAIN are excluded exactly as in the visual probe.

The graph/history block is constructed only from pre-action graph records:
visited-node count, ghost count, non-null graph size, current graph index,
high-level step, back-path length, candidate count and index, candidate
visited/current/backtrack/ghost flags, candidate history age, ghost index,
current-proposal flag and STOP flag. The candidate graph id is taken from the
recorded native graph action identity. Features are not derived from the
alternative endpoint or its continuation.

All models use leave-one-scene-out fitting. Training rows are weighted first
by inverse candidate count per state and then by class. Thus a large action
set cannot dominate a fold, and no state is split across folds.

## Results

| Model | AUROC | AUPRC | Within-state pairwise | Benefit precision | Benefit recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| F0 native | 0.7219 | 0.2129 | 0.7791 | 0.2406 | 0.3581 |
| F0 + geometry | 0.7275 | 0.2147 | 0.7832 | 0.2438 | 0.3674 |
| Graph/history only | 0.7481 | 0.2976 | 0.7698 | 0.2384 | 0.5721 |
| F0 + graph/history | 0.7235 | 0.2147 | 0.7779 | 0.2394 | 0.3535 |
| F0 + geometry + graph/history | 0.7274 | 0.2156 | 0.7801 | 0.2379 | 0.3535 |

The graph-only AUROC and AUPRC point estimates are higher, but its
within-state ranking is lower than F0. The ranking task is the relevant test
for choosing among candidates in the same decision state. Adding graph/history
to native features does not improve that ranking.

Scene-bootstrap 95% intervals for the residual tests are:

| Contrast | AUROC delta | AUPRC delta | Pairwise delta |
| --- | --- | --- | --- |
| Graph/history only − F0 | [-0.0511, +0.0969] | [-0.0153, +0.1692] | [-0.0525, +0.0256] |
| F0 + graph/history − F0 | [-0.0220, +0.0211] | [-0.0165, +0.0148] | [-0.0078, +0.0057] |
| F0 + geometry + graph/history − F0 + geometry | [-0.0210, +0.0178] | [-0.0149, +0.0126] | [-0.0094, +0.0017] |

Every residual interval includes zero.

## Complementarity and rescue audit

The state-level rescue audit asks whether the model's highest-scoring action
is BENEFIT in a state that contains at least one BENEFIT alternative. It is an
offline ranking diagnostic, not an intervention result.

| Model | Native-wrong / model-correct | Native-correct / model-wrong |
| --- | ---: | ---: |
| F0 native | 107 | 187 |
| Graph/history only | 92 | 187 |
| F0 + graph/history | 104 | 187 |

The graph block therefore does not produce a larger set of native-wrong /
model-correct rescues, and the fused model slightly reduces the F0 count. This
does not support a deployable late residual or an online override.

## Scene consistency

The graph-only AUROC exceeds F0 in six scenes and is lower in five. F0 plus
graph/history exceeds F0 in five scenes and is lower in six. The largest graph
only gains are in `QUCTc6BB5sX` (0.621 → 0.762) and `TbHJrupSAjP`
(0.820 → 0.891), while it falls in `oLBMNvg9in8` (0.767 → 0.593) and
`Z6MFQCViBuw` (0.863 → 0.728). This is scene-dependent association rather
than stable candidate-level complementarity.

## Decision

**NO-GO for adding explicit graph/history residual information to the native
ETP action score on this cohort.** Graph/history fields contain predictive
association in pooled rows, but they do not add stable within-state ranking
information beyond the native ETP representation. The current evidence is
consistent with graph/history effects already being represented by native
logits, embeddings, graph size and back-path context, with remaining gains
driven by scene-specific structure.

This result does not prove that every longer-history or transition-value model
is ineffective. It does rule out the current simple graph/history side branch
as a defensible next deployment step. The next highest-value offline test is a
proper history-conditioned transition representation evaluated on a new,
scene-disjoint cohort, with candidate ranking and continuation cost retained
as separate targets. If that representation also has no residual signal, the
local visual/semantic/world-model waypoint-value direction should be recorded
as a negative result rather than tuned further.

## Short temporal-history follow-up

To check whether the negative result was caused only by using the current graph
snapshot, a second probe appended summaries of the previous one and two native
pre-action states: graph size, visited/ghost counts, back-path length, native
selected logit and margin, selected action index/type, proposal and STOP flags.
These summaries were zero-padded at the start of each episode and never used
future outcomes.

| Model | AUROC | AUPRC | Within-state pairwise |
| --- | ---: | ---: | ---: |
| F0 native | 0.7219 | 0.2129 | 0.7791 |
| Temporal history only | 0.7329 | 0.2846 | 0.7692 |
| F0 + temporal history | 0.7160 | 0.2118 | 0.7763 |
| F0 + geometry + temporal history | 0.7196 | 0.2126 | 0.7791 |

The F0 + temporal-history residual intervals are [-0.0417, +0.0238] for
AUROC, [-0.0279, +0.0196] for AUPRC and [-0.0111, +0.0048] for within-state
pairwise. The temporal history-only block has 93 rescue states versus 107 for
F0. This follow-up also fails the candidate-level complementarity test.

Artifacts:

- `research/tools/prepare_graph_history_value_dataset.py`
- `research/tools/probe_graph_history_value.py`
- `research/tools/prepare_temporal_history_value_dataset.py`
- `research/results/graph_history_value_dataset_001/`
- `research/results/graph_history_value_probe_001/`
- `research/results/temporal_history_value_dataset_001/`
- `research/results/temporal_history_value_probe_001/`
