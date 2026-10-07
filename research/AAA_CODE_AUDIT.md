# Adaptive Action Abstraction: ETPNav Code Audit

Date: 2026-10-06  
Repository: `/home/wj/VLN-CE/ETPNav`  
Audited commit: `1c1a794` (`Use residual PPO from IL checkpoints`)

This audit describes the released ETPNav-style R2R-CE/RxR-CE implementation before any adaptive abstraction controller is introduced. The optional diagnostic changes in this cycle are behind configuration switches; `ACTION_ABSTRACTION.LEVEL=default` preserves the original waypoint NMS parameters.

## Module and function map

| Responsibility | File | Main functions/classes |
| --- | --- | --- |
| Experiment entry and config merge | `run.py`, `vlnce_baselines/config/default.py` | `run_exp`, `get_config` |
| R2R experiment config | `run_r2r/iter_train.yaml`, `run_r2r/r2r_vlnce.yaml` | IL/RL/task/simulator settings |
| High-level trainer | `vlnce_baselines/ss_trainer_ETP.py` | `RLTrainer.train`, `eval`, `rollout` |
| Waypoint predictor | `vlnce_baselines/waypoint_pred/TRM_net.py` | `BinaryDistPredictor_TRM.forward` |
| Heatmap NMS | `vlnce_baselines/waypoint_pred/utils.py` | `nms` |
| Waypoint and navigation policy | `vlnce_baselines/models/Policy_ViewSelection_ETP.py` | `ETP.forward` modes `waypoint`, `panorama`, `navigation` |
| Topological memory | `vlnce_baselines/models/graph_utils.py` | `GraphMap.identify_node`, `update_graph`, `front_to_ghost_dist`, `get_pos_fts` |
| Candidate feature batching | `vlnce_baselines/ss_trainer_ETP.py` | `_vp_feature_variable` |
| Graph tensor construction | `vlnce_baselines/ss_trainer_ETP.py` | `_nav_gmap_variable` |
| Continuous execution | `vlnce_baselines/common/environments.py` | `VLNCEDaggerEnv.step`, `single_step_control`, `multi_step_control`, `teleport` |
| Dataset metrics | `vlnce_baselines/ss_trainer_ETP.py`, `habitat_extensions/measures.py` | SR, SPL, nDTW, sDTW, path/collision measures |
| Optional abstraction diagnostics | `vlnce_baselines/adaptive_action/` | `action_generators`, `diagnostics` |

## Compact pipeline

```text
RGB/depth panorama (12 views) + instruction
                 |
                 v
  depth/RGB encoders -> waypoint heatmap [120 angles x 12 distances]
                 |
                 v
  softmax -> circular NMS (default: max 5, sigma=(7,5))
                 |
                 v
  variable waypoint candidates (angle, distance, RGB/depth, angle feature)
                 |
                 v
  GraphMap: current nodes + ghost/frontier nodes + graph edges
                 |
                 v
  global cross-modal graph navigator -> logits over [STOP, visited, ghosts]
                 |
                 v
  selected STOP or ghost; ghost -> nearest front node + target position
                 |
                 v
  teleport/control backtrack + collision-aware low-level TURN/FORWARD
                 |
                 v
  Habitat metrics, next observation, next graph update
```

## Current action space and data flow

The waypoint predictor receives 12 RGB/depth views. `BinaryDistPredictor_TRM` produces logits of shape `[B,120,12]`: 120 angular bins at 3 degrees and 12 distance classes. At inference, the logits are flattened and softmaxed, circularly padded, and passed through NMS. The original ETPNav call uses `max_predictions=5` and `sigma=(7.0,5.0)`. Each selected heatmap cell becomes one candidate:

```text
angle = 2*pi - angle_bin/120*2*pi
distance = (distance_bin + 1) * 0.25 m
```

The actual number of candidates is variable and can be smaller than five when NMS leaves zero cells in a row. The code does not pad this candidate list with a fixed semantic action type; it pads feature tensors only at batch collation time.

`_vp_feature_variable` concatenates candidate views with non-candidate panorama views. Candidate `nav_types` are 1 and panorama-only views are 0. The panorama transformer produces 768-dimensional candidate/current-view representations.

`GraphMap.update_graph` inserts the current real node, connects it to the previous node, localizes candidates against existing real nodes, and creates or merges remaining candidates as `g0`, `g1`, ... ghost nodes. It stores ghost position history, mean position, averaged visual embedding, front-node list, and (in training/video mode) real candidate positions. The graph is recomputed with NetworkX shortest paths after each update.

`_nav_gmap_variable` emits a variable-length graph sequence with the exact ordering:

```text
[STOP(None), visited real nodes..., unvisited ghost nodes...]
```

`gmap_visited_masks` masks real visited nodes but leaves STOP and ghosts available. Padding is represented by `gmap_masks`. Pairwise graph distances and 7-dimensional relative position features are passed to the global cross-modal encoder.

The navigator returns `global_logits [B,Lmax]` and graph embeddings `[B,Lmax,768]`. Invalid/padded/visited entries are set to `-inf`. In the current optional RL path, a zero-initialized residual head adds a small correction to valid logits; the IL global head and graph encoder are frozen.

## Action selection and execution

Evaluation uses argmax over the masked graph logits. Training uses categorical sampling in the RL path; the compatibility IL path uses categorical sampling followed by the existing teacher replacement schedule. A selected index of 0 is STOP. For a ghost, the trainer finds the nearest front node through `GraphMap.front_to_ghost_dist`, then constructs a high-level `act=4` action.

`VLNCEDaggerEnv.step` executes a high-level option. With `IL.back_algo=control`, it follows the graph back path using `multi_step_control`, then uses `single_step_control` to turn and move toward the ghost. With `teleport`, it teleports to the front/stop position before low-level movement. STOP similarly returns to the selected historical node and then calls Habitat STOP. The environment returns Habitat metrics and, when `RL_TOPO.ENABLED`, before/after geodesic distance for the high-level reward.

## Training and evaluation

The released ETPNav baseline is imitation learning from an ETP language/SAP checkpoint, with waypoint predictor weights loaded from `data/wp_pred/check_cwp_bestdist_hfov90` (R2R) or `...hfov63` (RxR). The current branch also contains an opt-in residual topology PPO path: it records one transition per high-level option, computes GAE, and replays graph snapshots through a 769-parameter residual head and a 394,241-parameter critic. It does not yet implement adaptive action abstraction.

R2R and RxR evaluation are performed in `RLTrainer._eval_checkpoint`. The evaluator computes success at distance <=3 m, SPL from the predicted path and initial goal distance, nDTW from fastDTW against the reference path, sDTW as nDTW times success, and oracle success when any recorded position is within 3 m. It also records primitive simulator steps, path length, collision count normalized by path positions, ghost count, and optional high-level stop statistics.

## Fixed assumptions and dimensionality boundaries

1. The waypoint heatmap is fixed at 120 angular bins, 12 views, and 12 distance classes.
2. Default NMS retains at most five cells with `sigma=(7,5)`.
3. Candidate lists are variable length, but downstream graph masks and padding assume one STOP followed by real/ghost graph nodes.
4. Graph embeddings and position features are 768 and 7 dimensions respectively; the global policy head expects this representation.
5. Panorama candidate features assume 12 total views and `nav_types` distinguish candidate from non-candidate views.
6. High-level action index 0 is STOP; other indices must resolve to current ghost IDs. Any new candidate generator must preserve graph ID/mask alignment.
7. `max_traj_len=15` is a high-level rollout limit in the trainer. At the final step, a sampled non-STOP option can be executed as a forced STOP while the diagnostic transition must distinguish the sampled and executed actions.
8. `TASK.MEASUREMENTS` in the base R2R YAML are mostly commented out; evaluator metrics come from the environment's position/collision info and trainer-side calculations.
9. R2R and RxR use different instruction encoders/data paths and waypoint HFOV checkpoints; their numerical results must not be pooled.
10. The released IL checkpoint predates the optional residual/critic heads. Baseline mode now freezes those heads and skips incompatible optimizer state, while state-dict loading remains strict=False.
11. The legacy `dagger_trainer.py` contains a separate `max_cand_len=6` assumption for its older candidate-action pipeline. It is not on the active `SS-ETP` GraphMap path audited here, but any cross-trainer comparison must keep this distinction explicit.

## First-cycle diagnostic extension

`vlnce_baselines/adaptive_action/action_generators.py` defines three controlled NMS specifications: `default=(5,(7,5))`, `coarse=(3,(12,7))`, and `fine=(8,(4,3))`. They share the same predictor, encoders, graph update, navigator, and low-level controller. These are fixed proposal-density proxies for a go/no-go experiment; they are not presented as the final adaptive method.

When `ACTION_ABSTRACTION.DIAGNOSTICS_ENABLED=True`, the trainer writes one JSONL record per high-level decision. Records include episode/scene/step, pose, instruction, all generated waypoint angles/distances/NMS scores, candidate count and angular dispersion, proposal-distribution proxy, graph candidates/logits, selected target, policy entropy, goal/reference-path distances before execution, after-execution distance, progress, done flag, near-goal flag and collision count when present. Fields that use goal/reference information are marked `privileged_analysis_only` and are not inputs to a deployable selector. Reference-path distance is explicitly marked unavailable when the evaluation environment does not expose the reference data.

When `ACTION_ABSTRACTION.ORACLE_ENABLED=True`, the waypoint forward pass also returns the same state's `[120,12]` probability heatmap. The trainer derives coarse/default/fine NMS candidates from that tensor and calls the worker-local `cand_dist_to_goal` method for each candidate. That method restores the exact simulator pose after each counterfactual forward rollout. Oracle records are written separately from ordinary diagnostics and never affect graph updates, action selection or the executed trajectory.

For downstream failure analysis, `GraphMap.last_candidate_vps` records which
real or ghost node each current candidate became after localization/ghost
merging. Diagnostics expose the graph index, validity mask, visited mask,
navigator logit and selected status for each mapping. This allows a matched
candidate-to-graph selection analysis without changing the action path.

`ACTION_ABSTRACTION.GRAPH_SELECTION_ORACLE_ENABLED` is a separate
analysis-only intervention. It preserves default waypoint generation and graph
construction, then uses privileged one-step goal progress to choose among
valid current ghost nodes for an evaluation rollout. Its trajectory metrics are
observed privileged outcomes, not a certified optimal bound or deployable agent.

`ACTION_ABSTRACTION.INCLUDE_RANKER_EMBEDDINGS` optionally stores the frozen
768-dimensional instruction-conditioned graph embedding for each candidate in
the diagnostic JSONL. This is an offline feature-extraction path only; it does
not alter the graph logits or the executed action.

## Action-label contract correction (2026-10-07)

`VLNCEDaggerEnv.cand_dist_to_goal` is a raw-candidate proxy: it sets the desired
heading continuously and executes forward steps, then restores the pose.
`single_step_control` instead quantizes turns and can use collision tryout;
`step(act=4)` first follows a front/back path and then targets the merged ghost.
Consequently the probe is not a faithful counterfactual of a full graph option.
Pose reset alone has not been shown to restore all task, collision and RNG state.

Multiple raw candidates can share a graph ID, its logit and its 768-dimensional
embedding while receiving different probe-progress labels. Current-candidate
diagnostics also omit ranking labels for historical ghosts and STOP. See
`RANKER_VALIDITY_AUDIT.md`. Earlier references to upper bounds and causal
selection/execution decomposition should be read as privileged proxy analyses.

## Full-option trace and replay path (2026-10-07)

The original navigation/controller code is unchanged by this extension:

```text
research/tools/run_option_capture.py
  -> normal run.run_exp / SS-ETP evaluator
  -> explicitly registered VLNCEOptionTraceEnv (capture only)
  -> original VLNCEDaggerEnv.step
       -> multi_step_control(back_path)
       -> single_step_control(merged ghost) OR Habitat STOP
       -> wrap_act -> original primitive simulation + measure updates
  -> typed full action / poses / RNG / per-primitive events / sensor hashes

research/tools/replay_option_calibration.py (separate simulator process)
  -> sequential episode replay OR reset + restore captured pose/RNG
  -> same original controller
  -> compare endpoint, progress, action/collision sequence, done and RNG
```

Important contract details:

- `use_tryout = IL.tryout and not ALLOW_SLIDING`; the evaluated baseline has
  sliding=true, hence no stochastic tryout is executed despite IL.tryout=true.
- Keep NumPy array/scalar dtypes and tuples in serialized actions. Converting
  float32 positions to float64 lists can change quantization boundaries.
- STOP includes a path to the node with the highest historical stop score; it
  need not stop at the current pose. STOP itself bypasses `wrap_act`, so the
  tracer records a separate event with collision=null.
- The separate reset-based branch resets task metrics rather than restoring
  whole-episode measure history. Its calibrated labels are local execution
  outcomes, not counterfactual episode SPL/nDTW.
- The trace environment and replay retain goal distances only for analysis.
  Observation arrays are hashed without feeding distances into the policy.
- `TASK_CONFIG.DATASET.DATA_PATH` can point to an explicitly recorded balanced
  diagnostic subset through the research runner; source samples and vocabulary
  are preserved, and original split files are untouched.

Results and limits: `OPTION_CALIBRATION_REPORT.md`. Alternative graph options
must be captured before consumed ghosts are deleted. The new opt-in collection
path below implements that ordering (`OPTION_LABEL_PROTOCOL.md`).

## Complete graph-option capture (2026-10-07 follow-up)

`adaptive_action/graph_option_capture.py` adds the explicitly registered research
trainer `SS-ETP-OptionCapture`. The original rollout has two guarded callbacks:
prepare after logits/STOP scores are known but before ghost consumption, and
commit after native action construction but before environment stepping. With
the ordinary trainer, the callback attribute is absent and these calls are inert.

The capture enumerates STOP plus valid unvisited graph IDs using existing masks;
visited nodes and padding are excluded. Every entry uses the merged ghost target,
the native nearest front node and shortest back path. STOP uses the highest
historical STOP-score node. Serialized arrays retain their dtypes. The shadow
selected action must exactly equal the native typed action dictionary before a
record is written. This check includes forced STOP and backtracking.

One graph ID yields one option even when several current proposals alias it.
Historical ghosts are included and marked separately from current proposals.
Frozen 768-dimensional instruction-conditioned graph embeddings are recorded as
features; privileged distances remain in the separate execution trace only.

At the last permitted high-level decision, non-STOP structural options are marked
inadmissible and **not executed** by the branch probe. Rank-only regret excludes
all actual learned/forced STOP states. The simulator/controller implementation,
evaluation definitions, ghost deletion and native action selection remain intact.

`research/tools/probe_graph_options.py` joins graph and native trace records,
requires a passed untraced control, executes every admissible option in isolated
resets, checks the originally selected action as a sentinel, and reverses option
order on two preselected states per episode. All raw branch traces are retained.
`analyze_graph_options.py` reports local goal-progress differences alongside
controls limiting primitive count, movement distance and current-proposal access.
These labels do not represent counterfactual episode SR/SPL/nDTW.
## Ordered-route offline extension (2026-10-07)

Research-only tools added after the calibrated full-option probe:

- `research/tools/prepare_option_expansion.py`: hashes training scene/route/
  instruction metadata; excludes pilot routes and verifies validation-route
  disjointness. Dataset scene IDs resolve under `data/scene_datasets/`.
- `research/tools/run_graph_option_cycle.py`: serial capture -> control ->
  noninterference -> complete probe -> independent integrity -> analysis;
  fails before a dependent stage when a prerequisite fails. Each stage has
  separate exclusive stdout/stderr logs and source/config provenance.
- `research/tools/route_alignment.py`: pure NumPy exact ordered-prefix DTW,
  monotonic endpoint choice, consecutive-position deduplication and common-
  selected-endpoint route gate. No dependency on the policy or simulator.
- `research/tools/analyze_route_options.py`: joins original full-option traces,
  native dense GT references and unchanged baseline prefixes. Reconstructs
  native float32 path length and fastdtw nDTW/SDTW before any branch analysis.
  The diagnostic DTW deliberately uses float64 and is not the evaluator.
- `research/tools/inspect_route_results.py`: verifies all previous cost gains
  and state identities; saves explicitly post-hoc descriptive strata and
  horizontal/elevation plots without changing the preregistered gate.

No native navigation, graph, observation, STOP or controller code is edited
by this extension. Counterfactual goal distance, reference path and actual
branch costs stay in research outputs; they must never become deployment inputs.
## Single-action continuation interceptor (2026-10-07)

`adaptive_action/single_intervention.py` is imported only by
`research/tools/run_single_intervention.py` and registers the separate
`SS-ETP-SingleIntervention` trainer. It reuses the existing `prepare/commit`
callback locations: one scheduled non-STOP index is changed before the native
action builder updates its front node and deletes the chosen ghost. No simulator
step, reset, teleport, graph rewrite or baseline suffix replay occurs in the
hook. After intervention, every action index remains the native navigator's
choice. The original trainer and controller source files are unchanged.

Before intervention, the hook verifies graph IDs, masks, logits, typed position
and native action against the archived actual prefix. `commit` verifies the
resulting full action and native selected-ghost consumption. Invalid STOP,
horizon, stale/masked/visited alternative or repeated event causes a hard error.
`finish` requires each scheduled event exactly once. The seven event choices
are privileged research inputs, never a deployable policy.

`audit_single_intervention.py` verifies disabled noninterference first, then
enabled prefix/alternative-execution fidelity, the other 57 original routes,
and paired native continuation metrics. It reports rescue and harm separately.
Disabled 64-route control matches all 474 full traces, 1,152 episode metrics
and 18 aggregates exactly; 13 focused interceptor/audit tests pass.
# Prospective continuation follow-up additions (2026-10-07)

The existing opt-in single-intervention hook is reused without changing its
navigation or action construction logic. No tracked navigation/controller core
file is changed in this cycle. New research-only tools are:

- `prepare_continuation_sample.py`: metadata-only sampling excluding prior
  training diagnostic scenes, with route/validation-disjointness checks.
- `prepare_continuation_schedule.py`: freeze one typed H=1 event per eligible
  route before altered continuation outcomes.
- `continuation_labels.py`, `analyze_prospective_continuation.py`: registered H=2
  future-outcome diagnostic, with full-cost and terminal-window accounting.
- `run_continuation_cycle.py`: gated route audit, disabled/enabled runs,
  independent reconstruction, then conditional mixed-schedule verification.
- `decompose_continuation_costs.py`: post-hoc immediate/later-movement/STOP
  accounting tied exactly to native episode metrics.
- `analyze_equal_primitive_budget.py`: post-hoc offline navmesh queries on
  retained primitive poses, requiring native geodesic calibration first.
- `plot_continuation_diagnostics.py`: all-five-case descriptive curves.

`audit_single_intervention.py` now derives population/group sizes from the
frozen schedule/config rather than assuming seven treated / 57 control routes.
Its original seven-route results and gates were re-audited with exact equality;
prior source snapshots/results are preserved. Empty groups are explicitly absent
performance estimates, never zero-score evidence. None of these tools is an
inference-time abstraction controller or reads privileged values into the native
model. Tests cover sampling exclusions, event selection, variable group sizes,
cost gates, future-label isolation, STOP windows and the existing action/mask hook.


## Offline full-return feature audit additions (2026-10-07)

`research/tools/audit_full_return_features.py` joins pre-decision graph fields
with accepted native full-return outcomes by scene/episode/step and exact action
dictionary. Extracted fields are logit gap, mask-aware normalized entropy,
commanded graph polyline delta, back-path node count delta and current-proposal
flag. The polyline uses agent-known graph geometry and is not simulator geodesic
distance or a realized controller trajectory. Privileged outcomes never enter
the feature extractor. `audit_full_return_feature_cost.py` reuses existing
cost breakdown code to separate immediate, later navigation and terminal costs.
Both tools are offline; no new abstraction implementation or training path was
added. Five feature-contract tests cover leakage, geometry and mask handling.


## Native STOP diagnostic clarification (2026-10-07)

At `ss_trainer_ETP.py:1751`, each current graph node stores its STOP probability.
At `ss_trainer_ETP.py:1814`, a STOP command selects the historical node with
maximum stored probability and executes its back-path before primitive STOP.
Therefore a current-pose goal-distance test is not a faithful STOP execution
oracle. Native final and oracle success at lines 1996-1997 use **<= 3 m**;
older endpoint-only census used < 3 m, with no exact-boundary case in the present
STOP audit. Research-only `audit_native_stop_feasibility.py` reads actual saved
index-0 branches. `localize_primitive_arrivals.py` loads the matching split's
dataset goals and navmeshes to locate intermediate arrivals without executing
new actions. These tools add no deployment feature or new action mode.
