# Controller-consistent graph-option labels: executed protocol

Date: 2026-10-07. Status: training pilot and unseen measurement completed;
ranker training has not run. This is a separately scoped graph-decision diagnostic,
not a positive result for Adaptive Action Abstraction.

## Question and prerequisite

Does a better graph decision remain available when every label refers to the
actual merged ghost, front/back path and primitive controller? The selected
option must first reproduce in a separate simulator. OPTION-CALIBRATION-001
passes on 511 selected options; OPTION-CALIBRATION-002 passes on another 600
across all 11 unseen scenes (`OPTION_CALIBRATION_REPORT.md`). Passing selected replay does not
validate unexecuted alternatives.

## Unit of comparison

One row is `(scene_id, episode_id, high_level_step, graph_action_id)`, captured
after graph construction/navigation logits and **before selected-ghost deletion**.
Use actual valid graph actions, including historical ghosts. Multiple raw
waypoints mapped to the same ghost are one executable action, never independent
training pairs with contradictory labels.

For each state retain:

- Pose, simulator/controller configuration and typed Python/NumPy RNG state.
- Ordered graph IDs, validity/visited masks, finite logits and selected index.
- Merged target, exact front node and shortest back path from the current node.
- The complete typed controller action dictionary; retain float32/float64 arrays.
- Existing frozen graph/instruction features if available, separate from labels.
- Learned STOP decision, forced-stop reason and effective episode limit.

The STOP option follows the baseline's highest historical STOP-score node and
its back path. A literal STOP at the current pose is a different intervention.
Separate rank-only comparisons preserve the learned/forced STOP gate.

## Isolated execution and guards

1. Capture the frozen baseline without online probes. Verify per-episode metrics
   against an uninstrumented control with identical checkpoint and episode IDs.
2. In another process, reset episode/task, restore recorded pose and controller
   RNG, and execute the original `VLNCEDaggerEnv.step` action.
3. Include the actually selected option as a sentinel in every captured state.
   Require the existing 1e-5 m/rad tolerances, identical primitive/collision
   sequences, RNG and terminal flag. Report observation hashes separately.
4. Execute every valid non-STOP graph option with the same restored RNG state;
   record incomplete/invalid branches explicitly. Do not impute zero progress.
5. On a preselected small subset, reverse alternative enumeration order and
   repeat each action. Outcomes must agree; this detects leaked simulator state.
6. Compare masked action IDs and serialized selected actions against the native
   selection path. Variable action count, visited masks, STOP and back-path cases
   require focused tests before scaling collection.

The current baseline has `ALLOW_SLIDING=True`, so `IL.tryout and not
ALLOW_SLIDING` is false. Noise, video mode and stochastic tryout require separate
calibration. Do not enable them to improve label quality in this protocol.

Implementation and completed checks are in `GRAPH_OPTION_REPORT.md`. The pilot
preserves the final-decision action budget explicitly: non-STOP options that
would be converted to forced STOP are not executed as candidate alternatives.
The capture stores features, but no fitting uses either split in this cycle.

Completed results: 760 train branches and 7,113 unseen branches, all selected
sentinel and reversed-order checks pass. Joint primitive/path caps reduce the
unseen mean local opportunity to 0.148787 m, versus zero in the small training
pilot. Interpretation, limitations and the next gate are in `GRAPH_OPTION_REPORT.md`.

## Labels and interpretation

Privileged goal geodesic progress is the primary **diagnostic** label, not an
input to a deployable scorer. Save endpoint, actual movement path length,
primitive count, per-primitive collision, and target residual alongside it.
Goal progress can penalize necessary detours/backtracking; report reference-route
progress only with an explicit monotonic alignment rule, and do not treat nearest
reference-point distance alone as route completion.

Report the selected action versus the best **executed** alternative on identical
states, including ties, candidate coverage and state/scene distributions. This
is a local privileged regret estimate, not an end-to-end optimal upper bound.
Collision and target miss are observable symptoms, not proof of a causal failure.

## Sampling and controls

- First pilot: eight preselected training episodes spanning four training scenes,
  one instruction per route; no validation-driven fitting or tuning.
- Validation measurement: frozen scene-balanced R2R val_unseen manifest from
  `results/option_calibration/unseen_balanced66_sampling/manifest.json`, all
  11 scenes, six distinct routes/scene, 66 episodes, zero train-route overlap.
  This subset is a diagnostic distribution, not the standard benchmark score.
- Keep the val_seen64 calibration sample for regression checks, not selection
  of hyperparameters. Archive any added training scene/route manifest first.
- Preserve released checkpoint, sensors, controller, SLIDING, maximum decisions,
  success threshold and evaluator. Do not duplicate the repository.
- Record git commit plus changed-source hashes, configuration, dataset/checkpoint
  hashes, hardware, seed, stdout/stderr and an exclusive output directory.

## Decision gate

If selected sentinels or alternative-order repeatability fail, repair measurement
before collecting more labels. If controller-consistent regret is negligible,
stop the graph-ranking pivot. If it is substantial, test a simple frozen-feature
ranker only after label coverage and actual continuation pass their additional
gates, with episode/route/scene-cluster uncertainty. Local regret alone does not
satisfy those gates.
Neither local oracle regret nor a single validation seed justifies deployment.
Require fair held-out trajectory improvement, then at least three training seeds
when feasible, before joint fine-tuning. No RL or VLM is scheduled at this gate.

## Completed ordered-route follow-up (2026-10-07)

Training64 expands coverage to 16 scenes and 64 new routes, with 5,090 full
branches and all fidelity gates passing. Joint-cost-capped local gain is
0.053452 m on 410 non-STOP states; the frozen matched-prefix route gate leaves
0.042087 m and 14 material states in only seven routes/five scenes. Unseen66
retains 0.126962 m and 49 material states; pilot8 remains zero. See
`ROUTE_COMPATIBILITY_PROTOCOL.md` and `ROUTE_COMPATIBILITY_REPORT.md`.

These labels remain privileged targets/diagnostics. Actual counterfactual
primitive/path costs may not be used to filter a deployed action set. The next
continuation intervention has now completed; its result is summarized below.

## Completed native-continuation gate (2026-10-07)

`GRAPH-SINGLE-INTERVENTION-TRAIN7-001` changes one action on each of seven frozen
training routes, preserving the other 57 as exact controls. All fidelity checks
and the independent 384-metric reconstruction pass. On the seven treated routes,
SPL increases 5.5570 pp and nDTW 4.2386 pp, but success remains 4/7, two failures
deteriorate, mean goal error rises 0.6155 m and total primitives rise 674 -> 711.
All immediate options satisfy the cost caps and collectively save 17 primitives;
later continuation accounts for the reversal. See `SINGLE_INTERVENTION_REPORT.md`.

**Current labels do not pass the gate for training/integrating a scorer.** Some
trajectory-efficiency signal remains, so a narrower diagnostic is CONDITIONAL
GO. Prospectively compare one-step labels against short native-continuation
labels on new training routes, recording all final outcomes and total costs.
H=2 exposes the two harmful reversals in this sample, but this is post-hoc and
must not be promoted to a validated filter. New samples/thresholds must be frozen
before their outcomes; no such new experiment is claimed here. Original NMS
adaptive abstraction remains NO-GO; no RL, VLM or new action representation is added.

## Prospective short-window validation completed (2026-10-07)

`CONTINUATION_LABEL_REPORT.md` now completes that prospective test: 64 routes in
16 new-to-diagnostics training scenes, excluding all 19 prior training scenes.
Only five routes / three scenes have material route-compatible local options.
All five were successful baseline routes. H=1 interventions save 214 primitives
in aggregate, but one route saves 244; the other four together add 30.
H=2 rejects the adverse route and two useful routes, retains two interventions
with zero aggregate primitive saving, and is verified in a separate mixed run.
The later equal-primitive check still misses delayed benefits and admits an old
harmful case. No short-window label is promoted to a reliable training target.

The next design must select states/candidates without local oracle-gain screening
and record full native-continuation outcome vectors. This changes the diagnostic
sampling question, not the baseline policy or success definition. See
`FULL_RETURN_LABEL_NEXT_GATE.md`; no such new sample/model is claimed executed.
