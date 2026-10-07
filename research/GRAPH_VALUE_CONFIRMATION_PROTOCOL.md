# Independent graph-value confirmation protocol

Registered 2026-10-07 after `CRITICAL-GRAPH-VALUE-FRESH-001`. This is a
training-only development experiment. It is not a benchmark evaluation and it
must not use R2R val_unseen or RxR labels.

## Scene-pool audit result

The current R2R train source was audited before sampling. The registered
exclusions cover 59 of 61 training scenes, leaving only two scenes and four
routes. This is below the six-scene/96-route gate, so no confirmation dataset
was written and no outcome was inspected. The audit is recorded in
`research/results/graph_value_confirmation_train96/scene_pool_audit_001.json`.
The protocol remains registered, but execution is **NO-GO with the current
scene pool**. It can resume only after adding a genuinely new data source or
explicitly changing the protocol to a non-independent study.

## Population

Sample at least 96 training routes from at least six scenes by metadata before
reading any full-return outcome. Exclude every scene and trajectory used by the
graph-option, continuation, full-return, critical-census and ranker-diagnostic
cohorts listed in `GRAPH_VALUE_CONFIRMATION_001.json`. Keep the scene groups
fixed before feature extraction and outcome reading.

## Frozen decision protocol

Capture the native graph state and enumerate every admissible native graph
action at registered critical states. Execute each action from the exact state
through the unchanged controller and navigator until native STOP or the normal
15-decision limit. Preserve action identity, graph masks, sensor settings,
primitive costs, collisions, RNG and termination semantics. Run selected-action
controls and independent prefix checks before reading outcomes.

## Frozen features and utility

Use schema version 2 from `graph_value_feasibility.py`: graph logit/rank,
policy entropy, state STOP probability, candidate count, STOP/current-proposal
flags, ghost distance, current-to-first plus within-path back-path length,
back-path node count and existing embedding summary statistics. Do not use goal
distance, reference route, oracle arrival, future collision, full-return
metrics or continuation outcomes as features.

Pairwise preference uses strict dominance on success, SPL, nDTW, goal distance,
path length and primitive count. Mixed/tied pairs are retained as unresolved.
Fit a frozen linear pairwise baseline only on training scenes. Compare it with
native graph logit and a seeded random admissible-action selector offline,
then execute the pre-registered selected alternatives; no policy retraining.

## Gate

Require route-quality rescue (SR with nondecreasing nDTW) in at least two
independent scenes, native STOP preservation, no primitive-budget inflation
outside the predeclared cost analysis, and scene-cluster confidence intervals.
If this fails, stop graph-value fitting and investigate reachable proposal
coverage with a separately registered oracle. Do not add RL/PPO, VLM or an
interrupt policy before this gate.
