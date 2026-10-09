# Full-return Graph Preference Protocol

Registered: 2026-10-09  
Branch: `research/adaptive-action-audit-20261007`

This protocol freezes the first feasibility experiment. Existing validation
outcomes are not used to select features, routes or thresholds.

## Unit and sampling

The unit is one native ETPNav graph decision state. Training routes are selected
metadata-only, route-disjoint and scene-balanced from R2R train. Validation is
scene-disjoint with at least four scenes and no route/instruction overlap. A
route is never split across train and validation. Previously inspected
validation and diagnostic routes are excluded and their exclusion manifest is
stored. Critical states are the registered union of collision-related state,
endpoint deviation, forward stall, last non-STOP action, terminal STOP, high
policy uncertainty/small top-1/top-2 margin and revisit/historical-ghost
decision. The first execution uses only predicates already present in the
registered source cohort; uncertainty/revisit predicates are recorded only
when their native trace fields exist, never added post hoc to increase yield.

## Native actions and continuation

For each state, enumerate exactly the admissible native mask from the current
ETPNav graph: current ghosts, historical ghosts and STOP. Deduplicate only by
the native graph action identity used by the policy; do not merge raw waypoint
proposals. Execute each action from the frozen prefix with the native graph
update, controller, sensors, RNG, STOP mechanism and unchanged 15-decision
horizon. Replace one action, then resume the original navigator until native
termination. The native selected action is always included as a sentinel.

Each branch stores raw per-step continuation and final metrics, action identity,
selected/alternative marker, forced versus active STOP, collision count,
primitive count, high-level decisions, path/backtracking cost and termination
reason. Raw outcomes remain available if the preference rule changes.

## Feature contract

Allowed inputs are current instruction/ETP language representation, current
panoramic and graph/history representations, recurrent state, entropy and STOP
probability, native graph logit/rank, node/ghost embeddings, current versus
historical type, bearing/distance, visited/revisit counts, graph path/back-path
cost, requested native polyline cost and current-observation geometry.

Forbidden inputs are goal/reference/future geometry, GT shortest path, future
collision, final success/SPL/nDTW/SDTW, continuation return, oracle arrival,
future landmark progression and any simulator signal unavailable at the state.
The serializer must fail if a forbidden key enters the deployable feature view.

## Preference rule

Preferences are stored as pairwise relations and raw metrics, not a fixed scalar
reward. `a_i` dominates `a_j` only under this lexicographic rule: (1) success
beats failure; (2) if success agrees, higher nDTW/SDTW wins when the difference
exceeds 0.01 absolute; (3) if both trajectory scores are within that deadband,
lower primitive count wins when the difference exceeds 2 primitives; (4) lower
SPL/path cost breaks a remaining tie only beyond 0.01 absolute. Otherwise the
pair is marked `mixed_or_tie` and excluded from strict training pairs. All raw
metrics are retained.

## Baselines and evaluation

B0 is native graph-logit ordering. B1 is a linear/ridge pairwise ranker over
the frozen deployable handcrafted features (distance, bearing, visited,
current/historical, entropy, STOP and back-path cost). B2 is a small residual
MLP over existing graph/action embeddings with `Q=logit+alpha*delta`, bounded
alpha and no full-policy replacement. Fit only on training scenes, freeze
preprocessing/weights, then evaluate validation scenes.

Report pairwise accuracy, top-1, NDCG/rank correlation, regret, rescue
preference recall, harmful-replacement rate and calibration, each per scene.
Bootstrap whole routes and whole episodes separately; report scene-level
intervals and never individual-decision bootstrap as the only uncertainty.

## Online gate

Only after offline validation passes may a critical-state gated intervention run:
native action remains default unless the state is registered critical and the
learned margin over native exceeds a frozen threshold. Report intervention and
changed-episode rates, beneficial/neutral/harmful interventions, SR/SPL/nDTW,
primitive count, collisions and STOP behavior with paired route and scene CIs.

## GO / NO-GO

GO requires stable scene-disjoint improvement over B0/B1, nonzero rescue
preference recall, confidence gating that controls harmful replacements,
positive small-scale intervention trends, multi-scene support and no identity
bug for historical ghosts or STOP. Any in-sample-only gain, unstable unseen
scene result, frequent native-success damage, privileged leakage, route
concentration or uninformative confidence is NO-GO. A NO-GO stops model scaling
and g3D-LF/RL work.
