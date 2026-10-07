# Prospective short-continuation label validation

Date: 2026-10-07. Experiment: CONTINUATION-LABEL-TRAIN64-001.
Status at registration: not executed. This file and its config are frozen before
sampling outcomes. It follows the seven-route exploratory intervention study;
the original NMS-density AAA decision remains NO-GO.

## Question

Does requiring a promising one-step alternative to remain useful after two
native high-level decisions reject harmful interventions on new training data?
This tests an **H=2 confirmation filter on an H=1 candidate**, not exhaustive
H=2 action search, a new abstraction policy, or a learned/deployable model.

## Sampling and controls

Select 64 routes, four per each of 16 training scenes, by seeded SHA256 metadata
order. Exclude all scenes used by train8/train64 full-option studies and earlier
training oracle-extraction records. Use one instruction per route; verify zero
route overlap with validation. Missing assets are blockers, never grounds for
outcome-driven substitution. Keep all sampled routes in original simulator order.
No resampling if positive opportunities are rare.

Reuse released checkpoint, seed 100, sensors, controller, sliding=true,
effective tryout=false, learned/forced STOP, 3 m success criterion and 15-decision
horizon. First collect baseline and untraced control, then require complete-option
selected sentinels, reversed enumeration and independent integrity checks.
Use the existing ordered-reference gate and numerical tolerances unchanged.

## One-step candidate and intervention schedule

At every baseline non-STOP state, use the existing best full option under both
primitive/path caps and matched-prefix reference gate. A material opportunity
requires >=0.25 m more goal progress than the original option. Keep the existing
tie-breaking rule. For each eligible route choose **one** eligible state by
SHA256(20261007|scene|episode|step); include every eligible route. Freeze the exact
typed action and graph identities before any altered native continuation runs.
Routes without a material state remain controls; do not silently replace them.
Zero eligible routes ends this sample with no learning justified.

Run the interceptor disabled and require exact traces/metrics. Then apply each
frozen replacement once and let the native policy continue through STOP without
teleportation, replaying an old suffix, extending the horizon, or changing masks.
Verify every prefix, isolated alternative execution and unaffected route.

## Frozen H=2 confirmation rule

H=1 contains the replacement; H=2 contains the replacement plus one subsequent
native decision. Compare against the original policy's matching decision window.
STOP is absorbing for offline comparison only; no padding actions are executed.
Accept the H=1 candidate at H=2 only if all three conditions hold:

1. relative goal progress at H=2 is >=0.25 m;
2. total H=2 primitive count is no greater than the paired baseline;
3. total H=2 movement path is no longer than baseline +1e-5 m.

The H=1 route gate remains mandatory; H=2 itself is not a semantic certificate.
Window costs include any native STOP return path. Count actual decisions and
remaining horizon; fewer remaining actions are not hidden. H=3 is descriptive
sensitivity only and cannot replace the registered primary rule.

All horizon inputs are privileged future outcomes used only to assess labels.
Do not pass them as deployed features, candidate masks or controller inputs.

## Outcomes and denominators

Retain every eligible event, including rejection, ties, success losses and cost
increases. Report final SR/SPL/nDTW/SDTW, goal error, high-level decisions,
primitive/collision counts and path length, both over treated routes and all 64.
Tabulate accepted/rejected H=2 groups and all paired episodes. Report how many
events terminate within H=2; also describe final changes strictly beyond that
window to expose mechanical overlap of label and endpoint metrics.

Compare baseline, all H=1 interventions, and H=2-confirmed interventions. For the
third result, selecting original versus intervened complete trajectories per
route is an **offline privileged composition** until verified by a separate
mixed-schedule rollout. Never present this composition as an executed model.
If H=2 accepts no events or all events, report that fact; do not tune its threshold.

Report route and scene coverage, including rejected routes. With at least eight
treated scenes use 5,000 paired scene bootstrap samples (seed 20261007), retaining
all routes within a sampled scene. Smaller samples get descriptive ranges and
counts, not significance claims. One simulator seed; no learned-model seed claim.

## Stop/go and automatic next step

- Failed fidelity gates: repair measurement before interpreting outcomes.
- Sparse support (fewer than eight treated scenes): no scorer training; preserve
  this sample and report the limitation without favorable resampling.
- H=2 confirms actions that still lose success or worsen total primitive cost:
  the filter is insufficient; diagnose later continuation rather than add capacity.
- H=2 rejects every action or removes all efficiency gains: do not train this rule.
- Only if confirmed actions retain efficiency gains, no success loss and no
  aggregate primitive increase should a separate mixed-schedule rollout verify
  the composed result. That is a further diagnostic gate, not permission for RL/VLM.

Save immutable configs/source snapshots, hashes, seed, hardware, git state,
stdout/stderr, machine-readable outputs and the experiment table. Distinguish
completed stages from planned ones; never overwrite accepted prior artifacts.
