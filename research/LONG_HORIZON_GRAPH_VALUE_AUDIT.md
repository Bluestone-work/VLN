# Long-Horizon Graph Value Audit

Date: 2026-10-09  
Branch: `research/adaptive-action-audit-20261007`

## Confirmed conclusions

The current evidence supports studying native graph-action choice, not waypoint
density adaptation or raw-candidate ranking. Raw candidates are not the action
executed by ETPNav after ghost merging, historical-node routing, controller
quantisation and STOP handling. One-step goal progress is therefore not a
ground-truth value label.

The accepted full-return studies show that alternatives in the native action
set can rescue some failures, but the signal is not yet a deployable result.
The fresh critical cohort reported 0.9386 in-cohort strict-pair accuracy versus
0.8520 for native logit, while a cross-cohort transfer check was 0.8421 for
both. The later 96-route intervention improved point-estimate SR by 3.125 pp,
but scene-cluster intervals crossed zero for SPL, nDTW and primitive cost; the
prospective replication failed its cost gate. These are feasibility/causal
diagnostics, not a generalization claim.

## Closed directions

Coarse/default/fine waypoint selection, adaptive NMS, raw-candidate rankers,
one-step progress labels, collision interruption policies, direct PPO/GRPO,
and threshold/horizon/controller/evaluator changes are closed for this cycle.

## Open question

Before adding g3D-LF or RL, test whether a small residual ranker can predict
full-return preferences from features available at the decision, on a frozen
scene-disjoint split, while preserving native actions by default.

## Infrastructure audit

The repository contains a native graph-option enumerator and option capture in
`vlnce_baselines/adaptive_action/graph_option_capture.py`, plus isolated native
branch execution in `research/tools/probe_graph_options.py`. Accepted captures
include current ghosts, historical ghosts and STOP, retain typed action identity,
pre-pose/RNG sentinels, controller primitives, collisions, endpoint and raw
distance metrics, and pass independent coverage/non-interference audits.

This is sufficient to generate a pilot and to generate full-return branches
for registered states. It is not by itself sufficient for the final dataset:
existing cohorts were often failure-selected or outcome-inspected, and the
stored analysis rows discard some state/action features and complete episode
metrics. A new serializer must retain raw branch traces and raw episode returns
without replacing prior artifacts.

## Remaining risks and missing fields

* A prospective dataset must freeze route/scene sampling and critical-state
  predicates before continuation outcomes are read.
* Every state must preserve the complete admissible native set, including
  historical ghosts and STOP, with stable action identity and the native
  selected action.
* Each action needs complete continuation metrics: success, final goal
  distance, SPL, nDTW, SDTW, path length, primitive count, collisions,
  high-level decisions, forced/active STOP reason, backtracking cost and the
  selected/alternative identity.
* The deployable feature view must be separate from raw outcomes and must reject
  goal/reference geometry, future collisions, final metrics and continuation
  fields.
* State restoration is faithful for isolated options when the accepted
  pre-pose, RNG, action, sensor and prefix audits pass. It is not a license to
  treat direct probes as counterfactual policy trees: only one action is
  replaced, then native navigation resumes.

## Compute estimate

The existing one-worker RTX 4090 captures executed roughly 500--1,000 native
branches per diagnostic cohort in hours, with tens to hundreds of thousands of
primitive events. A 16-route pilot is practical. A scene-balanced first
dataset of roughly 32--64 routes and all registered critical states should be
budgeted at one to several GPU-days, depending on branch count and simulator
startup. No long run is justified until the pilot passes.

## Audit decision

Proceed only with the frozen protocol and the small pilot. Do not fit a model
or claim feasibility from the historical in-cohort numbers until scene-disjoint
data, raw outcome retention, action identity checks and feature leakage checks
all pass.
