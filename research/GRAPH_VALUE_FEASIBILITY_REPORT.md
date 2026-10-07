# Graph-value / preference feasibility report

## 2026-10-08 — training-support audit completed: metric mismatch

**NO-GO remains for the frozen strict-pair linear scorer plus first-disagreement
recipe.** On its own 47-state / 996-action fitting census, all-pair accuracy is
94.34% versus native logit 85.20%, but native-versus-move accuracy is
403/418 = 96.41%, below logit's 404/418 = 96.65%. The 518 extra correct pairs
come from +404 between non-native moves and +115 involving STOP, offset by −1
in native-relative moves. Only 7.37% of strict fitting pairs are native-move pairs.

A post hoc directional split finds 404 native-better and 14 alternative-better
strict move comparisons. The model recognizes **0/14** native improvements,
even in-sample. Ten protected STOP states remain unchanged; among 37 movable
states it abstains 35 times and overrides twice. One override strictly degrades
return (+59 primitives); the other rescues SR but loses 9.175 nDTW points and
adds 13 primitives. All 47 native continuations are failures, so this fitting
population cannot estimate override harm on already successful native routes.

47 states, 996 actions and 5,668 strict pairs reproduce exactly. Six tests pass;
an independent computational verifier passes 12,159 pair labels, 47 masked
choices and 564 metric components. No model was refit, no new rollout occurred,
and prospective replication outcomes were not read. This is an in-sample,
correlated-state postmortem, not held-out or episode-policy performance.

Retire all-pair accuracy as a sufficient gate. A future proposal needs native-
relative full-return decision evidence, abstention and successful-route controls;
these observations alone do not authorize a new loss, RL/VLM, interrupt policy
or waypoint predictor. Report: `PREFERENCE_TRAINING_SUPPORT_REPORT.md`.
Earlier protocol/config status remains the immutable pre-run specification;
`results/preference_training_support_001/status.json` records completion.

## 2026-10-08 — continuation mechanism audit completed

The frozen recipe remains **NO-GO**. A retrospective audit retains all 27 prior
FULL and 29 replication FULL events, plus 97 overlapping COST control events.
All 153 prefix/action and cost/collision/path checks pass; four diagnostic tests
pass. No model or simulator rollout was added.

Immediate-cheaper/episode-costlier reversals occur in 10/27 and 14/29 FULL events.
Only 2/10 and 2/14 show immediate replacement execution symptoms; stable next
reselection of the displaced target occurs in 4/10 and 7/14, also appearing in
other events. Neither observation establishes a majority causal mechanism or
an interrupt remedy. These are selected replacement actions, not a census of
all native-agent execution failures.

All 56 FULL alternatives are current proposals without backtracking; 53 have
shorter final target segments. Frozen-score attribution gives execution geometry
the largest positive contribution in 49/56 choices and 23/24 cost reversals.
This is algebraic explanation, not a causal ablation or reliable failure detector.
The earlier −161 total primitive result includes −155 from route 10199; another
route avoids 91 primitives of native budget-STOP return. In replication, later
navigation adds 200 primitives and STOP return adds 54 after immediate −133.

No structural filter is selected or trained. Next: the specified old-development
training-support audit checks whether strict-pair accuracy represents actual
native-relative decisions, retaining mixed/tied returns and STOP distinctions.
It has not run and cannot overturn the prospective gate using in-sample data.
Report: `PREFERENCE_CONTINUATION_MECHANISM_REPORT.md`; next specification:
`PREFERENCE_TRAINING_SUPPORT_PROTOCOL.md`.

## 2026-10-08 — prospective replication: NO-GO for scaling this recipe

The preregistered 96-route / 8-scene follow-up is complete. FULL changes 29
routes, rescues four and loses one: SR/SPL/nDTW changes are
**+3.125/+1.771/+0.938 pp**, but primitives increase **+1.260/route**.
This fails the frozen no-cost-increase gate. All four primary scene-bootstrap
intervals include zero (SR: −2.083 to +8.333 pp). The conclusion is **NO-GO
for scaling the frozen first-disagreement recipe**, not rejection of all graph
value research or evidence of proposal/execution failure dominance.

COST-own matches FULL SR with only graph logit/rank and execution-distance
features, but costs +3.427 primitives/route. At FULL's timing, COST-matched
achieves +2.083 pp SR; FULL's extra +1.042 pp has CI [0, +3.125], lower mean
nDTW and higher cost. The added-feature gate fails. Random SR changes are
0/−1.042/−1.042 pp, with worse SPL/nDTW/cost. Beating random is insufficient.

Post hoc accounting: FULL's changed options save 133 primitives, but remaining
continuations add 254, leaving +121. Fourteen routes have a cheaper option and
a more expensive complete episode. All six arm fidelity/metric audits pass;
838 identical-action cross-arm comparisons agree exactly. A pre-episode worker
startup failure was preserved and retried under a versioned operational amendment.

The sample is disjoint from declared prior research routes, with overlapping
baseline-training scenes, one simulator seed and an offline one-action harness.
Do not retune on this cohort, pool it with the exploratory positive cohort to
claim a passed gate, or expand RL/VLM/interrupt/waypoint-density training.
Next diagnostic question: can pre-action evidence distinguish harmful early
substitutions when the target includes full continuation cost? Include native
success routes and abstention; register separately before any further learning.
Full report: `GRAPH_VALUE_PROSPECTIVE_REPLICATION_REPORT.md`.
Earlier decisions below are chronological evidence, superseded for this recipe.

## 2026-10-07 — route-disjoint execution update

**CONDITIONAL GO for a prospective replication**, not a benchmark improvement.
The invalid first 96-route sample had 12 historical diagnostic overlaps and is
retained only as a pilot. The replacement 96-route / 8-scene training sample
passes the declared-route exclusion audit. Its complete census has 561 native
full returns at 57 critical states: 4/7 SR rescue opportunities, 3/7 also preserve
nDTW, 0/4 interrupt rescues, three unresolved routes. Independently audited.

A scorer trained only on the prior fresh 47-state census was actually executed
with one first-disagreement action per route: 27 changes, 3 rescues, 0 success
losses across all 96 routes. SR/SPL/nDTW change = **+3.125/+2.302/+0.956 pp**;
primitive count **−1.677/route**. Three matched random seeds have SR changes
0/−1.042/0 pp and all worsen mean SPL/nDTW/cost. All disabled/enabled and metric
reconstruction gates pass. SPL/nDTW/cost confidence intervals vs native still
cross zero; 13 treated routes lose nDTW. This is scene-overlapping, one-seed,
exploratory training follow-up after target census inspection, not untouched
validation. Late critical-state non-rescue missed an earlier step-3 rescue on
route 3161. Next: unchanged frozen scorer vs a simple cost-aware scorer on a
new prospectively frozen route sample; no adaptive density, RL, VLM or interrupt
policy training. Full evidence: `GRAPH_VALUE_ROUTE_EXECUTION_REPORT.md`;
protocol: `GRAPH_VALUE_ROUTE_INTERVENTION_PROTOCOL.md`; next gate:
`GRAPH_VALUE_ROUTE_NEXT_GATE.md`.

## Decision

**CONDITIONAL GO for a separately registered held-out confirmation.** The
failure-critical full-return census supports continuing a narrow,
execution-aware long-horizon graph-choice study. It does not support inserting
the scorer into navigation, training RL, or adding a VLM.

## Fresh development cohort

`CRITICAL-GRAPH-VALUE-FRESH-001` was registered from 64 routes in 16 training
scenes after excluding prior graph-option, continuation, full-return and
selector-diagnostic scenes. Ten baseline failures in six scenes were frozen by
metadata before their full-return outcomes were read. The failure-critical
subset contains 47 states and 996 admissible native graph actions. Each action
was executed from the exact frozen state, followed by the unchanged navigator,
controller, native STOP logic and 15-decision limit.

All 996/996 action cases completed. Ten controls and four interrupt routes were
checked independently. The accepted run has 8,144 metric reconstructions; the
independent audit has 10,868 exact option/prefix checks, 144,979 primitive
records and eight interrupt prefix/cost checks. Every validity gate passed.
The first worker-reset attempt (`full_001`) produced no action cases and is
preserved as a failed attempt; it is excluded from all denominators.

| Outcome opportunity | Routes | Scenes | Denominator |
| --- | ---: | ---: | ---: |
| Any non-STOP graph-choice rescue | 9 | 6 | 10 |
| Rescue with SR and nondecreasing nDTW | 7 | 5 | 10 |
| Termination rescue | 6 | 4 | 10 |
| Unresolved after tested graph/termination actions | 1 | 1 | 10 |
| Interrupt SR rescue | 2 | 2 | 4 tested routes |

The rows overlap. They are opportunity counts, not a causal partition. In the
two interrupt rescues, route 3865 gains SR but loses 17.898 nDTW points and
uses 97 more primitive actions; route 4702 is the only route with a clean
SR+nDTW rescue (+3.397 nDTW points, +9 primitives). Routes 10506 and 10754
are not rescued. Interrupt therefore remains a secondary diagnostic.

## Frozen feature feasibility

The outcome-blind feature contract uses only fields available at the decision:
graph logit and rank, policy entropy, STOP probability, candidate count,
STOP/current-proposal flags, ghost distance, back-path geometry and size, and
existing candidate embedding summary statistics. Goal/reference geometry,
future collisions, oracle arrivals and full-return outcomes are excluded.

Strict dominance pairs require simultaneous improvement or equality on success,
SPL and nDTW, and no increase in goal distance, path length or primitive cost.
Mixed and tied alternatives are retained as unsupported cases and are not
forced into a scalar label.

The corrected feature implementation produces 5,668 strict pairs across all
six scenes. Leave-one-scene-out pair accuracy is 0.9386 for the frozen linear
pairwise model versus 0.8520 for native graph logit. The earlier feasibility
JSON used an indexing bug for the state STOP probability and omitted the first
segment of back-path cost; it is retained as `feasibility_001.json` for audit,
while `feasibility_002.json` is the corrected analysis. Unit tests now cover
both cases. This correction does not change the full-return census, rescue
counts or any navigation rollout.

The corrected result remains feasibility evidence only:

- predicted actions were not executed;
- one random seed and one retrospective failure cohort were used;
- strict dominance discards mixed outcomes;
- scene folds with few route-quality opportunities remain weak;
- no benchmark SR/SPL/nDTW gain is claimed.

## Independent transfer check

The earlier independent full-return training cohort remains a separate
confirmation of signal, with 16 states across eight scenes and 19 strict pairs.
Using the frozen feature contract, pair accuracy is 0.8947 for the learned
model versus 0.8421 for native logit (`transfer_003.json`). This is also not a
deployable result because only two alternatives per state were executed and
the predictor was not put in the navigation loop.

As a stricter cross-cohort check, training the corrected model on the fresh
47-state cohort and transferring it to those same 16 confirmation states gives
0.8421 for both the learned scorer and native logit (`transfer_002.json`). This
null transfer result is evidence against treating the fresh in-cohort pair
accuracy as a generalization result. The old train-new-to-train16 transfer is
retained as a feasibility signal, while the independent confirmation remains
mandatory.

## Next gate

Register a new training cohort before reading outcomes, with at least six
scenes and a scene-grouped split. Freeze the corrected feature schema and
utility definition, execute every admissible action at registered critical
states, and evaluate a frozen ranker against native logit and random choice
with no navigator retraining. Require route-level SR rescue with nDTW
nondegradation in at least two independent scenes, native STOP and primitive
budget preservation, and confidence intervals over scenes. If this gate fails,
stop graph-value fitting and investigate reachable proposal coverage. RL/PPO,
VLM assistance and interrupt-policy learning remain paused.

## Confirmation availability audit

The proposed confirmation was audited before sampling. The exact local R2R
train source has 61 scenes, while the registered exclusions cover 59. Only two
scenes and four routes remain, so the six-scene/96-route protocol cannot be
executed without reusing covered scenes. RxR train has the same MP3D scene set;
its additional language annotations do not create scene-independent evidence.
The sampling attempt wrote no dataset and read no outcome labels. See
`research/results/graph_value_confirmation_train96/scene_pool_audit_001.json`.

This changes the operational status of the next gate to **NO-GO with the
current local data pool**, while leaving the fresh-cohort feasibility decision
as CONDITIONAL GO. A new data source or an explicitly weaker episode-level
overlap protocol is required before any learned ranker can be evaluated.

## Reproducibility

Primary records:

- `research/configs/GRAPH_VALUE_FRESH_TRAIN64_001.json`
- `research/configs/CRITICAL_GRAPH_VALUE_FRESH_001.json`
- `research/results/critical_graph_value_fresh/plan_001.json`
- `research/results/critical_graph_value_fresh/full_002/results.jsonl`
- `research/results/critical_graph_value_fresh/analysis_001/summary.json`
- `research/results/critical_graph_value_fresh/feasibility_002.json`
- `research/results/critical_graph_value_fresh/independent_audit_001/summary.json`
- `research/results/critical_graph_value_fresh/provenance_001.json`
