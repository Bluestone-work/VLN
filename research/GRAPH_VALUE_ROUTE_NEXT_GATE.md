# Next gate after route-disjoint execution diagnostic

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

Do not reopen adaptive coarse/default/fine, train an interrupt policy, add PPO,
or add a VLM. This study tests native graph choice only.

The next informative study is a prospectively frozen replication with a simple
geometry/cost control. The existing target results have already been inspected;
reusing them to choose an intervention rule or coefficients would be exploratory.
A new target sample must exclude all previous diagnostic routes and both 96-route
samples, keeping scene overlap explicit. It is still not a baseline-training
holdout. Freeze all coefficients, timing rules and random seeds before its native
capture and continuation, and preserve all successes in the denominator.

Use the same 47-state/996-action fresh training corpus only. Freeze: (1) the
current schema-v2 ridge-10 preference scorer unchanged; (2) a reduced scorer
using graph logit/rank plus commanded back-path/ghost distances, fitted with the
same strict-dominance pairs and ridge strength. This is a low-capacity cost-aware
control, not a new architecture. Native graph logits remain the primary baseline.
Do not select the reduced feature set by target outcomes. Compare all-route
first-disagreement policies and matched random controls; add a previous-option
symptom timing ablation only as a separately declared arm, not as a replacement
when another arm fails.

Primary question: does the full-return preference offer evidence beyond a simple
shorter-execution preference? Report paired SR/SPL/nDTW, success losses, cost,
and scene-cluster intervals. If the reduced cost-aware scorer explains the full
scorer, describe the mechanism accordingly and do not claim a learned general
long-horizon value function. If gains fail replication, stop scaling the ranker.
Current state-level entropy/STOP probability/candidate count cancel in the linear
pairwise scorer; there is no evidence yet for instruction or uncertainty conditioning.

The critical census misses early ranking errors: route 3161 was not rescued at
its registered steps 13/14 but the frozen scorer rescued it at step 3. Therefore
no-rescue at the sampled critical times cannot identify proposal coverage failure.
For unresolved routes, inspect earlier timing and actual reachable graph actions
before changing the waypoint predictor. Interrupt first-event cuts rescued 0/4
routes in this cohort; they are a small cut-time experiment, not an exhaustive
upper bound over all possible interruption policies.
