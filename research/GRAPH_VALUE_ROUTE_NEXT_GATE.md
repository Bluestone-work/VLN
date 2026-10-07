# Next gate after route-disjoint execution diagnostic

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
