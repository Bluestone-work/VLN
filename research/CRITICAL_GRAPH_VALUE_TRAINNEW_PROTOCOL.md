# Prospective failure-only graph-value confirmation

Registered 2026-10-07 after the first critical census. This is a training-only
development cohort and is not a benchmark score. The 64-route continuation-label
sample was frozen by metadata before its continuation outcomes were inspected;
its eight baseline failures are retained in full, spanning seven scenes. The
prior nine-route critical census, all unseen66 outcomes, and all branch outcomes
are excluded from feature selection and labels.

## Frozen population

Use the eight failures in `CONTINUATION-LABEL-TRAIN64-001` exactly as observed in
its native control run. Do not add or remove routes after reading full-return
outcomes. At every registered critical state, enumerate every admissible native
graph action, including the selected action and STOP where the graph mask admits
it. Critical states are the union of non-STOP collision >=1, horizontal endpoint
deviation >=0.5 m, forward stall >=3 primitives with <=0.1 m motion, the last
non-STOP decision, and terminal STOP. Deduplicate by episode and high-level step.

Each intervention replaces one graph action, then uses the unchanged navigator,
controller, sensors, graph updates, STOP rule and 15-decision limit until native
termination. Controls and selected-action sentinels must reproduce recorded
metrics, primitive prefixes, collision sequences, sensors and RNG. This is
exhaustive over registered states and native actions, not a future policy tree.

## Outcome-blind feature contract for later feasibility work

Freeze before reading these full-return outcomes: native action logit and rank,
policy entropy and STOP probability, candidate bearing and distance, commanded
polyline/back-path cost, ghost/proposal flag, visited/revisit counters, collision
and history counters, instruction embedding and existing visual/topological
candidate embeddings. Future collision, goal/reference geometry, oracle arrival,
full-return metrics and continuation outcomes are forbidden deployable features.
No model is fitted in this census.

## Gate

Report route-quality rescue (success with nDTW nondegradation), SPL, path and
primitive cost, termination rescue, unresolved routes and overlaps. A preference
study is only CONDITIONALLY justified if at least two routes in distinct scenes
show route-quality rescue with exact controls. A strict majority is not assumed;
unresolved routes remain in the denominator. If support is sparse or harmful
cost dominates, stop before fitting. Unseen66 remains diagnostic-only and cannot
supply training labels or tune features.
