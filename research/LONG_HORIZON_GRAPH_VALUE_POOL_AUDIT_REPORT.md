# Long-horizon Graph Value Pool Audit

Date: 2026-10-09

The metadata-only pool audit was rerun after the pilot, using the frozen
exclusion list from the confirmation protocol. It found 59 of 61 R2R training
scenes excluded by prior diagnostic cohorts. Only two scenes remain, with four
routes total; neither scene has the 12 routes required by the registered
confirmation sample. The audit therefore stops before sampling and before any
continuation outcome is read.

Decision: **blocked / NO-GO for a clean independent R2R confirmation with the
current pool**. Reusing one of the 59 scenes would change the estimand and would
violate the frozen scene-disjoint requirement. The next experiment requires an
explicitly registered expanded data pool or a separately labeled cross-scene
study; it must not silently reuse validation or previously inspected outcomes.

Artifact: `research/results/long_horizon_graph_value_pool_audit_002/summary.json`.
