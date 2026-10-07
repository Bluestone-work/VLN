# Native STOP feasibility diagnostic

2026-10-07. Post-hoc diagnostic; previously collected trajectories and coarse
outcomes have been inspected. This protocol is fixed before computing the new
STOP cross-tabulations, not claimed as an independent prospective evaluation.

Keep the five config-listed cohorts separate, retaining all baseline successes
and failures. Read only accepted forward index-0 branches with full action/RNG
identity and independent integrity checks. Native terminal success uses distance
**<= 3 m**, as implemented in ss_trainer_ETP.py. Earlier endpoint census used
strict < 3 m: quantify boundary cases rather than silently changing that result.

At each baseline decision record pre-decision proximity, actual selected action,
historical STOP target identity/back-path, STOP branch terminal distance and
primitive count. Four cells: near/successful STOP, near/failed STOP,
far/successful STOP and far/failed STOP. Group by route; do not treat decisions
as independent subjects. Verify selected native STOP exactly against baseline
execution and baseline final success against <= 3 m.

For each failed route record: native oracle_success; any boundary arrival;
any successful counterfactual native STOP; missed successful STOP indices;
whether all decision boundaries remain outside despite native oracle_success.
These categories overlap. A route without successful STOP is unresolved, not
proof of proposal/execution failure. A successful branch is an oracle timing
opportunity under the unchanged STOP target/controller, not a learned fix.

No metric or threshold tuning, no new scenes, no training. Do not report a
synthetically selected branch aggregate as an actually executed agent. Positive
STOP feasibility does not revive the rejected NMS-density abstraction method.
