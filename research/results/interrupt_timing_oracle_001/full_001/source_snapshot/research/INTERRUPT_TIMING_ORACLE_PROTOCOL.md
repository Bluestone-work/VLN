# Bounded collision interruption timing oracle

2026-10-08. Retrospective training diagnostic, not a deployable method. Frozen
after the coverage audit and before new intervention outcomes. Current learned
ranking recipe and adaptive density remain NO-GO; no RL/VLM is introduced.

Population: every known eligible cut from the fresh64 stratum of
`interrupt_coverage_002/analysis_001/cuts.json`. Exactly 175 cuts: 87 on 7 failed
routes, 88 on 12 successful routes. No outcome-based subsampling of cuts. Successful
nonempty-back-path options with unknown phase remain outside scope. Episodes were
previously examined; this is not held-out validation and does not support benchmark
claims. All cuts involve forward collision; do not claim semantic interruption.

At each cut run the existing three arms: same panorama then finish native option;
cancel suffix and call unchanged navigator with native ghost consumption; same but
restore pending ghost. Reuse the already audited executor and driver. Native
baseline for each of 19 routes is replayed as the fourth comparison. Every case
starts at the original reset, matches graph/logit/action/physical/RNG prefix, makes
only one intervention and continues to native termination. Sensing-only must match
the complete native return. Sensor hashes at the cut must agree among the three
arms. No physics rewind; realized costs retained. Original seed100, checkpoint,
sliding=true, effective tryout=false, sensors, STOP and 15-decision limit remain.

Smoke: all 19 native controls plus one event on the lexicographically first failed
eligible route and one on the first successful eligible route, chosen before
outcomes. Full run only after valid smoke: 19 controls + 175*3 = 544 complete
rollouts. Never overwrite an attempt. Save config/plan, input and source hashes,
hardware, stdout/stderr, complete primitive traces and machine-readable outcomes.

Primary ORACLE questions, separately by consume/retain:

1. Among the seven failed routes, how many have any cut rescuing SR with no nDTW
   loss (>1e-6 tolerance)? Also report the stricter no-increase-in-primitives subset.
   Preserve SPL, error, path, cost and collisions; mixed outcomes stay visible.
2. Compare this timing oracle with the fixed earliest eligible cut per route.
   Selecting best returns is privileged; never call it a learned-policy SR.
3. Among twelve successful routes, report any tested SR harm, nDTW harm, and cost
   increase, plus earliest-event outcomes. Any-cut harm is a stress-test exposure,
   not the harm rate of an unimplemented trigger. Declining to interrupt remains
   a valid native control; report no-op ties explicitly.
4. Report per route/scene, option and cut, with separate final ghost semantics.
   Do not treat the 175 correlated cuts as 175 independent routes or pool consume
   and retain into a deployable policy. No threshold/model selection follows.

Gate: fewer than two rescued failed routes in distinct scenes with SR and
nondecreasing nDTW in either fixed ghost-semantics arm is NO-GO for expanding this
collision-interrupt recipe. Meeting that condition permits only an independently
specified confirmation with successful-route controls, not training. If timing
oracle succeeds but earliest event harms success routes, clearly state that timing
and abstention remain unsolved. No-run/invalid controls are INCONCLUSIVE, not NO-GO.

Novelty remains unverified: variable action duration and closed-loop macro control
have precedents. This experiment establishes or rejects recoverable opportunity,
not a paper contribution.
