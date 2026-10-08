# Next gate: event-specific replanning or merely an extra decision boundary?

Status: design specified after the completed timing oracle; no new sample or
confirmation rollout has been launched. Current decision is CONDITIONAL GO for
this causal confirmation only. Do not train on the 175 inspected cuts.

The timing oracle found consume quality-rescue on four failed routes in three
scenes, but the first-event rule reduced nDTW on eight of twelve successful
routes and increased mean primitive cost. A useful opportunity is established;
a reliable event trigger is not. Neither execution-majority nor novelty is shown.

## Independent population and frozen controls

Before intervention returns, register an outcome-blind training sample of 128
distinct routes across eight scenes (sampling seed20261009), excluding all
previously registered research routes. Audit scene/route identity and checkpoint
provenance. If this population cannot be sampled under the exclusions, stop and
record a design amendment before running it. Baseline-training scenes are allowed;
this does not become an unseen-environment benchmark.

Capture all native routes with primitive phases and episode metrics, including
successful routes. No failure-only selection. The event is the first eligible
non-final forward collision within a final ghost segment, excluding backtracking.
It uses the current collision and pending controller segment, not future goal,
reference-route or return labels. Keep native ghost consumption fixed; retaining
the ghost remains a historical ablation, not another parameter to select here.

For each exposed route, freeze these comparisons:

- Native complete continuation.
- Event-time sensing then continuation.
- Event-time sensing then cancel/replan once with the unchanged navigator.
- Three outcome-blind uniform interior-forward cuts in the **same native option**,
  selected with seeds20261011/12/13; each has sensing-only and consume/replan arms.

Uniform cuts may coincide with the event or another collision; retain and disclose
that overlap. If there is no other admissible cut, mark the timing comparison as
unsupported, not a favorable negative control. These are matched-option causal
controls, not deployable random policies. The three timing seeds are not three
independent simulator seeds. Routes without an event retain native behavior in
the all-sampled-route accounting; never report only rescued/exposed routes as the
method population.

Require exact native prefixes, matched sensor hashes at each paired cut, unchanged
checkpoint/sensors/collision settings/STOP, original 15-decision budget, no physics
rewind, no repeated interruptions and separate render/policy/primitive cost counts.
Smoke must include successful and failed controls if both occur, but population
and treatment rules cannot be tuned after baseline outcomes are known.

## Decision evidence

Report paired all-route SR/SPL/nDTW/cost, exposed-route rescue and harm, per-scene
counts and route/scene-cluster uncertainty. Compare event time against uniform
timing using the same eligible options, not different favorable states. Preserve
native abstention as a separate unchanged arm. Point gains that increase overall
primitive cost or reduce mean nDTW fail a cost/quality-controlled first-event
method claim, even if binary SR rises. Uncertainty spanning zero remains explicit
and cannot be converted into a positive method claim.

At least two independent route-quality rescues in distinct scenes are necessary
for mechanism continuation, but are not sufficient for learning GO. If event time
has no reliable advantage over matched uniform timing, narrow the interpretation
to added decision boundaries; do not train an event classifier. If success/control
harm or cost erases the benefit, stop this fixed event rule. If controls are invalid
or timing support insufficient, report INCONCLUSIVE rather than a negative result.

Only after this comparison can a separate deployable-feature/abstention study be
justified. No RL/VLM, new waypoint predictor, threshold search or preference-label
harvesting is authorized by the current oracle result alone.
