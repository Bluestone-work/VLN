# Independent event-timing confirmation

Specified 2026-10-08, before sampling or reading outcomes. This is the next gate
after `INTERRUPT_TIMING_ORACLE-001`; it tests whether collision timing has value
beyond adding an arbitrary replanning boundary. It is not a deployable policy.

## Population

Select 128 distinct R2R-CE train routes, 16 routes in each of eight scenes, using
seed 20261009 and metadata only. Exclude every route listed by the registered
prior diagnostic/confirmation manifests and JSONL route traces, including the
fresh64 cohort used by the timing oracle. Scene overlap with prior work is allowed;
route overlap is zero. Select one instruction episode per trajectory by the frozen
seeded ordering. Do not inspect baseline success, collision, primitive outcomes,
reference-route returns or graph branches during selection.

Capture all 128 native episodes with the unchanged release checkpoint, seed 100,
sensors, sliding=true, effective tryout=false, graph controller, STOP behavior and
15-decision limit. Preserve primitive phases, poses, collisions, RNG, observation
hashes and graph actions. Run exact native controls and sensing/noninterference
checks before reading any event outcomes.

## Frozen timing comparisons

The event is the first eligible non-final forward collision in a final ghost
segment, after backtracking, using only executed prefix information. Routes without
an event retain native behavior and remain in all-route denominators. For every
exposed event, freeze:

1. native complete continuation;
2. same cut panorama, then finish the pending option (sensing-only);
3. event-time sensing, cancel the suffix and call the unchanged navigator once,
   consuming the pending ghost as in the original executor;
4. three outcome-blind interior-forward cuts in the same native option, selected
   with seeds 20261011, 20261012 and 20261013 before intervention outcomes, each
   with sensing-only and consume/replan arms.

Uniform cuts may coincide with a collision; retain and disclose overlap. If there
is no eligible interior cut, mark the route unsupported for that timing contrast,
not favorable. Do not use retaining/restoring the pending ghost in this
confirmation; it is a historical semantic ablation, not a selected method.

Each route receives at most one event intervention and three uniform-cut controls.
No repeated interruptions, threshold search, outcome-based cut selection, RL, VLM,
waypoint changes or graph-ranker fitting are allowed. Native abstention is always
available and reported separately.

## Validity and analysis

Use a smoke containing both successful and failed native controls if present, then
run the frozen full schedule. Every pair must match the native physical prefix,
graph masks/action identity, RNG and cut sensor hashes; no physics is rewound.
Every added navigator call counts against the original 15-decision budget. Count
renders, encoder/policy calls, primitive actions, collisions and high-level steps
separately. Reconstruct SR, SPL, nDTW, SDTW, endpoint error, path and primitive
cost from recorded traces.

Report all 128 routes, exposed and unexposed strata, per-scene route-cluster
intervals, event versus uniform timing, sensing-only versus replan, and native
continuation. Keep successful-route harm and cost visible. A quality rescue means
SR rescue with nDTW nondegradation; a cost-capped rescue also requires no extra
primitive actions. Do not pool correlated cut arms as independent routes.

Decision gate: event timing must show at least two route-quality rescues in distinct
scenes and outperform matched uniform timing on route-level quality without a
successful-route SR loss or a material mean nDTW/cost penalty. If event and uniform
timing are indistinguishable, interpret the previous oracle as evidence for an
extra decision boundary only. If controls are invalid or event exposure is too
small, report INCONCLUSIVE. Passing authorizes only a second independent
confirmation, never policy training or a benchmark claim.
