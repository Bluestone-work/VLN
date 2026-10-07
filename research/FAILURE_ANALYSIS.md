# First-Cycle Failure Analysis

> 2026-10-07 correction: the selection/execution numbers in this historical
> analysis use direct raw-candidate probes, not faithful graph-option rollouts.
> Treat them as proxies, not mutually exclusive causal failure counts. See
> `RANKER_VALIDITY_AUDIT.md` for geometry, action identity and uncertainty checks.

Date: 2026-10-06  
Scope: R2R-CE `val_seen`, eight episodes, released ETPNav IL checkpoint, fixed coarse/default/fine proposal-density variants.

This is a diagnostic report, not a claim that the categories below are already perfectly identified. The current logs observe proposal availability, selected action progress, collision counters and terminal outcomes. Separating abstraction failure from selection and execution requires the matched-state oracle described in `REPORT.md`.

## Observable decomposition

| Variant | Decisions | Zero waypoint proposals | Non-positive progress | Decisions with collision count > 0 | Terminal decisions |
| --- | ---: | ---: | ---: | ---: | ---: |
| coarse | 93 | 0 | 45 (48.4%) | 29 (31.2%) | 8 |
| default | 73 | 0 | 30 (41.1%) | 43 (58.9%) | 8 |
| fine | 65 | 0 | 33 (50.8%) | 33 (50.8%) | 8 |

Collision count is cumulative in the Habitat metric, so “decisions with collision count > 0” is an episode-history indicator rather than a per-option collision label. It should not be interpreted as 29/93, etc. exact execution-failure rates.

## Candidate/proposal failure

No decision in these three short runs had an empty waypoint proposal list. This rejects only the narrow hypothesis that the default NMS frequently produces no candidate on these episodes. It does not show that all useful candidates are present: a proposal can exist but point away from the route, be too short/long, or be unreachable.

## Abstraction failure

The expanded matched-state oracle provides a candidate-level estimate. Across
511 decisions, the best fine candidate averaged `1.552 m` one-step goal
progress, versus `1.266 m` for default and `1.116 m` for coarse. Fine was the
privileged best level on 49.3% of states and default on 50.7%; coarse was never
best. The fine-minus-default bootstrap 95% CI was `[+0.243,+0.329] m`.

The paired default diagnostics show a larger bottleneck: realized progress was
`0.610 m`, leaving a `0.656 m` gap to the best default candidate (`0.672 m`
when STOP decisions are excluded). This combines graph-action selection and
low-level execution, so it is not a pure selection estimate. It nevertheless
means the current abstraction signal is smaller than the existing downstream
gap.

## Selection failure

The 64-episode candidate-to-graph analysis now provides a counterfactual
estimate. All 2,295 default candidates mapped to graph nodes. Among valid,
unvisited current candidates, the best candidate averaged `1.072 m` progress,
while the candidate represented by the selected graph action averaged `0.729 m`.
The resulting proxy graph-selection gap was `0.471 m`; 95.3% of ghost decisions
did select a current candidate ghost. Because raw-candidate probes and actual
graph options do not have the same action identity, this does not establish
selection as the largest causal failure category.

## Execution failure

The historical selected-candidate proxy versus realized progress gap was
`0.052 m` on matched ghost decisions. This is not a complete execution metric,
because candidate-to-ghost merging and control backtracking are included, but
it cannot establish that execution is smaller than graph ranking. The new
opt-in controller trace records per-option primitive/collision events; results
are appended below. Failure attribution still requires valid alternative actions.

## Recovery failure

The current short fixed runs do not contain a reliable off-route/revisit label. The full released R2R baseline has 133 episodes that entered the 3 m success neighborhood but ended outside it, while RxR has 967. This is evidence for a terminal/recovery opportunity, not proof that abstraction caused it. A future analysis should mark recovery states from graph revisits, reference-path distance increases and previous-option progress.

## Stopping failure

The evaluator records active and forced stopping separately. In the existing full R2R residual-PPO analysis, episodes with any max-length forced stop had SR 15.48% versus 63.25% for episodes without one; this is confounded by episode difficulty. The short fixed runs show forced-stop rates of 2.5% coarse, 1.7% default and 0% fine. A stopping study must distinguish an intentional STOP node, a max-length truncation and a no-ghost fallback.

## Current interpretation

The fixed comparison and matched-state oracle show that proposal density affects
trajectories and has candidate-level potential. The privileged intervention
motivates a separate graph-decision study, subject to action-label calibration
and held-out validation. A deployable ranker has not yet shown stable gains.
The privileged graph-selection intervention is an observed rollout:
with default waypoint proposals unchanged, selecting the current ghost with the
largest one-step goal progress produced `100.0%` SR, `98.26%` SPL and `92.58%`
nDTW on 64 episodes. It changed 22.3% of decisions. The intervention remains
privileged and non-deployable; it is not a certified optimal upper bound.

Offline signal attribution shows that the current graph logit is not random:
its candidate-level Spearman correlation with privileged progress is `0.531`,
with a `58.7%` top-1 rate and `0.522 m` mean regret. The waypoint heatmap score
is weaker (`0.197` Spearman and `0.854 m` regret), so proposal confidence alone
does not explain the ranking discrepancy. Subsequent frozen-feature tests have
not established a stable improvement, and their raw-proposal labels require repair.

The corrected abstraction comparison changes the go/no-go interpretation. Fine
is the best fixed candidate level by mean progress, yet the statewise oracle
adds only `0.0287 m` over fixed fine. On the matched 64-episode navigation
sample, default is best among coarse/default/fine on SR, SPL and nDTW. Thus the
observed ranking failures should not be relabeled as evidence for adaptive
abstraction; they support a separate graph-ranking investigation.

## GraphMap aliasing control

The expanded default diagnostics contain 7,813 states from 960 train episodes
and six scenes. In 3,860 states (`49.4%`), two or more current proposals map to
the same graph ID. Across 4,382 aliased groups, privileged candidate progress
differs by `0.250 m` on average and up to `1.200 m`, so candidate-level oracle
labels are not always separable by the downstream graph action.

The matched `merge_ghost=false` control reduces the aliasing-state rate to
`12.4%` (63/510 states), but keeps SR at `79.69%` and lowers SPL/nDTW to
`69.65/73.01` from `70.16/73.61`. It also raises mean ghost count to `28.50`
from `21.53125` on the matched 64-episode baseline (23.85 belonged to the full
unseen split). This is evidence for a representation ambiguity and a useful
control for future rankers, but it does not support removing merging as a
standalone fix.

## Full-option execution symptoms (2026-10-07)

New source: `results/option_calibration/execution_001/summary.json` and
`execution_002/summary.json`. The frozen default controller is traced directly;
counts below are per executed option, not cumulative episode collision flags.

| Symptom | seen64 / 4 scenes | balanced unseen66 / 11 scenes |
| --- | ---: | ---: |
| Ghost options | 447 | 534 |
| Ghost options with at least one collision | 91 | 88 |
| Ghost endpoint >0.5 m horizontally from requested target | 58 | 48 |
| Ghost endpoint >1 m horizontally from requested target | 30 | 26 |
| Nonpositive-progress ghost options | 125 | 148 |
| Failed episodes | 13 | 22 |
| Failed episodes with any collision | 7 | 10 |
| Failed episodes with forced STOP | 3 | 5 |
| Failed episodes with native oracle-success=1 | 4 | 2 |

These counts overlap and do not partition failure causes. Target residual uses
the merged ghost target, not goal distance; a 0.5 m diagnostic threshold does
not redefine success or prove that a target is reachable. Necessary recovery
can have negative one-step goal progress. Both samples use seed 100 and effective
tryout=false; this is not a three-seed model claim.

Representative records (steps are zero-based, selected by largest horizontal
target residual, not by desired research conclusion):

- Seen episode 78, step 8: nine primitive events, eight collisions, no movement,
  requested target still 2.25 m away; goal distance stays 2.6612 m.
- Unseen episode 1126, step 13: 18 primitive events, 11 collisions, actual movement
  path 0.4368 m, target residual 2.3300 m, goal progress -0.0748 m.
- Unseen episode 413, step 10: three-node back path, 21 primitive events, seven
  collisions, target residual 1.9292 m, but positive goal progress 0.3551 m.

The last case illustrates why a target miss cannot automatically be labeled a
navigation failure. Six-way causal failure counts remain open; the next step is
the matched full-controller option experiment in `OPTION_LABEL_PROTOCOL.md`.

## Complete-option evidence after cost controls

The full-controller experiment is complete (`GRAPH_OPTION_REPORT.md`). Across
534 non-STOP decisions from the balanced unseen66 sample, 59 decisions have an
alternative with at least 0.25 m more goal progress while using no more primitive
actions and no longer a movement path. Those decisions occur in 28 episodes.
Ten selected options collide and 49 do not. This is not a collision-only signal.

Baseline-failed episodes contain 36 such states out of 205 non-STOP decisions;
successful episodes contain 23 out of 329. Outcome stratification is descriptive
and confounded by difficulty; it cannot assign a single cause to an episode.
The goal-progress criterion also does not guarantee instruction-route correctness.

Example: episode 1238, step 9, selected current ghost executes 16 primitives and
2.2611 m for -1.2807 m goal progress. Another current ghost executes 9 primitives
and 2.2601 m for +2.2318 m progress; neither collides. This is concrete evidence
of a lower-cost local alternative in that state, not a demonstrated recovery or
episode-success intervention. All such outcomes come from isolated branch
execution with passing native selected-action sentinels.

Unconstrained mean opportunity (1.2846 m) shrinks to 0.1488 m with both cost caps.
The large reduction prevents treating long historical-ghost backtracking as a
free one-step ranking improvement. Six-way causal decomposition remains open.
## Ordered-route follow-up (2026-10-07)

The 534-state unseen full-option study now includes a geometric ordered-reference
proxy. Joint-cost-capped mean local opportunity falls from 0.148787 to 0.126962 m
under the matched-prefix route gate; 49 rather than 59 states retain >=0.25 m.
See `ROUTE_COMPATIBILITY_PROTOCOL.md` for the frozen gate and its limitations.

This does **not** turn local outcomes into causal failure categories. Post-hoc,
34 of the 49 selected actions have negative goal progress, but 12 of the retained
alternatives also have nonpositive goal progress. Nineteen states occur in
episodes that the original policy successfully completes. Removing such actions
may interrupt necessary backtracking. Sixteen selected alignment endpoints are
still at reference start and ten at reference end; the proxy cannot certify
landmark compliance or all useful recovery behavior.

Episode 382, decision 0 shows why goal-only labels can mislead: an alternative
gains 1.8310 m over the selected action in goal distance, yet its matched route
endpoint regresses from 5 to 0; no material alternative survives the primary
gate. Episode 268, decision 8 retains a 5.2438 m relative gain but still has
-0.0862 m absolute alternative progress, and the baseline episode succeeds.
Both examples are geometry/continuation warnings, not proof of a semantic error.
Plots and original instructions are archived in
`results/route_compatibility_001/unseen66_interpretation/`.

## Actual single-action continuations (2026-10-07)

The seven previously frozen training events now have actual native continuations
after one privileged graph-action replacement. All fidelity gates pass; the
other 57 routes are exact controls. These are seven post-selected cases, not an
unbiased estimate of failure prevalence or a six-way causal partition.

- Success remains 4/7; none of the three failures is rescued. All four successful
  routes improve SPL/nDTW. Among the three failed routes, one improves nDTW and
  two worsen it. Mean goal error increases 0.6155 m.
- **9045 / step 7:** local relative progress +0.8181 m, immediate cost -2
  primitives, but absolute alternative progress is still -0.8197 m. Subsequent
  native movement goes farther away twice, then stops. Final error increases
  7.9071 -> 9.8087 m; nDTW drops 14.798 pp; primitives increase by 17.
- **6962 / step 10:** local gain +3.5544 m and immediate saving of four
  primitives. The next policy choice selects the displaced original ghost
  `g24`; native forced STOP at step 14 returns through two nodes to historical
  node `8`, using 29 primitives versus one in the baseline. Final goal error
  rises 3.5230 -> 8.5082 m. Total extra primitives are -4 immediately, +41 in
  later movement and +28 at STOP, totaling +65. Only this state on the route
  is intervention-tested; do not extrapolate harm to its other local labels.
- **875 / step 6:** nDTW improves 13.089 pp and final error falls 1.6729 m,
  but native STOP occurs at 4.8004 m, outside the unchanged 3 m criterion.
  This is a stopping symptom; overriding STOP has not been tested.

All seven immediate options satisfy both cost caps. They save 17 primitives
in total, but complete episodes use 37 more: later continuation adds 54 relative
to the immediate difference. Thus action-level cost control does not establish
episode-level cost control. The two adverse routes reverse their local progress
advantages at H=2 in an explicitly **post-hoc** offline examination. Fixed
high-level windows have unequal primitive costs; no validated new label rule
or STOP policy follows from this observation.

Sources: `SINGLE_INTERVENTION_REPORT.md`,
`results/single_intervention_train7/enabled_audit_001/paired_episodes.json`,
`interpretation_001/immediate_cost_controls.json` and
`horizon_posthoc_001/summary.json` under the same result root. A graph-decision
efficiency question remains open; current one-step labels are not yet a reliable
recovery objective. No new scorer is trained.

## New-scene confirmation and delayed effects (2026-10-07)

The prospective follow-up excludes all 19 prior training diagnostic scenes and
samples 64 routes / 16 new scenes. Only five routes / three scenes retain a
material joint-cost and route-compatible H=1 option, and all five are baseline
successes. Actual continuations preserve success 5/5, improve nDTW on four and
worsen it on one. The registered H=2 filter rejects the adverse case **10121**
but also two useful cases **1773** and **1374**. Its two accepted interventions
are separately executed and audited; one is already terminal by H=2.

Concrete counterexamples to the short-window objective:

- **1773 / step 5:** H=2 extra progress -1.1042 m and +2 primitives; final nDTW
  +31.293 pp and -244 primitives. Cost decomposition is -31 immediately, -143
  in later navigation and -70 at STOP. Baseline forced STOP at decision 14
  uses 71 primitives through four graph nodes; altered navigation naturally
  stops at decision 10 with one primitive. No STOP intervention is performed.
- **10121 / step 3:** immediate local gain +1.0570 m and -3 primitives, followed
  by +35 later navigation primitives. Final nDTW -2.549 pp, final goal distance
  +0.4926 m and total +32 primitives. It remains successful under the unchanged
  3 m criterion. SPL alone would hide part of this deterioration.
- **1374 / step 5:** H=2 extra progress -0.2361 m, yet final nDTW +3.227 pp and
  -2 primitives. A transient goal-distance regression is not necessarily a bad
  full-episode action.

None of the new five reselects the displaced original ghost ID. The recurrence
seen in old route 6962 is therefore not a sufficient general explanation. The
descriptive all-case cost decomposition retains requested-target drift for ID
matches rather than assuming IDs always imply unchanged full actions.

A follow-on equal-primitive-budget analysis first reconstructs 456 native
geodesic endpoint measurements exactly. It still rejects 1773 and would admit
old harmful case 9045. This exploratory result does not justify tuning another
budget/horizon. Both measurement clocks can mislabel delayed behavior.

Sources: `CONTINUATION_LABEL_REPORT.md` and `results/continuation_label_train64/`
(`enabled_audit_001`, `h2_enabled_audit_001`, `cost_decomposition_001`,
`equal_primitive_001`, `figures_001`). The two training batches have different
selection histories; do not pool their scores or infer six-way causal failure
frequencies. Current evidence concerns label validity and sparse efficiency
opportunity, not a proven abstraction, recovery or STOP fix.


## 2026-10-07 full-return feature/cost follow-up

In the frozen 16-route / eight-scene training pilot, 32 alternative continuations
are fully accounted for. Top-logit alternatives add 248 primitives (109 current
option, 139 later navigation, no additional STOP cost); seeded alternatives add
1,008 (330 current, 643 later navigation, 35 STOP). These execution partitions
are exact accounting categories, not causal selection/execution/recovery failure
classes. Selection changed future observations, graph history and decisions.
The sole dominating baseline rescue has low normalized policy entropy 0.0727;
a high-entropy trigger would not necessarily catch it. There is no fitted trigger
or validated feature predictor. See `FULL_RETURN_FEATURE_AUDIT_REPORT.md`.


## 2026-10-07 native STOP counterfactual audit

The earlier endpoint census is refined with actual saved native STOP branches.
ETPNav STOP can return to the historical node with maximum saved STOP score;
proximity at the current pose does not guarantee successful execution. Across
600 unseen66 states, 9 near states / 8 routes have a failed native STOP, while
17 far states / 3 routes have a successful one. These include successful routes
and do not establish 8 extra failed episodes. Only 1 of the 22 failed unseen
routes (73) has a successful native STOP branch at an earlier decision.

The other oracle-arrival failure (1593) reaches 2.91524 m inside a forward
macro-action; only three primitive endpoints are inside 3 m. Its nearest
high-level boundary remains at 3.10743 m and its final error is 3.20625 m.
This is a concrete temporal-resolution symptom with no deployable trigger and
only one observed scene. Eighty-four pre/post geodesic measurements for the
five oracle-arrival failures across cohorts reproduce native distances exactly.
The three training failed routes with successful native STOP branches are
10587, 393 and 2508, kept in their original cohorts. All other failed routes
remain unclassified causally. No candidate, controller or STOP rule was changed.

See `NATIVE_STOP_FEASIBILITY_REPORT.md` and its saved cross-tabs/primitive traces.
