# Failure-critical full-return census + interrupt oracle

Date: 2026-10-07. Coarse/default/fine selector development is closed; its results
remain negative diagnostics. This cycle follows the user's explicit request to
test critical-state graph choices and interruption before choosing a mechanism.
No learned selector, RL or VLM was added.

## What was actually run

The training experiment includes all nine failures in the existing train64
cohort (eight scenes). Critical states are every non-STOP collision/deviation/
stall option, the final non-STOP state, and terminal STOP, deduplicated. This
produces 46 states and 509 admissible native actions, including 46 native selected
actions. Each case replays from the original episode start, changes only one
choice, then runs the original navigator to native STOP within 15 decisions.
This is exhaustive at those states, not an exhaustive future policy tree.

The checkpoint, sensors, seed 100, sliding=true, effective tryout=false,
backtracking controller and success <=3m are unchanged. A nine-route dataset
subset only changes episode iteration; its episode records and vocabulary are
identical to the source. The new executor is opt-in through a registered research
trainer/environment and reuses the existing two graph-action callbacks. No
baseline core file was modified in this cycle.

The checkpoint is loaded once, then graph state and environment are reset per
case. There are 9 baseline controls, 509 action cases, and 12 interrupt/control
cases: 530 actual complete rollouts. The outer evaluator writes one explicitly
labeled baseline control; counterfactual results are in `full_001/results.jsonl`
and must not be read from that outer aggregate as an oracle benchmark score.

## Training results

| Failed route | Critical states | Native actions | Non-STOP replacement rescues SR | Termination replacement rescues SR | Interrupt rescue |
| --- | ---: | ---: | --- | --- | --- |
| 3208 | 2 | 28 | no | no | not tested |
| 6068 | 2 | 40 | yes | yes | not tested |
| 6962 | 5 | 83 | yes | no | no |
| 7221 | 14 | 102 | yes | no | no |
| 8219 | 14 | 107 | yes | no | no |
| 8343 | 3 | 40 | no | no | no |
| 875 | 2 | 44 | no | no | not tested |
| 9045 | 2 | 44 | yes | no | not tested |
| 9747 | 2 | 21 | yes | yes | not tested |

Native non-STOP replacement rescues 6/9 routes in five scenes. Only 4/9 in three
scenes can also avoid nDTW degradation: 6962, 7221, 8219, 9747. Thus SR-only
ranking opportunity is a majority, but the route-quality-aware criterion is not.
Termination opportunities overlap (6068 and 9747); do not sum these columns into
a causal partition. Across all 509 tested action cases, 41 end successfully.

Representative success cases below maximize final nDTW among successful tested
choices per route. This is a retrospective oracle illustration, not a deployed
selection rule:

| Route / step / action | Final error (m) | nDTW change (pp) | Primitive change |
| --- | ---: | ---: | ---: |
| 6962 / 7 / 13 | 1.0408 | +14.849 | -94 |
| 7221 / 2 / 8 | 1.0546 | +18.518 | +12 |
| 8219 / 1 / 7 | 1.3337 | +58.327 | +13 |
| 9747 / 4 / 13 | 0.6770 | +21.372 | +13 |
| 6068 / 7 / 22 | 2.2061 | -11.022 | +29 |
| 9045 / 7 / 23 | 1.2650 | -13.092 | +55 |

The last two rows explain why success alone is insufficient. Route 6962's final
nDTW is still only 0.1628 despite improvement. Success rescue is not instruction
compliance certification or a cost-free correction.

## Small interrupt oracle

Four training routes have a frozen first eligible ghost-segment event: 6962 at
step10/primitive28, 7221 at step1/primitive7, 8219 at step1/primitive8, and 8343 at
step1/primitive5. All four are collision cuts. Cuts occur after backtracking and
strictly before the option's last primitive. This cycle does not test
backtracking interruption or semantic-change triggers.

At each cut, compare native completion, extra sensing then completion, and
extra sensing then cancellation/replanning with either native ghost consumption
or restoration of the pending ghost. The restoration affects only ghost entries;
physical prefix, RNG, collision and primitive cost remain real. The front-node
predecessor retains the baseline convention; no partial backtracking edge is
introduced. Fresh sensing uses exactly the existing sensors. Every subsequent
policy invocation counts against the original 15 decisions.

| Arm, 4 routes | Success rescues | Mean nDTW change (pp) | Total primitive change |
| --- | ---: | ---: | ---: |
| Matched sensing, continue | 0 | 0.000 | 0 |
| Interrupt, native ghost consumption | 0 | +2.981 | +81 |
| Interrupt, retain pending ghost | 0 | -1.481 | +83 |

The sensing-only arm exactly reproduces the original complete trajectory and
metrics. The two interrupt arms see identical cut panoramas to their sensing
controls. Route8219 improves nDTW by 13.107pp with consumption but still fails;
route6962 becomes worse. These four cuts do not support an interrupt model now.
They are not an optimized cut-time upper bound, and cannot reject every possible
trigger, recovery controller or semantic interruption policy.

## Verification and failures retained

- Nine baseline controls and all 46 selected-action sentinel full returns match
  the source metrics; 4,240 final metric comparisons reconstruct independently.
- A separate audit checks 4,709 complete option/prefix records and all 52,012
  primitive records exactly, including poses, actions, collisions, sensors and RNG.
- All eight actual interrupted-prefix/cost checks pass; four matched sensing
  controls pass. There are 12 arms, but only eight actual interrupted executions.
- Five eligibility boundary tests pass, including backtracking exclusion,
  final-primitive exclusion and forward-stall versus turning distinction.
- An earlier legacy-template adapter failed the population gate before rollout.
  Its outputs are retained and marked `SUPERSEDED.json`; none enter the results.
  First smoke failed on integer/string episode metric keys after a completed
  control, the second was manually interrupted during debugging, and smoke003
  passed. No failed attempt is counted as a completed experiment.

Artifacts: `results/critical_causal_train64/{plan_001.json,full_001,analysis_001,
independent_audit_001,figures_001}`. Source snapshots, checkpoint/input hashes,
hardware, stdout/stderr and per-case graph/primitive traces are retained.

## Current gate and remaining uncertainty

**CONDITIONAL GO for long-horizon graph choice investigation; NO-GO for training
an interruption policy from this pilot.** The native candidate set contains
rescues; graph-value/preference work now has stronger support than density
selection. Route3208, 8343 and 875 remain unresolved. No rescue among tested
native actions does not identify proposal coverage failure or justify changing
the waypoint predictor. Execution, graph selection and termination remain coupled.

The next registered confirmation applies the same critical-state definition to
the 22 failures in the already-inspected unseen66 cohort. This is a separate
cross-scene diagnostic, not an untouched test set. No validation-derived labels,
feature tuning or training are permitted. See `CRITICAL_CAUSAL_UNSEEN_PROTOCOL.md`.
Novelty against Macro-action PPO, AgenticNav, C²Nav and SparseNav still needs a
method/code comparison before a learned contribution is claimed.


## Cross-scene unseen66 confirmation (diagnostic only)

The same registered rule was applied to all 22 failures in the previously
inspected 11-scene unseen66 diagnostic subset. It covered 71 states and 985
native actions; all 22 controls completed and all 8,152 metric reconstructions
passed. Native non-STOP replacement rescues SR on 7/22 routes; 6/22 also avoid
nDTW degradation. Termination replacement rescues 6/22, overlapping the ranking
set. Four collision cuts again yield 0/4 interrupt rescues. Sensing-only arms
match native exactly; interrupt consume/retain arms change mean nDTW by +1.570
and +0.398 percentage points respectively, without a success rescue.

This subset was used in earlier diagnostics, so it is cross-scene confirmation,
not an untouched held-out benchmark. The combined train64 + unseen66 descriptive
counts are 10/31 route-quality-aware ranking opportunities, 8/31 termination
opportunities and 0/8 tested interrupt routes; rescue sets overlap and unresolved
routes remain in the denominator. Ranking is the leading mechanism observed, but
not a strict majority. No method is trained from unseen outcomes. See
`CRITICAL_CAUSAL_UNSEEN_PROTOCOL.md` and the separate result directory.
