# Prospective continuation-label study

Date: 2026-10-07. Experiment: CONTINUATION-LABEL-TRAIN64-001.
Status: **completed**. Both H=1 and H=2-confirmed schedules were executed and
audited. Two further descriptive studies (cost decomposition and equal-primitive
windows) also completed. No learned model, RL or VLM was trained.

## Question and registered scope

Does an H=2 progress-and-cost confirmation filter make promising H=1 oracle
labels more reliable on new training scenes? This tests the fixed H=1 candidate,
not exhaustive trajectory search or a learned action abstraction controller.
Protocol: `CONTINUATION_LABEL_PROTOCOL.md`.
Config: `configs/CONTINUATION_LABEL_TRAIN64_001.json`.

## Independent sample and baseline

Metadata-only selection fixed 64 routes in 16 training scenes after excluding
19 scenes used by earlier training diagnostics. No selected scene overlaps
those prior scenes, and no route overlaps either validation split. All selected
routes remain in the sample; missing positive labels will not trigger resampling.

Baseline versus untraced control matches all 1,152 per-episode metrics and
18 aggregates exactly. Baseline SR/SPL/nDTW/SDTW = 87.5000 / 81.4706 / 81.5775 /
74.6881%. Mean goal error 1.6619 m, path length 9.8834 m, primitive count 61.5
and high-level decisions 7.328125. These are training-subset control scores,
not standard validation or a new model's performance.

Released checkpoint, seed 100, one RTX 4090/worker, native sensors/controller,
sliding=true, effective tryout=false, original STOP, success distance 3 m and
15-decision horizon are preserved. Sampling tests and label/audit regression
tests pass; no model, RL or VLM is trained.

## Complete-option and route results

All 469 selected sentinels and 1,365 reverse-order checks pass, including exact
sensor matches and zero endpoint error. Independent integrity verifies all
6,699 branch executions (5,334 forward, 1,365 repeated), covering 216,772
primitive events. Fifty-two final-budget non-STOP options are excluded under
the unchanged native horizon. The route audit reconstructs 192 native baseline
path/nDTW/SDTW values exactly.

On 405 non-STOP baseline states, joint-cost-capped local gain averages 0.025680 m;
the unchanged ordered-route gate leaves **0.008900 m**, with a median of zero.
Material opportunities shrink from 13 states / 11 routes / eight scenes to
**six states / five routes / three scenes**. This is sparse diagnostic support,
not evidence that all 16 scenes benefit. New scenes are new to this research
diagnosis, not held out from the released checkpoint's original training.

The predefined SHA256 rule selects one event on every eligible route:
1773/5, 6126/0, 1374/5, 10121/3 and 7581/5 (zero-based steps).
The exact typed actions and source hashes are frozen in `schedule_h1_001.json`
under the result root before any alternative continuation outcomes.

## Actual continuation outcomes

The disabled interceptor exactly reproduces all 469 trace records, 1,152 episode
metrics and 18 aggregates. H=1 replaces five actions, leaves 59 control routes /
1,062 metrics unchanged, and preserves 441 complete traces outside the altered
suffixes. All five replacements reproduce their isolated branch executions.

The registered H=2 rule accepts 6126 and 7581, rejects 1773, 1374 and 10121.
Because retained actions meet the predeclared mixed-rollout gate, the driver
automatically executes the mixed schedule after another disabled control.
Exactly two actions change; the other 62 routes / 1,116 metrics stay identical.
All 469 mixed-run full traces equal the earlier offline composition exactly.
Both enabled runs independently reconstruct 384 final metrics each with zero
error. Thus the H=2 result below is **executed**, not only a composed estimate.
The original `h2_analysis_001` snapshot still labels its earlier computation as
offline; `h2_composition_verification_001.json` records the subsequent execution.

All five eligible routes were baseline successes. These experiments therefore
do not test rescuing a failed route; no failed route had a material opportunity
under this particular screening rule. That absence does not prove no useful
recovery alternative exists under other labels.

| Five eligible routes | Baseline | H=1 interventions | H=2-confirmed schedule |
| --- | ---: | ---: | ---: |
| Success count | 5/5 | 5/5 | 5/5 |
| SPL (%) | 66.0905 | 73.9205 | 69.6394 |
| nDTW / SDTW (%) | 68.0487 | 77.0138 | 70.6196 |
| Total primitive actions | 692 | 478 | 692 |
| Mean final goal distance (m) | 1.3673 | 0.7497 | 1.1360 |
| Mean path length (m) | 17.0658 | 12.0125 | 16.8175 |
| Mean high-level decisions | 9.2 | 8.8 | 9.2 |

Across the original 64 routes, SR remains 87.5000% in every run. Baseline /
H=1 / H=2 SPL = 81.4706 / 82.0823 / 81.7478%; nDTW = 81.5775 / 82.2779 /
81.7784%; primitive count = 61.5000 / 58.15625 / 61.5000 per episode.
These are privileged training-subset results. No deployment, validation gain,
runtime saving, statistical significance or novelty is established.

## What the fixed H=2 rule gets wrong

| Episode / step | H=2 accepted | H=2 extra progress (m) | Final nDTW delta (pp) | Final primitive delta |
| --- | --- | ---: | ---: | ---: |
| 1773 / 5 | No | -1.1042 | +31.293 | -244 |
| 6126 / 0 | Yes | +0.4423 | +10.996 | 0 |
| 1374 / 5 | No | -0.2361 | +3.227 | -2 |
| 10121 / 3 | No | -0.0893 | -2.549 | +32 |
| 7581 / 5 | Yes | +0.6310 | +1.859 | 0 |

The rule rejects the adverse 10121 case but also rejects two useful alternatives,
including the largest efficiency gain, 1773. It retains two nDTW improvements
without extra primitives, but preserves none of the aggregate primitive saving.
One of its two accepted cases, 7581, has already terminated in both runs by H=2;
that case does not demonstrate prediction beyond the observation window.

Support remains five treated routes / three scenes, below the registered
eight-scene learning gate. Do not pool the earlier seven development cases into
this prospective sample to satisfy that gate. No confidence/significance claim
is made, and no new route is sampled to obtain a favorable answer.

## Automatically continued study: complete cost decomposition

After the primary result, `CONTINUATION-COST-DECOMPOSITION-001` includes **all**
seven earlier and five new interventions, with separate batch accounting. Every
primitive and collision decomposition matches native episode totals.

For the new five, immediate replacements save 39 primitives, later navigation
saves 105 and native STOP behavior saves 70: total -214. However, 1773 alone
saves 244. Removing it descriptively changes the other four routes to **+30**
primitives. This leave-one-out calculation exposes concentration; it is not a
new selected performance estimate.

In 1773, baseline forced STOP at decision 14 traverses four graph nodes and uses
71 primitives. The alternative naturally stops at decision 10 with one primitive.
Its 244-action reduction consists of 31 immediately, 143 in later navigation and
70 at STOP. No STOP logic is overridden. Its H=2 goal advantage is negative but
turns positive at H=3; positive progress and lower cumulative primitive cost
coexist from H=4 in this case. Those later horizons are **post-hoc descriptions**,
not tuned replacement label rules.

In 10121, the replacement saves three actions immediately, but later navigation
adds 35, ending with +32 and nDTW -2.549 pp. None of the five new cases reselects
the displaced original ghost ID, unlike earlier case 6962. Therefore simply
blocking that ID does not explain all observed later inefficiency. ID matches,
when present, also retain requested-target drift rather than being assumed to
identify an unchanged full action.

## Automatically continued study: equal primitive budgets

`EQUAL-PRIMITIVE-CONTINUATION-001` checks a timing confound after the H=2 results.
It compares actual trajectories at the number of primitives consumed by the
baseline's H=2 window, with STOP absorbing only in offline analysis. Independent
navmesh queries reproduce **456 recorded pre/post goal distances exactly** before
any intermediate-pose interpretation. No simulator actions, thresholds or
evaluation metrics change. Equal primitive count is not equal wall-clock time;
intermediate option positions are observations, not a new executable action.

The exploratory rule changes three accept/reject decisions in the old batch and
two in the new batch. It still rejects 1773; it also admits the previously harmful
9045 case. On the new five it accepts 1374/7581 instead of 6126/7581. Therefore
duration normalization alone does not make short-window progress a reliable
full-episode target in these cases. This is descriptive evidence, not independent
validation of another selector. Do not tune the budget or horizon on these cases.

## Decision and next gate

- **NO-GO:** original NMS-density AAA, unchanged.
- **NO-GO for direct training on these short-window confirmation labels.**
  Sparse coverage, rejected delayed gains, and duration-dependent decisions
  prevent treating the H=2 filter as a reliable general training objective.
- **CONDITIONAL GO:** narrow full-return graph-decision diagnosis. Some actual
  efficiency gains remain, but they are concentrated and not learned gains.

The next informative design is a small **outcome-blind state/candidate sample**
whose alternatives are followed through native STOP, retaining complete returns
and costs. Local goal gains must not determine eligibility for that study. This
tests the label/sampling problem before model capacity. See
`FULL_RETURN_LABEL_NEXT_GATE.md`; no such new experiment is claimed executed.

## Artifacts

All accepted outputs and exclusive logs are in `results/continuation_label_train64/`:
`cycle_001/`, `route_001/`, `schedule_h1_001.json`, `continuation_cycle_001/`,
`enabled_audit_001/`, `h2_analysis_001/`, `h2_enabled_audit_001/`,
`h2_composition_verification_001.json`, both native reconstructions,
`cost_decomposition_001/`, `equal_primitive_001/` and `figures_001/`.
The figure plots all five new cases, including the adverse one. Code/config/source
snapshots and original results are preserved. Twenty-two focused tests pass;
existing tracked navigation/controller core files are unchanged in this cycle.
