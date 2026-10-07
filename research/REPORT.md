# Adaptive Action Abstraction in VLN-CE

> Validity update (2026-10-07): see `RANKER_VALIDITY_AUDIT.md` and the Chinese
> progress summary `RESEARCH_STATUS_CN.md`. The 64-episode unseen ranker sample
> covers one scene only. Candidate probes differ from actual graph options;
> selection/execution gap numbers below are proxies. Privileged intervention
> rollouts are observed results, not certified optimal upper bounds. Historical
> decision-bootstrap intervals are supplemented by clustered intervals below.

## 1. Research question

Does a VLN-CE agent benefit from choosing the form and granularity of its next action according to the current navigation state, rather than using one fixed waypoint proposal abstraction throughout an episode?

The initial hypothesis is deliberately testable: coarse proposals may be adequate in open corridors, denser proposals may help at divergent junctions, and a useful adaptive selector should outperform the best fixed proposal setting under the same observations and execution protocol.

Prior adaptive trajectory truncation in Beyond Waypoints narrows this premise;
generic adaptive granularity is not a supported novelty claim. See the
method-level comparison in `LITERATURE_METHOD_AUDIT_20261007.md`.

## 2. Existing system

The audited ETPNav agent predicts a 120-by-12 waypoint heatmap from 12 RGB/depth views, applies NMS, inserts selected candidates as ghost/frontier nodes in an online GraphMap, and selects one graph action from STOP or unvisited ghosts. The selected ghost is executed by a continuous controller with optional graph backtracking. Full details are in `AAA_CODE_AUDIT.md`.

The original waypoint abstraction is NMS with at most five proposals and `sigma=(7,5)`. Graph action count is variable after graph construction; treating it as fixed would be an incorrect experimental description.

## 3. Empirical motivation

The released full baselines are R2R-CE val_unseen SR/SPL `57.0962/49.1014` and RxR-CE val_unseen `55.4334/45.1361`. A current-code one-episode smoke reproduced the existing R2R smoke metrics.

The first controlled R2R `val_seen` comparison used the same checkpoint, seed, simulator, controller and eight episodes for all variants:

| Fixed proposal level | SR | SPL | nDTW | high-level decisions | mean candidates/decision | positive-progress decisions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| coarse (3, `(12,7)`) | 50.0% | 32.31% | 41.35% | 11.63 | 2.43 | 51.6% |
| default (5, `(7,5)`) | 87.5% | 68.92% | 70.86% | 9.13 | 4.38 | 58.9% |
| fine (8, `(4,3)`) | 62.5% | 39.99% | 49.73% | 8.13 | 7.37 | 49.2% |

These are only eight episodes and must not be presented as final benchmark results. They do establish that proposal density changes the behavior substantially under a fixed navigator: it is not a numerically inert switch. The default setting is best in this small sample, so the result does not yet show that adaptive selection beats the best fixed setting.

The same variants were then evaluated on all 1,839 R2R-CE `val_unseen`
episodes. The default row is the current-code full baseline and matches the
published reference to floating-point tolerance:

| Fixed proposal level | SR | SPL | nDTW | NE | path length | primitive steps | high-level decisions | mean ghost count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| coarse | 54.2143% | 45.4745% | 59.6499% | 5.0061 | 12.3327 | 83.6074 | 9.0288 | 17.3143 |
| default | 57.0962% | 49.1014% | 62.3051% | 4.7344 | 11.5693 | 80.2741 | 8.7776 | 23.8499 |
| fine | 57.5313% | 47.2602% | 60.0452% | 4.6983 | 13.0260 | 90.4894 | 8.0109 | 26.7945 |

Paired against default over the same episode IDs, coarse changes SR by
`-2.8820` percentage points, SPL by `-3.6269` points and nDTW by `-2.6551`
points. Fine changes SR by `+0.4350` points, but SPL by `-1.8413` points and
nDTW by `-2.2598` points. Scene-cluster bootstrap intervals (11 scenes,
10,000 resamples, seed 20261006) for fine-minus-default are SR
`[-1.3817,+2.6210]` points, SPL `[-3.6014,+0.3776]` points and nDTW
`[-3.6190,-0.6732]` points. Thus fine is a success/efficiency tradeoff, not a
clear improvement. The full machine-readable comparison is
`research/results/action_abstraction/full_comparison_summary.json`.

The geometric diagnostic is suggestive but not decisive. Default decisions had mean progress `0.406 m` and positive-progress rate `58.9%`; coarse had `0.116 m/51.6%`; fine had `0.193 m/49.2%`. Proxy labels differ in frequency across variants because changing proposals changes the observed graph, so scenario-conditioned comparisons must be run on matched state snapshots before drawing conclusions.

## 4. Oracle study

The matched-state oracle was expanded to 64 `R2R-CE val_seen` episodes (511
decisions). At each decision, one waypoint heatmap is computed, all three NMS
levels are derived from that heatmap, and every candidate is evaluated through
`cand_dist_to_goal`, which restores the Habitat worker state after probing. The
oracle therefore does not alter the rollout and is explicitly privileged
analysis only.

| Same-state best candidate | Mean one-step goal progress (m) | Oracle selection rate |
| --- | ---: | ---: |
| coarse | 1.116 | 0.0% |
| default | 1.266 | 50.7% |
| fine | 1.552 | 49.3% |

Fine has a mean upper-bound advantage of `0.285 m` over default (paired
bootstrap 95% CI `[0.243, 0.329]`), while coarse is below default by `0.150 m`.
The preference is state dependent: default wins on `62.8%` of decisions within
3 m of the goal, while fine wins on `60.8%` of decisions 3–6 m from the goal.
This is evidence that proposal abstraction can be state-dependent at the
candidate-availability level.

That comparison is against default only and should not be mistaken for an
adaptive gain over the best fixed abstraction. Fine is the best fixed level by
mean candidate progress (`1.552 m`), but the statewise oracle exceeds fixed
fine by only `0.0287 m` on average (standard error `0.0060 m`). There were 218
tie states between default and fine, so the tie-stable oracle selection counts
(`50.7%` default, `49.3%` fine) overstate a meaningful preference. Among strict
non-tie states, fine won `86.0%` and default `14.0%`.

The same 64-episode rollout also logged the actual default decisions. Their
realized progress averaged `0.610 m`, compared with `1.266 m` for the best
default candidate available at the same state. The combined candidate-selection
and execution gap was `0.656 m` overall and `0.672 m` on ghost actions. This is
larger than the fine-versus-default abstraction gap, so selector training is
premature. The reproducible artifacts are
`research/results/action_abstraction/oracle_valseen64.yaml`,
`oracle_valseen64.stdout`, `oracle_valseen64.jsonl`,
`oracle_valseen64_summary.json`, and
`oracle_valseen64_selection_gap.json`.

The matched fixed rollout comparison is more decisive for deployable behavior:

| Fixed abstraction | SR | SPL | nDTW |
| --- | ---: | ---: | ---: |
| coarse | 62.50% | 54.20% | 66.29% |
| default | 79.69% | 70.16% | 73.61% |
| fine | 71.88% | 58.83% | 68.19% |

This is the same 64-episode `val_seen` sample and checkpoint. Default is best
on all three navigation metrics. The complete record is
`research/results/action_abstraction/fixed_comparison_valseen64.json`.

The historical candidate-to-graph proxy analysis attempted to separate this gap. All 2,295
default candidates mapped to a GraphMap node. Among 447 ghost decisions, 95.3%
selected a ghost represented by a candidate in the current observation. For
currently valid, unvisited candidates, the best available candidate had mean
progress `1.072 m`; the candidate represented by the selected graph action had
counterfactual progress `0.729 m`. Their graph-selection gap was `0.471 m`,
whereas the subsequent proxy execution gap was `0.052 m`. The action-identity
audit invalidates a causal comparison of those two numbers: raw proposals and
executed merged-ghost options are different actions. Details are in
`research/results/action_abstraction/graph_selection_valseen64_analysis.json`.

An earlier privileged graph-selection **and STOP** intervention kept the default waypoint generator and
GraphMap unchanged but executed the valid current ghost whose counterfactual
one-step goal progress was largest. On 64 `val_seen` episodes, this oracle
rollout reached SR `100.00%`, SPL `98.26%`, nDTW `92.58%`, path length `7.75 m`
and 6.31 high-level decisions. The fixed-default rollout on the same sample
was SR `79.69%`, SPL `70.16%`, nDTW `73.61%`, path length `9.72 m` and 7.98
decisions. The intervention changed 22.3% of graph decisions. These are
privileged trajectory metrics, not a deployable method or certified optimal
bound. The joint intervention cannot attribute its gain to graph ranking alone;
it is separate from the current native-STOP single-intervention protocol.
It motivated a separately controlled graph-decision diagnostic.
Records and summary are in
`research/results/action_abstraction/graph_selection_oracle_valseen64.jsonl`
and `graph_selection_oracle_valseen64_summary.json`.

An offline ranking attribution over the 511 matched default decisions found
that the existing graph logit is informative but imperfect: candidate-level
Spearman correlation with privileged progress was `0.531`, its top-1 best-
candidate rate was `58.7%`, and mean regret was `0.522 m`. The raw waypoint
heatmap score was weaker (`0.197` Spearman, `50.1%` top-1, `0.854 m` regret),
so simply exposing heatmap confidence is unlikely to solve the downstream
problem. These numbers originally motivated frozen-feature ranker tests; the
later validity audit requires controller-consistent labels before more fitting.

As a first train-to-validation feasibility check, a ridge linear ranker was
fit on 64 training-split episodes (1,809 candidate rows, 485 states) using
graph logit, heatmap score, candidate distance/angle, policy entropy and
candidate-count features. On the 64-episode `val_seen` set it reached only
`54.99%` top-1 and `0.537 m` regret, below the existing graph logit's `58.71%`
and `0.522 m`. This small offline model is not a deployable result, but it
rejects the assumption that a simple feature concatenation immediately fixes
ranking.

A pairwise linear objective on the same training records also failed to improve
validation ranking (`54.60%` top-1, `0.535 m` regret). Its learned weight was
dominated by the existing graph logit, with little contribution from the added
features. The negative result makes a simple post-hoc ranker an insufficient
next step and reinforces the need to inspect the navigator representation or
training target before adding adaptive abstraction.

Frozen instruction-conditioned GraphMap embeddings give only weak additional
evidence. With 320 training episodes from two scenes (2,919 states and 10,808
eligible candidates), a pairwise embedding ranker achieved `0.507 m` validation
regret versus `0.522 m` for graph logit, but top-1 was lower (`57.34%` versus
`58.71%`) and the paired regret bootstrap 95% CI was `[-0.060,+0.031] m`.
Pointwise embedding regression was worse (`0.551 m`). This does not justify
integration or a claim that richer embeddings solve graph ranking.

The larger six-scene extraction contains 960 training episodes, 7,813 decision
records and 29,710 eligible candidates. It does not change that conclusion on
the same 511 validation decisions: graph-logit regret is `0.522 m`, while the
pointwise and pairwise embedding rankers reach `0.510 m` and `0.511 m`. Their
paired bootstrap intervals against graph logit are `[-0.040,+0.015] m` and
`[-0.053,+0.031] m`; pairwise top-1 is actually lower (`57.14%` versus
`58.71%`). The small regret reduction is therefore not statistically stable and
has not been connected to a trajectory-level improvement.

An independent 64-episode `val_unseen` extraction from a **single scene** gives the same cautious
picture. Graph-logit regret is `0.368 m`; pairwise embedding is `0.347 m`
(`63.78%` versus `62.37%` top-1), but the paired bootstrap interval is
`[-0.052,+0.010] m`. Pointwise embedding is worse at `0.385 m`. This is a
promising directional signal for a future ranker study, not evidence sufficient
to deploy the scorer or reopen the abstraction selector.

The expanded logs also expose a GraphMap representation issue. In `49.4%` of
7,813 states, at least two waypoint proposals were merged into one real or
ghost graph ID. Within those 4,382 aliased graph groups, privileged one-step
progress differed by `0.250 m` on average and by up to `1.200 m`. This means a
graph ranker cannot distinguish all candidate-level labels that the oracle
analysis treats as separate actions. An explicit `merge_ghost=false` control on
the matched 64-episode validation sample reduced the aliasing-state rate from
`52.6%` to `12.4%`, but navigation did not improve: SR stayed `79.69%`, while
SPL changed from `70.16%` to `69.65%` and nDTW from `73.61%` to `73.01%`.
Ghost count increased from `21.53125` to `28.50` per episode on the matched sample
(the previously quoted `23.85` was from the full unseen split). Thus aliasing is a
real representation ambiguity, but removing it alone is not a justified method
and may increase graph cost without recovering performance.

The oracle supports a separate graph-ranking diagnostic, not a learned
abstraction selector. The fixed navigation comparison and the corrected
best-fixed oracle margin make the current NMS-based adaptive-abstraction
formulation a **NO-GO**. Any future abstraction work needs a new action
representation or stronger evidence than changing NMS density.

## 5. Failure decomposition

The first eight-episode diagnostics found zero decisions with no waypoint proposal in any fixed level. Negative one-step progress occurred in 30/73 default, 45/93 coarse and 33/65 fine decisions. These are observable symptoms, not mutually exclusive failure causes: a negative-progress decision can reflect selection, execution, recovery or a poor proposal. The detailed decomposition is in `FAILURE_ANALYSIS.md`; a matched-state oracle is needed to separate abstraction failure from selection failure.

## 6. Proposed method

No adaptive model has been implemented. The first controlled variants are fixed proposal-density proxies only:

| Variant | NMS setting | Intended diagnostic interpretation |
| --- | --- | --- |
| coarse | max 3, sigma `(12,7)` | sparse proposals with stronger suppression |
| default | max 5, sigma `(7,5)` | released ETPNav |
| fine | max 8, sigma `(4,3)` | denser local proposals |

The future deployable selector, if justified by the oracle and fixed comparisons, must use only current visual/topological/instruction features already available to the agent. No reference path or goal distance may enter that selector.

## 7. Training

Abstraction-selector training has not started and is gated off by the current
NO-GO. Separate frozen-feature pointwise/pairwise rankers have been fit offline;
none controls navigation. No new RL or VLM component is justified by these results.

## 8. Experiments

Baseline smoke, current full reproduction and fixed coarse/default/fine results
are in `research/results/baseline/` and
`research/results/action_abstraction/`. The full fixed comparison is complete
for R2R-CE; diagnostics are currently on the smaller matched episode sample.
All runs use separate experiment names, commands and output files.

The graph-ranking follow-up now includes a six-scene, 960-episode training
extraction, an offline embedding-ranker evaluation, and a matched GraphMap
no-merge control. These remain diagnostic studies; no fitted ranker has been
inserted into the evaluation loop.

## 9. Ablations

Required after the hypothesis survives the first gate: fixed coarse, default, fine, random selector, heuristic selector, learned selector, oracle selector, and removal of instruction/topology/uncertainty features. No ablation should be interpreted before the fixed-resolution and oracle studies establish an action-abstraction gap.

## 10. Limitations

The current coarse/fine definitions change NMS density and suppression radius; they do not yet create a separately learned macro waypoint generator or trajectory action. Scenario labels are geometric proxies, not semantic annotations. One smoke episode and existing single-seed full baselines cannot establish generalization or statistical significance.

## 10.1 Novelty check status

The codebase scan is now supplemented by primary-source abstract/metadata
screening (`LITERATURE_CHECK_20261007.md`). It covers ETPNav, frontier macro-action
closed-loop RL, DifNav, trajectory-centric Beyond Waypoints, AgenticNav and
SparseNav. Generic graph PPO, trajectory/controller consistency and hybrid
candidates already have relevant prior work. Full-method/code inspection and
the wider hierarchical/dynamic-action-space literature remain incomplete.
No claim of novelty or absence of an equivalent method is made.

The follow-up method audit above identifies prior overlap; associated-code and
broader-literature coverage remain incomplete.

## 10.2 Evaluation-validity follow-up

RANKER-VALIDITY-001 exactly reproduced historical offline metrics. Whole-episode
bootstrap intervals for pairwise-minus-graph regret are `[-0.08892,+0.05327] m`
on val_seen and `[-0.05232,+0.00929] m` on unseen. Both cross zero; unseen has
one scene and no valid scene-generalization interval. The 960 training episodes
contain 328 routes, with no train/validation overlap on scene/trajectory ID.

There are 3,937 training pairs with identical graph features and conflicting raw
candidate progress labels (8.77% of pairs). Mean/min/max deduplication sensitivity
also fails to produce a stable improvement. These aggregates are not full-option
labels. Geometric and code audits confirm that a raw direct-forward probe omits
merged ghost geometry, front/back control, quantized turns and tryout. The earlier
selection/execution decomposition is not a causal identification of those losses.
Selected-option calibration now passes as a separate measurement gate;
current NMS-selector development stays NO-GO. The baseline's effective tryout is
false because sliding is enabled; potential tryout behavior belongs to a
different controller configuration.

## 10.3 Full-option calibration

`OPTION_CALIBRATION_REPORT.md` records the new opt-in trace environment and
isolated simulator replay. On the matched seen64 sample, all 511 options pass
both sequential and isolated replay, with zero endpoint/progress error, exact
primitive/collision sequences and 511/511 observation-hash matches per mode.
The untraced control matches every one of 1,152 per-episode metrics exactly.
Coverage includes 64 STOP options, 29 nonempty back paths and 376 collision
events; stochastic tryout is not covered.

Actual selected execution has 91/447 ghost options with a collision and 58/447
more than 0.5 m horizontally from the selected target. These are symptoms, not
causal execution-failure counts. They reinforce that the old 0.052 m proxy
cannot establish negligible execution error. A frozen balanced unseen extension
covers all 11 scenes, six distinct routes each, without training-route overlap.
The next all-option experiment is specified in `OPTION_LABEL_PROTOCOL.md`;
selected-action replay alone does not validate any alternative label.

The balanced unseen extension completes on 66 routes from 11 scenes: all 600
options pass both replay modes with zero endpoint/progress error and exact
primitive/collision sequences. All 1,188 episode metrics match the untraced
control. Subset SR/SPL/nDTW is 66.67/54.55/60.56%, not a full-split benchmark
result. Its 534 ghost actions include 88 with a collision and 48 with horizontal
target residual over 0.5 m. Across both samples, 1,111 selected options pass in
each mode; no alternative-action labels or new learned gains are established.

## 11. Current conclusion

### Controller-consistent follow-up

`GRAPH_OPTION_REPORT.md` now completes the full-option study. The training
pilot covers eight routes/four scenes (63 states, 760 admissible branches),
with all identity, selected sentinel, order-repeatability and baseline controls
passing. Raw mean local progress opportunity is only 0.038215 m and joint
primitive/path-capped opportunity is zero; no model is trained on this pilot.

The preselected unseen66 sample covers 11 scenes, 600 states and 7,113 branches.
All 600 selected sentinels and 1,516 order checks pass with zero endpoint error;
all 1,188 episode metrics match untraced and previously archived baseline values.
On 534 non-STOP states, raw mean local opportunity is 1.284608 m, but joint
primitive/path caps reduce it by 88.42% to **0.148787 m**. Fifty-nine states have
capped gain >=0.25 m, across 28 episodes; all 11 scene means are positive.
Scene-cluster CI is [0.093913, 0.204321] m, conditional on this fixed diagnostic
sample and privileged local max. No episode SR/SPL/nDTW gain is established.

The broader training and ordered-route follow-up now completes. Sixty-four
new routes / 16 training scenes yield 474 states and 5,090 admissible branches;
all 474 sentinels, 1,359 reverse checks and 1,152 baseline metric comparisons
pass. On 410 non-STOP states, joint-cost-capped local opportunity is 0.053452 m;
the frozen matched-prefix route gate reduces it to **0.042087 m**, with 14
material states in only seven routes / five scenes. Seven of the 14 occur in
one route; this concentration is a limitation, not independent training support.

On unseen66, the same route gate retains **0.126962 m** mean local opportunity
and 49 material states over 26 routes / 11 scenes, versus 0.148787 m and 59
before the route gate. Median state gain is zero in both splits. All 414 native
path-length/nDTW/SDTW reconstructions across 138 episodes are exact. These are
privileged geometric prefix diagnostics, not semantic certificates or closed-loop
gains. Some retained alternatives still have negative absolute goal progress.
See `ROUTE_COMPATIBILITY_REPORT.md` for protocols, CIs, limitations and examples.

**NO-GO for Adaptive Action Abstraction as a standalone primary contribution under the current fixed-NMS formulation.** The full unseen comparison showed a tradeoff, but the matched 64-episode `val_seen` fixed comparison has default best on SR/SPL/nDTW (`79.69/70.16/73.61`), while the statewise abstraction oracle exceeds the best fixed candidate level by only `0.0287 m` in one-step candidate progress. Earlier privileged ranking/STOP experiments motivated graph-decision follow-up, but their joint `100.00/98.26/92.58` result changes STOP and cannot establish a ranking-only gain. A six-scene embedding ranker still has paired regret intervals crossing zero against the existing graph logit, with action-label contract limitations detailed in `RANKER_VALIDITY_AUDIT.md`. GraphMap aliasing is measurable, but removing merging leaves SR unchanged and slightly lowers SPL/nDTW. Current decisions therefore rely on the calibrated complete-option and native-continuation evidence below. Further work should be reported as a graph-decision study or require a redesigned abstraction whose action representation changes beyond NMS density.

**CONDITIONAL GO for a narrow graph-decision efficiency study; do not train on
the current one-step oracle labels yet.** The previously frozen seven-route
single-intervention experiment is now complete (`SINGLE_INTERVENTION_REPORT.md`).
Disabled control matches all 474 traces and 1,152 episode metrics exactly;
enabled intervention preserves all 57 other routes and all treated prefixes.
Seven actual alternatives reproduce their calibrated probes exactly, followed
by the unchanged navigator's natural continuation.
Independent reconstruction of 384 final metrics across the 64 actual trajectories
matches exactly; the existing tracked core diff is preserved for this cycle.

On the targeted seven training routes, SR remains 4/7. SPL improves
46.0484 -> 51.6054%, nDTW 63.7303 -> 67.9689%, but two failed routes have worse
nDTW and final goal error. Total primitives increase 674 -> 711. None of the
three failures is rescued, and no successful route is lost. Across the original
64-route population, SR remains 85.9375%, SPL rises 0.6078 pp and nDTW 0.4636 pp;
these are **privileged training-subset interventions**, not learned or validation
scores. The unchanged 57 routes are controls, not independent treated samples.

A strong local label can reverse on continuation: in episode 6962, the step-10
alternative gains 3.5544 m locally, but the next action chooses displaced ghost
g24; forced historical-node STOP later adds a return path. Final error increases
3.5230 -> 8.5082 m and primitive count rises by 65. Only this state on that route
was intervention-tested; do not invalidate other untested states by association.

Some efficiency benefit survives, but recovery and episode-level cost benefits
were unproven in that seven-route sample. Its two harmful cases reverse goal
advantage within two high-level decisions, an explicitly post-hoc observation
that motivated the now-completed prospective study below.

### Prospective short-continuation gate and automatic follow-ups

`CONTINUATION-LABEL-TRAIN64-001` freezes 64 new routes / 16 training scenes,
excluding all 19 previously inspected training scenes. Baseline/control metrics
match exactly; 469 selected sentinels, 1,365 reversed checks and all 6,699 branch
executions pass their gates. On 405 non-STOP states, joint-cost local gain is
0.025680 m; the unchanged route gate leaves **0.008900 m** and only six material
states in five routes / three scenes. This is below the registered scene-coverage
gate for learning. All five eligible routes were already baseline successes.

The frozen one-intervention H=1 schedule and a separately executed H=2-confirmed
schedule both complete. On the same five eligible routes, baseline / H=1 / H=2
SR = 5/5 throughout, SPL = 66.0905 / 73.9205 / 69.6394%, nDTW = 68.0487 /
77.0138 / 70.6196%, primitive totals = **692 / 478 / 692**. The H=2 mixed run
matches all 469 composed trace records exactly, and both actual enabled runs
independently reconstruct 384 final metrics each. These remain privileged
training outcomes, not learned or validation improvements.

H=2 rejects the adverse 10121 case but also useful 1773 and 1374. Route 1773
alone saves 244 primitives, including 70 in native STOP return behavior. Without
it, the other four H=1 cases use 30 more primitives. One of the two accepted H=2
cases is already terminal by H=2, limiting its value as long-term prediction
evidence. No failed route is rescued by these selected interventions.

Two automatic descriptive follow-ups are complete: full cost accounting across
all twelve old/new cases, and an equal-primitive-budget check calibrated against
456 exact native geodesic endpoint measurements. Duration normalization still
rejects 1773 and admits harmful old case 9045. These post-hoc results do not
validate another horizon, budget or selector.

The short-window study's next gate, described in the immutable
`FULL_RETURN_LABEL_NEXT_GATE.md`, has now been executed. Its results follow;
all earlier short-window adverse cases remain in `CONTINUATION_LABEL_REPORT.md`.

### Outcome-blind complete-return gate (2026-10-07)

The predeclared `FULL-RETURN-LABEL-TRAIN16-001` study sampled 16 routes from eight fresh-to-diagnostics training scenes and ran two distinct candidate alternatives from one outcome-blind state per route through native STOP. All controls and 96-metric independent reconstructions per enabled schedule passed.

Across 32 alternatives, only one dominated the baseline on success, SPL, nDTW, goal error, path length and primitive count. It covered one scene and rescued one baseline failure. The highest-logit schedule had mean SPL/nDTW changes of `-15.23/-12.08` points and `+15.5` primitives; the seeded schedule had `-28.77/-24.27` points and `+63.0` primitives. The rescued route was farther from the goal than baseline at both H=1 and H=2 and improved only after later continuation.

**Decision: NO-GO for fitting a selector on this pilot.** This full-return evidence is too sparse and adverse for an abstraction controller, ranker, RL objective or VLM feature. The original NMS abstraction remains NO-GO. The next diagnostic step is a terminal/recovery symptom census on existing calibrated baseline traces, keeping cohorts separate. Do not resample simply to bypass the failed learning gate. Details and plots are in `FULL_RETURN_LABEL_REPORT.md`.

### Terminal/recovery symptom census

A descriptive census over accepted baseline traces separates two cohorts. In the
64-route training capture, 9 failures never had an endpoint within 3 m of the
goal; 3 episodes used forced STOP, and all 3 were failures. In the 16-route
full-return capture, 2 failures split into one never-arrived episode and one
arrival-then-departure episode (`2508`); one forced STOP occurred and it was not a
failure. Endpoint samples can miss proximity between high-level decisions, so
these counts do not identify a causal STOP or recovery defect. The machine
readable census is `research/results/terminal_recovery_census_001/summary.json`.

The broader calibration cohorts reinforce the separation between an endpoint
symptom and a causal diagnosis: val_seen64 has 13 failures, 4 with an
endpoint-level success-neighborhood visit; val_unseen66 has 22 failures, 1 with
that symptom. Native `oracle_success` counts show that some episodes visited a
success-compatible state between recorded endpoint decisions, so the census is
reported as a symptom only. The val_unseen census is archived at
`research/results/terminal_recovery_census_valunseen_001/summary.json`; the
val_seen census is under `terminal_recovery_census_003/`.

### Full-return feature audit (post-hoc)

The pre-action graph fields were audited against the already accepted 32
full-return alternatives without fitting. Commanded polyline difference was
strongly associated with final path and primitive cost (pooled Spearman `0.877`
and `0.851`) and negatively associated with SPL/nDTW (`-0.615/-0.775`). This is
an expected cost proxy and candidate-source confound, not a quality predictor.
The native logit gap was also unstable across top-logit and seeded alternatives.
The paired top-versus-seeded success, SPL and nDTW scene intervals all crossed
zero. **NO-GO for feature-based selector fitting on this data.** Details:
`FULL_RETURN_FEATURE_AUDIT_REPORT.md`.


Follow-up cost accounting ties all 32 primitive/collision deltas to native
metrics. Top alternatives add 109 immediate + 139 later navigation = 248
primitives; seeded alternatives add 330 immediate + 643 later navigation + 35
STOP = 1,008. Geometry versus later navigation cost still correlates at
0.629/0.700 across the respective schedules. This post-hoc association motivates
measuring continuation costs; it does not establish a cost penalty or learned
policy. The original abstraction and current selector-fitting NO-GO decisions
remain unchanged. Twenty-three relevant tests pass; baseline core diff is unchanged.

### Branch alignment and native STOP feasibility diagnostics

At the 16 full-return sampled states, all 32 scheduled alternatives had lower
one-step goal progress than the native selected action. This is a shortlist
limitation: it leaves one delayed rescue (`2508`) but no locally positive
full-return examples. It cannot be used to conclude that all graph alternatives
lack value. Details are in `FULL_RETURN_BRANCH_ALIGNMENT_REPORT.md`.

A separate read-only STOP audit checked all 1,738 accepted baseline decisions
across five cohorts. In train64, none of 9 failed routes had a successful native
STOP branch. In continuation-train64, 2 of 8 failures did; in full-return16, 1
of 2 did; in unseen66, 1 of 22 did. Primitive-level navmesh reconstruction
located interior arrival on 2 continuation failures, route 2508 and two unseen
failures; only unseen route 1593 had no successful high-level STOP endpoint.
The result supports a few STOP-timing opportunities, not a STOP-only bottleneck.
See `NATIVE_STOP_FEASIBILITY_REPORT.md`.

### Interruptible execution audit

The unseen1593 interior-arrival symptom does not directly support a mid-option
STOP head. In normal non-video execution, primitive control uses
`sim.step_without_obs`; the high-level policy receives an observation after the
control segment. The trainer consumes the selected ghost and updates its graph
predecessor before stepping the environment. Interruption would therefore need a
new transaction contract for pending targets, partial costs, collisions, RNG and
observations. Executed physics, costs, collisions and RNG must be preserved;
only pending graph/control intent may be committed or cancelled.
This is an implementation redesign with an explicit information
budget. Beyond Waypoints has a related but distinct pre-execution trajectory
truncation mechanism; no novelty claim is made. The static audit passes 13
source-anchor checks, which are not dynamic tests, and is documented in
`INTERRUPTIBLE_EXECUTION_AUDIT.md`.


### User-directed pivot: failure-critical complete returns

Coarse/default/fine selector development is closed; earlier results remain
negative diagnostics. New controlled interventions enumerate 509 native actions
at all 46 registered critical states in nine training failures. Six routes have
a native non-STOP rescue; four also avoid nDTW loss. Two termination rescues
overlap. The four matched collision-interruption trials rescue none. Baseline,
physical prefix/RNG, cost and independent final-metric checks pass. This supports
a CONDITIONAL GO for long-horizon graph choice investigation, not a deployable
method or a complete failure partition. Three routes remain unresolved.
Details and explicit scope limits: `CRITICAL_CAUSAL_REPORT.md`.

### Cross-scene failure-critical confirmation

Applying the frozen critical-state protocol to 22 failures in the previously
inspected unseen66 diagnostic subset covers 71 states and 985 full native action
returns. Seven routes are rescued by a non-STOP replacement; six also preserve
nDTW. Six termination replacements rescue, overlapping the graph-choice set.
Four collision interruptions again rescue none, while sensing-only controls are
exact. Combined route-quality-aware graph-choice evidence is 10/31, termination
8/31, and interrupt 0/8; unresolved cases remain in the denominator. This is
cross-scene confirmation, not an untouched held-out benchmark. No unseen result
is used for training.

The next gate is a training-only, scene-grouped graph-value/preference feasibility
study with unseen labels held out, defined in `GRAPH_VALUE_NEXT_GATE.md`. It does
not authorize selector training from the current retrospective outcomes.
