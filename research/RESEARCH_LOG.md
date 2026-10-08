# Adaptive Action Abstraction Research Log

## 2026-10-06 / AUDIT-001

**Question:** What action abstraction does the current ETPNav implementation actually expose?

**Change:** Read the trainer, waypoint predictor, NMS, GraphMap, navigator, controller, configs, checkpoint metadata and evaluation code at commit `1c1a794`.

**Control:** No navigation behavior changed during the audit. The later diagnostic extension is disabled by default.

**Result:** ETPNav predicts a `[120 angle x 12 distance]` heatmap, applies circular NMS with at most 5 proposals and `sigma=(7,5)`, creates variable-length ghost nodes, then chooses one graph action from `[STOP, visited nodes, ghosts]`. A selected ghost is converted to a front-node/backtrack operation followed by continuous low-level control.

**Interpretation:** The current fixed abstraction is specifically a fixed waypoint proposal-density rule plus a variable graph action set. It is a suitable target for a controlled action-abstraction diagnostic. It is not correct to describe the current agent as having a fixed number of graph actions.

**Next:** Run fixed coarse/default/fine NMS-density variants and record state-conditional outcomes.

## 2026-10-06 / BASELINE-001

**Question:** Can the current code reproduce the known baseline before adding abstraction variants?

**Change:** Evaluated the released R2R checkpoint on one `val_seen` episode with single GPU, `RL_TOPO.ENABLED=False`, `IL.back_algo=control`, and the published language checkpoint. Also archived the existing full R2R/RxR reference metrics.

**Control:** Same simulator 0.1.7, EGL headless mode, checkpoint, language weights, sensors, success distance (3 m), and evaluation code.

**Result:** Smoke: SR 1.000000, SPL 0.646941, nDTW 0.833686, sDTW 0.833686, NE 1.157351, path length 7.042108, 59 primitive steps, 6 high-level decisions, 0 collisions. It matches the prior one-episode smoke metrics. The first attempt exposed and the second attempt repaired old-checkpoint optimizer-state incompatibility.

**Interpretation:** Baseline execution is valid for the next controlled tests. The repair is compatibility-only and does not alter inference behavior.

**Next:** Enable diagnostics on a small fixed split and compare proposal-density variants.

## 2026-10-06 / DIAG-FAIL-001

**Question:** Can the diagnostic logger query reference-path distance on every decision without changing the worker protocol?

**Change:** The first 16-episode default diagnostic run called the existing `current_dist_to_refpath` worker method with the route object on every decision.

**Control:** No policy or checkpoint change.

**Result:** The legacy Habitat worker exited with `EOFError` during the route call before metrics were produced. The failed stdout is retained as `research/results/action_abstraction/default_val_seen_16.stdout`.

**Interpretation:** Passing the full route through `VectorEnv.call_at` is unsafe in this legacy setup. This run is invalid and excluded from comparisons.

**Next:** Add a worker-local `current_episode_ref_distance()` method and rerun. The one-episode smoke and all 8-episode fixed diagnostics then completed successfully.

## 2026-10-06 / DIAG-001

**Question:** Does changing waypoint proposal density alter the available action abstraction and its state-conditional behavior?

**Change:** Added opt-in `default`, `coarse`, and `fine` NMS specs and an append-only JSONL decision logger. Default is the original `(max_predictions=5, sigma=(7,5))`; coarse uses `(3,(12,7))`; fine uses `(8,(4,3))`.

**Control:** Same ETP waypoint predictor, panorama encoder, GraphMap, cross-modal navigator, low-level controller, checkpoint, task split, and evaluation metrics. The variants do not use reference routes or goal distance to generate actions.

**Result:** Pending fixed-variant runs.

**Interpretation:** These are proposal-density proxies. They can support or weaken the hypothesis, but they do not yet establish a new adaptive model.

**Next:** Complete the small R2R comparison, then use the diagnostic records to test whether scenario proxies prefer different densities.

## 2026-10-06 / FIXED-002

**Question:** Do fixed waypoint proposal granularities produce different navigation behavior under the same ETPNav navigator?

**Change:** Evaluated coarse `(max_predictions=3, sigma=(12,7))`, default `(5,(7,5))`, and fine `(8,(4,3))` on the same first eight `R2R-CE val_seen` episodes with the released checkpoint and `IL.back_algo=control`. Enabled JSONL diagnostics.

**Control:** Same checkpoint, language weights, simulator, seed, sensors, GraphMap, cross-modal navigator, low-level controller, success distance and episode limit. Only NMS proposal density changed.

**Result:** Coarse/default/fine achieved SR `50.0/87.5/62.5%`, SPL `32.31/68.92/39.99%`, and nDTW `41.35/70.86/49.73%`. Mean candidates per decision were `2.43/4.38/7.37`; mean high-level decisions were `11.63/9.13/8.13`. Positive one-step progress rates were `51.6/58.9/49.2%`.

**Interpretation:** Proposal density is behaviorally consequential; the switch is not inert. This is evidence for a conditional continuation, not evidence that an adaptive selector is better: default wins this small sample, and trajectories diverge after the first decision. Proxy scenario frequencies also change with the action space.

**Next:** Implement matched-state simulator restore/probing for the oracle adaptive abstraction experiment. If that cannot be made faithful, do not call a sequential per-episode best-of-three result an oracle.

## 2026-10-06 / FIXED-003-FULL

**Question:** Does the proposal-density effect persist on the full R2R-CE unseen split?

**Change:** Evaluated coarse and fine fixed NMS variants on all 1,839 `val_unseen` episodes. The current-code default full run was evaluated in the same session and matched the published baseline.

**Control:** Same checkpoint, simulator, two-GPU/8-env protocol, controller, success definition, episode limit, and evaluator. Only `ACTION_ABSTRACTION.LEVEL` changed.

**Result:** Coarse/default/fine SR were `54.2143/57.0962/57.5313%`; SPL `45.4745/49.1014/47.2602%`; nDTW `59.6499/62.3051/60.0452%`. Fine used 90.489 primitive steps and 26.794 ghosts per episode versus default 80.274 and 23.850. Fine-minus-default paired deltas were `+0.4350` SR points, `-1.8413` SPL points and `-2.2598` nDTW points. Coarse was lower on all three.

**Interpretation:** Action abstraction is a material behavior/efficiency axis. Fine can trade efficiency for a small SR increase, while coarse is harmful under this fixed navigator. This supports a conditional adaptive/cost-aware investigation, but no adaptive gain has been shown and the fine SR interval includes zero.

**Next:** Implement a faithful same-state oracle with simulator state restore. Compare its Pareto frontier to default and fine before training any selector.

## 2026-10-06 / ORACLE-001

**Question:** Does the preferred waypoint proposal abstraction vary at the same navigation state?

**Change:** Added an opt-in oracle probe. One waypoint heatmap is computed per decision; coarse, default and fine candidates are generated from that same heatmap. Each candidate is evaluated with the worker-local `cand_dist_to_goal` method, which restores the simulator pose after probing. The oracle's goal-progress choice is logged separately and never controls execution.

**Control:** Eight `R2R-CE val_seen` episodes, released checkpoint, default fixed execution, same simulator seed, observation pipeline and navigator. Only the analysis-only probes add work.

**Result:** 73 matched decisions were recorded. Mean best-candidate one-step progress was coarse `0.958 m`, default `1.107 m`, fine `1.327 m`. The privileged oracle selected default on 39 decisions (`53.4%`) and fine on 34 (`46.6%`); coarse was never best. Fine's mean advantage over default was `0.220 m`.

**Interpretation:** Proposal availability has state-dependent potential in this small sample. The result is an upper bound at the candidate level, not an adaptive navigation result: the oracle uses goal distance and bypasses the learned graph-action selection. It therefore supports a conditional continuation while leaving selection and execution as unresolved bottlenecks.

**Next:** Expand the matched-state probe and add a fair option-selection analysis. Stop before selector training if fine's candidate advantage does not survive navigator selection or if it worsens the efficiency frontier.

## 2026-10-06 / COMPAT-ORACLE-001

**Question:** Did exposing the oracle heatmap change default navigation when the oracle is disabled?

**Change:** Ran the post-oracle default smoke with `ORACLE_ENABLED=False`.

**Control:** Same released checkpoint, first `val_seen` episode, simulator and controller as BASELINE-001.

**Result:** SR `1.000000`, SPL `0.646941`, nDTW `0.833686`, navigation error `1.157351`, path length `7.042108`, 59 primitive steps, 6 high-level decisions and 0 collisions. Stdout is saved at `research/results/action_abstraction/postoracle_smoke.stdout`.

**Interpretation:** The opt-in heatmap return and oracle instrumentation preserve the default smoke behavior.

**Next:** Expand matched-state probing before implementing any learned selector.

## 2026-10-06 / ORACLE-002

**Question:** Does the candidate-level abstraction preference persist beyond the initial eight episodes?

**Change:** Expanded the same-state privileged oracle to 64 `R2R-CE val_seen` episodes and 511 high-level decisions. Added distance-to-goal conditional summaries and paired bootstrap intervals.

**Control:** Default fixed execution, same checkpoint, seed, simulator, sensors, graph navigator and low-level controller. Coarse/default/fine candidates were generated from one heatmap at each state; only counterfactual worker rollouts were added.

**Result:** Best-candidate progress was coarse `1.116 m`, default `1.266 m`, fine `1.552 m`. Fine-minus-default was `+0.285 m` with bootstrap 95% CI `[+0.243,+0.329]`. Oracle selection was default `50.7%`, fine `49.3%`, coarse `0%`. Default won `62.8%` of states within 3 m of goal; fine won `60.8%` of states 3–6 m from goal.

**Interpretation:** The state-dependent candidate-level signal is stronger and more stable than in the eight-episode probe. It remains a privileged upper bound and does not imply an end-to-end adaptive gain.

**Next:** Measure how much of this potential is lost in the existing graph-action selection and low-level execution.

## 2026-10-06 / ORACLE-003

**Question:** Is the current default navigator able to realize the best candidate available in its own action space?

**Change:** Repeated the 64-episode default rollout with ordinary decision diagnostics and the same-state oracle, then matched records by episode and high-level step.

**Control:** Same trajectory protocol and fixed default execution; oracle records remain analysis-only.

**Result:** Realized default progress averaged `0.610 m`, while the best default candidate at the same state averaged `1.266 m`. The combined candidate-selection and execution gap was `0.656 m` over all 511 decisions and `0.672 m` over 447 ghost actions. The actual gap is larger than the `0.285 m` fine-versus-default candidate advantage.

**Interpretation:** Selection and execution are currently a larger measurable bottleneck than proposal abstraction. This prevents a responsible claim that a learned abstraction selector will improve navigation.

**Next:** Instrument candidate-to-ghost identity and controller reachability, or run a fixed-action selection oracle, before selector training.

## 2026-10-07 / GRAPHSEL-001

**Question:** Does the large default progress gap come from graph-action ranking or low-level execution?

**Change:** Added an analysis-only `GraphMap.last_candidate_vps` mapping and logged each current waypoint candidate's graph node, graph index, validity, visited mask, navigator logit and whether the policy selected it. Paired a 64-episode default diagnostic run with the same-state oracle records.

**Control:** Same checkpoint, seed, split, default proposal abstraction, GraphMap, navigator and controller. The mapping is disabled from behavior and only records existing graph state.

**Result:** All 2,295 default candidates mapped to a graph node. For 447 ghost decisions, 95.3% selected a current candidate ghost. The best valid/unvisited graph candidate averaged `1.072 m` progress; the selected candidate's counterfactual progress averaged `0.729 m`. Graph selection gap was `0.471 m`; execution gap after the selected candidate was `0.052 m`.

**Interpretation:** Graph-action ranking is the larger measured downstream bottleneck. The current controller is relatively reliable once a candidate ghost is selected. A learned abstraction selector alone is unlikely to recover the full oracle potential.

**Next:** Use a privileged graph-action selection oracle or candidate-ranking diagnostic to test how much end-to-end gain remains after replacing only the graph choice. Keep abstraction selection fixed during that test.

## 2026-10-07 / GRAPHSEL-ORACLE-001

**Question:** How much end-to-end performance is recoverable by fixing graph-action ranking while keeping waypoint abstraction fixed?

**Change:** Added an opt-in privileged intervention. At each evaluation decision, current valid/unvisited ghost candidates are counterfactually executed in the worker simulator; the candidate with largest one-step goal progress is executed. The default waypoint heatmap, NMS, GraphMap construction and low-level controller remain unchanged.

**Control:** Default abstraction, released checkpoint, R2R `val_seen`, same seed and simulator. Goal distance is used only by the analysis oracle.

**Result:** On 64 episodes, the intervention reached SR `100.00%`, SPL `98.26%`, nDTW `92.58%`, path length `7.746 m`, and `6.313` high-level decisions. The fixed-default 64-episode rollout was SR `79.69%`, SPL `70.16%`, nDTW `73.61%`, path length `9.724 m`, and `7.984` decisions. The oracle changed 90/404 decisions (`22.3%`).

**Interpretation:** Graph ranking is a large recoverable bottleneck. This privileged trajectory is an upper bound and cannot be claimed as a method, but it changes the next deployable priority from abstraction selection to a fair learned candidate ranker.

**Next:** Build a non-privileged graph-ranker diagnostic or supervised ranker using existing visual/topological/instruction features. Keep the action abstraction fixed while measuring whether it recovers part of the oracle gap.

## 2026-10-07 / RANKATTR-001

**Question:** Which existing non-privileged signals explain the privileged best-candidate ranking?

**Change:** Offline paired 511 default decisions with their candidate-to-graph mappings and privileged one-step progress labels. Compared current graph logits, waypoint heatmap mass, and simple distance heuristics without changing trajectories.

**Control:** Same matched states and candidate eligibility masks; progress labels are analysis-only.

**Result:** Graph logits had candidate-level Spearman correlation `0.531`, top-1 best-candidate rate `58.7%`, and mean regret `0.522 m`. Heatmap mass had Spearman `0.197`, top-1 `50.1%`, regret `0.854 m`. Distance heuristics were weaker.

**Interpretation:** The existing graph navigator contains useful ranking information but leaves substantial recoverable error. Heatmap confidence alone is insufficient. A small learned/calibrated ranker using existing graph, visual, topology and instruction features is a better next controlled experiment than an adaptive abstraction selector.

**Next:** Define a non-privileged ranker data contract and train only on oracle labels generated from training trajectories, then evaluate fixed default abstraction first.

## 2026-10-07 / COMPAT-ORACLE-002

**Question:** Does adding the graph-selection intervention preserve default behavior when disabled?

**Change:** Ran a one-episode default smoke with both privileged oracle flags disabled after adding candidate-to-graph logging and the graph-selection intervention.

**Control:** Same checkpoint, first `val_seen` episode, simulator and controller as BASELINE-001.

**Result:** SR `1.000000`, SPL `0.646941`, nDTW `0.833686`, navigation error `1.157351`, path length `7.042108`, 6 high-level decisions and 0 collisions. Output: `research/results/action_abstraction/final_compat_smoke.stdout`.

**Interpretation:** The new analysis hooks preserve the fixed baseline when disabled.

**Next:** If continuing toward a deployable method, train/calibrate a non-privileged graph ranker before revisiting adaptive abstraction.

## 2026-10-07 / RANKER-TRAIN-001

**Question:** Can a simple train-to-validation ranker recover the privileged graph-selection signal using deployable features?

**Change:** Collected 64 training-split episodes with default diagnostics and privileged labels. Fit a ridge linear scorer over graph logit, waypoint score, candidate distance/angle, policy entropy and candidate count. Evaluated offline on the 64-episode `val_seen` records.

**Control:** No navigation trajectory used the fitted weights. Validation states and eligibility masks were held fixed.

**Result:** The linear scorer reached `54.99%` top-1 and `0.537 m` regret, below the existing graph logit at `58.71%` and `0.522 m`. Training data contained 485 states and 1,809 candidates; validation contained 511 states and 1,838 candidates.

**Interpretation:** Simple feature concatenation does not immediately improve graph ranking. A deployable ranker would require better labels, pairwise/listwise objectives, richer graph/instruction representations, or a different problem formulation.

**Next:** Do not integrate this linear scorer. If pursuing ranking, test a strictly controlled pairwise/listwise objective; otherwise return to the abstraction hypothesis with the graph bottleneck explicitly acknowledged.

## 2026-10-07 / RANKER-EMBED-003

**Question:** Does a frozen instruction-conditioned GraphMap embedding become a reliable candidate ranker with substantially more scenes and training states?

**Change:** Expanded embedding and privileged-label extraction to 960 R2R train episodes across six scenes, yielding 7,813 decision records and 29,710 eligible candidates. Fit the same pointwise and pairwise ridge scorers used in RANKER-EMBED-002.

**Control:** No fitted ranker controlled a trajectory. Validation used the same 511 `val_seen` decisions, candidate masks and privileged one-step progress labels as prior comparisons; graph logit remained the deployable baseline.

**Result:** Graph-logit regret was `0.522 m` and top-1 `58.71%`. Pointwise embedding reached `0.510 m` / `58.71%`; pairwise reached `0.511 m` / `57.14%`. Paired regret bootstrap intervals versus graph logit were `[-0.040,+0.015] m` and `[-0.053,+0.031] m`.

**Interpretation:** More data does not produce a statistically stable offline improvement. The small regret reduction is insufficient to justify integration or a trajectory intervention.

**Next:** Test whether GraphMap ghost merging aliases distinct waypoint proposals, then keep the default representation unless a controlled ablation shows a navigation benefit.

## 2026-10-07 / GRAPH-ALIAS-001

**Question:** Does GraphMap merging collapse geometrically distinct proposals into one graph action and explain part of the ranking gap?

**Change:** Added `research/tools/analyze_graph_aliasing.py`, which matches current candidate-to-graph mappings with same-state oracle progress labels. Ran it on the 960-episode train extraction and a matched validation control with `MODEL.merge_ghost=false`.

**Control:** The aliasing script is analysis-only. The no-merge control keeps checkpoint, split, simulator, seed, waypoint abstraction, navigator and controller fixed; only ghost merging changes.

**Result:** Default merging aliases proposals in `49.4%` of 7,813 train states, with 4,382 aliased groups and mean progress spread `0.250 m` (maximum `1.200 m`). No-merge lowers the matched validation aliasing rate from `52.6%` to `12.4%`, but SR remains `79.69%`, SPL changes `70.16% -> 69.65%`, nDTW `73.61% -> 73.01%`, and ghost count rises `23.85 -> 28.50`.

**Interpretation:** Merging creates a real action-identity ambiguity, but removing it alone does not improve navigation and increases graph size. It cannot be claimed as a fix or as evidence for adaptive abstraction.

**Next:** Preserve the default GraphMap contract, report aliasing as a confound, and stop the current ranker pivot unless a future deployable ranker beats graph logit under multi-scene and trajectory-level controls.

## 2026-10-07 / RANKER-EMBED-004

**Question:** Does the expanded embedding ranker generalize to a separate `val_unseen` sample?

**Change:** Collected default-action embedding/oracle diagnostics for 64 `R2R-CE val_unseen` episodes and evaluated the frozen pairwise and pointwise scorers trained on the 960-episode, six-scene train extraction.

**Control:** No scorer controlled navigation. Candidate masks, graph-logit baseline and privileged one-step labels were held fixed within each validation state.

**Result:** Graph logit regret was `0.368 m` / `62.37%` top-1. Pairwise embedding reached `0.347 m` / `63.78%`; pointwise reached `0.385 m` / `61.77%`. Pairwise regret difference was `-0.021 m` with bootstrap 95% CI `[-0.052,+0.010] m`.

**Interpretation:** The independent split shows a consistent directional pairwise signal, but uncertainty still includes no improvement and no trajectory-level benefit has been demonstrated. This remains a ranker lead, not a reason to reopen Adaptive Action Abstraction.

**Next:** Keep the method at diagnostic status. Any continuation must implement a non-privileged scorer and evaluate trajectory-level paired controls before claiming a graph-ranking improvement.

## 2026-10-07 / RANKER-PAIRWISE-001

**Question:** Does changing the offline objective from pointwise regression to pairwise ranking improve train-to-validation candidate ordering?

**Change:** Fit a ridge linear scorer on 2,676 within-state candidate pairs from the 64 training-split episodes, using the same deployable feature set and privileged progress ordering.

**Control:** No navigation trajectory used the fitted weights; validation states and candidate masks were unchanged.

**Result:** Validation top-1 rate was `54.60%` with `0.535 m` regret, below the graph-logit baseline (`58.71%`, `0.522 m`) and the pointwise linear scorer (`54.99%`, `0.537 m`).

**Interpretation:** A simple pairwise objective does not recover the graph-selection oracle gap. The feature/representation or label interface needs deeper work; adding adaptive abstraction now would confound two unresolved bottlenecks.

**Next:** Keep adaptive abstraction conditional and avoid integrating either offline linear scorer.

## 2026-10-07 / ORACLE-CORRECTION-001

**Question:** Does the abstraction oracle improve over the best fixed abstraction, rather than only over default?

**Change:** Recomputed oracle statistics with tie handling and compared the oracle to the fixed level with the highest mean candidate progress.

**Control:** Same 511 matched decisions and privileged candidate outcomes from ORACLE-002.

**Result:** Fine is the best fixed level by mean candidate progress (`1.552 m`). The statewise oracle exceeds fixed fine by only `0.0287 m` on average (standard error `0.0060 m`). There are 218 default/fine tie states; strict non-tie states favor fine `86.0%` and default `14.0%`.

**Interpretation:** The earlier `+0.285 m` value was fine versus default, not adaptive versus the best fixed level. The standalone adaptive-abstraction gain is small at the candidate level.

**Next:** Complete same-sample fixed coarse/default/fine navigation metrics and factor graph ranking versus STOP oracle effects.

## 2026-10-07 / GRAPHSEL-CONTROL-001

**Question:** How much of the privileged joint graph oracle gain comes from ranking versus STOP decisions?

**Change:** Added analysis-only controls: `rank_only` preserves the learned STOP gate while replacing non-STOP ranking; `stop_only` uses the learned non-STOP ranker while applying the privileged 1.5 m STOP gate.

**Control:** Default waypoint abstraction, same checkpoint, `val_seen` 64 episodes and controller. Goal distance is used only by the interventions.

**Result:** `rank_only` reached SR/SPL/nDTW `96.88/88.24/88.46` and changed 23.97% of decisions. `stop_only` reached `81.25/73.72/73.47` and changed 16.45% of decisions. The previous joint intervention reached `100.00/98.26/92.58`.

**Interpretation:** The large joint-oracle gain is primarily graph ranking, with a smaller STOP contribution. All three are privileged upper bounds.

**Next:** Do not interpret the graph oracle as an adaptive abstraction result; use it to justify a separately scoped graph-ranking study.

## 2026-10-07 / FIXED-VALSEEN-001

**Question:** Do fixed abstraction levels differ on the same 64-episode navigation sample?

**Change:** Evaluated coarse, default and fine with unchanged navigator/controller and the same checkpoint.

**Control:** R2R `val_seen`, 64 episodes, seed 100, control backtracking, success distance 3 m.

**Result:** coarse/default/fine SR were `62.50/79.69/71.88%`; SPL `54.20/70.16/58.83%`; nDTW `66.29/73.61/68.19%`. Default was best on every metric.

**Interpretation:** The matched navigation result weakens the standalone adaptive-abstraction hypothesis. The full `val_unseen` fine SR increase does not establish a general adaptive benefit because it comes with lower SPL/nDTW and does not repeat on this matched `val_seen` sample.

**Next:** Mark the current AAA formulation NO-GO and preserve the artifacts as motivation for a separately scoped graph-ranking investigation.

## 2026-10-07 / COMPAT-CONTROL-003

**Question:** Does the corrected oracle-control code preserve baseline behavior when disabled?

**Change:** Ran default one-episode smoke with abstraction, graph-selection and all privileged oracle flags disabled.

**Control:** Same released checkpoint, first `val_seen` episode, simulator and controller as BASELINE-001.

**Result:** SR `1.000000`, SPL `0.646941`, nDTW `0.833686`, navigation error `1.157351`, path length `7.042108`, 6 high-level decisions and 0 collisions. Output: `research/results/action_abstraction/final_control_smoke.stdout`.

**Interpretation:** The corrected control modes are inert when disabled; baseline behavior remains reproducible.

**Next:** Treat current Adaptive Action Abstraction formulation as NO-GO and maintain graph-ranking work as a separate research question.

## 2026-10-07 / RANKER-EMBED-002

**Question:** Do frozen instruction-conditioned GraphMap embeddings support a robust non-privileged candidate ranker across scenes?

**Change:** Extracted 768-dimensional graph embeddings on 320 training episodes from two scenes (2,919 states, 10,808 eligible candidates), then fit pointwise and pairwise ridge rankers using privileged progress only as offline labels.

**Control:** Validation used the fixed 64-episode `val_seen` records. No fitted score controlled a navigation rollout.

**Result:** Pairwise embedding ranker achieved `0.507 m` regret and `57.34%` top-1 versus graph logit `0.522 m` and `58.71%`. Paired regret difference was `-0.0146 m`, bootstrap 95% CI `[-0.0595,+0.0306]`. Pointwise embedding regression reached `0.551 m` regret.

**Interpretation:** Embeddings contain a weak, unstable ranking signal, but evidence does not support integrating this ranker. Lower regret with lower top-1 and a CI crossing zero is insufficient for a deployable claim.

**Next:** Keep the graph-ranking pivot exploratory; require a stronger held-out multi-scene protocol and trajectory-level intervention before changing the policy.


## 2026-10-07 / RANKER-VALIDITY-001

**Question:** Do the small offline ranking gains survive action-identity checks and resampling that respects correlated episodes/routes?

**Change:** Added a standalone streaming pairing audit, reused the same deterministic frozen-feature fits, reproduced row-wise scores, and ran episode/route/scene bootstrap plus graph mean/min/max label sensitivities. Checked selected ghost geometry against raw proposals. Screened primary literature metadata/abstracts and archived the response.

**Control:** Same checkpoint, frozen observations/features, training records, ridge=100, validation states and labels. No new navigator/controller behavior, sensor, simulator, STOP rule or action-space change. No hyperparameter tuning on validation.

**Result:** Historical learned scorer regret/top-1 anchors reproduce exactly. Pairwise episode-cluster CIs are val_seen [-0.08892,+0.05327] m and unseen [-0.05232,+0.00929] m. Seen has 4 scenes and 23 routes; unseen64 has only 1 scene and 46 routes. Train960 has 328 routes, no train/validation route overlap, and 3,937 pairs with identical embedding but conflicting candidate-progress labels (8.77% of old pairs). Selected-ghost geometry differs from all corresponding raw targets by >10 cm in 75/426 seen and 86/415 unseen ghost decisions. All graph aggregation sensitivities still have pairwise episode CIs crossing zero.

**Interpretation:** No stable deployable gain is established. Raw direct-forward oracle labels are not full merged-ghost/front/back/controller outcomes; earlier 0.471/0.052 m selection/execution gaps are proxy decompositions. The actual privileged rank-only rollout gain remains valid evidence of a possible graph-decision opportunity, not a certified optimal upper bound. Abstract-level literature screening does not establish novelty.

**Next:** Calibrate isolated full-controller option probes against actual selected-action execution before collecting further ranker labels; then preselect balanced held-out scenes. Keep current NMS Adaptive Action Abstraction selector development NO-GO.

**Reporting corrections:** Previous GRAPH-ALIAS-001/table text used full-unseen ghost count 23.85 in a matched seen64 comparison. Correct matched default is 21.53125 versus no-merge 28.50. Old experiment artifacts remain unchanged. The first validity run used batched dot products that changed tied candidate rankings at machine precision; retain it as superseded, use validity_001_rowwise_20261007 for all reported checks. Neither resampling nor deterministic ridge refits count as 3 independent experimental seeds.

## 2026-10-07 / OPTION-CALIBRATION-001

**Question:** Can isolated replay of the actual selected graph option reproduce
the original controller, while tracing leaves baseline evaluation unchanged?

**Change:** Added opt-in typed full-action/primitive tracing and a separate
simulator replay tool. First ran one episode, then all 64 seen episodes in
sequential and reset-based isolated modes. Added an untraced paired evaluation.

**Control:** Released checkpoint SHA256
`4e70d2a1b4a6cfb32158901b7300cbc2be657641f16c9a17f35ca8570470d8b3`,
git HEAD `1c1a794148a774940d59a12dc601ea51febf3d7e` plus source hashes,
seed 100, val_seen64, one RTX 4090/worker, original sensors and controller,
control backtracking, sliding=true, 3 m success and 15 high-level decisions.
All old oracle/embedding hooks and RL are disabled. Effective tryout is false.

**Result:** Both modes pass 511/511 options across four scenes; maximum endpoint
and progress errors are 0 m, rotation error 4.21e-8 rad. Primitive/collision/RNG/
done sequences and observation hashes match. Coverage: 4,293 primitive events,
376 collisions, 64 STOP options and 29 nonempty back paths. Traced/untraced
1,152 episode metrics and 18 aggregates match exactly; SR/SPL/nDTW remain
79.6875/70.1608/73.6074%. Ghost options have 91/447 collision incidence and
58/447 horizontal target residual >0.5 m. See `results/option_calibration/`.

**Interpretation:** Selected-option local execution can be reproduced under this
protocol. This does not validate unselected actions, stochastic tryout, causal
failure decomposition or a deployable improvement. Collision/residual symptoms
also prevent treating the historical 0.052 m execution proxy as negligible loss.

**Next:** Repeat the same preregistered gates on a scene-balanced unseen sample.

## 2026-10-07 / OPTION-CALIBRATION-002

**Question:** Does selected-option calibration hold across all unseen scenes,
rather than only the single scene in the old unseen64 ranker extraction?

**Change:** Preselected six distinct routes from each of 11 unseen scenes by
seeded metadata hashing (selection seed 20261007), one instruction per route.
Ran frozen baseline capture, untraced control and both replay modes on all 66
episodes. Original source samples/vocabulary are preserved in a new subset file.

**Control:** Same checkpoint, simulator, seed 100, GPU, sensors, controller,
sliding, masks and evaluation rules as OPTION-CALIBRATION-001. The only sample
change is explicit in `configs/OPTION_CALIBRATION_002.json`. Source/checkpoint/
subset hashes, logs and per-episode outputs are retained in exclusive directories.

**Result:** All 66 sampled route/episode IDs appear, with zero train-route overlap.
Both modes pass 600/600 options: endpoint/progress errors 0 m, rotation maximum
4.21e-8 rad; every action/collision/RNG/done sequence and observation hash matches.
Coverage: 5,730 primitives, 341 collisions, 66 STOPs, 41 nonempty back paths.
All 1,188 episode metrics and 18 aggregates match the untraced control. Subset
SR/SPL/nDTW=66.6667/54.5468/60.5601%; these are not full-benchmark scores.
Ghost options: 88/534 have a collision, 48/534 target residual >0.5 m. Among
22 failed episodes, ten have a collision, five forced STOP and two native
oracle-success=1; symptoms overlap. All 24 focused tests pass, along with
compilation and diff whitespace checks.

**Interpretation:** The selected-action and baseline noninterference gates now
pass on both seen and multi-scene unseen. This is measurement progress, not
navigation improvement. The NMS-density AAA selector stays NO-GO; graph-decision
diagnostics remain CONDITIONAL GO. No new scorer, RL or VLM is integrated.

**Next:** Follow `OPTION_LABEL_PROTOCOL.md`: capture valid graph IDs and complete
action dictionaries before ghost deletion on a small training pilot, replay
all alternatives with selected-action sentinels and reversed enumeration order.
Only after this passes measure controller-consistent regret and consider fitting.

## 2026-10-07 / GRAPH-OPTION-PILOT-001

**Question:** Do complete, controller-consistent graph action labels reveal a
selection opportunity after preserving the episode horizon and action costs?

**Change:** Added an opt-in capture trainer with two inert-by-default callbacks.
Capture valid graph IDs and full action dictionaries before ghost deletion;
require native selected-action equality. Execute all admissible actions in an
independent simulator, including original selected sentinels and reversed-order
repeats on two metadata-selected states per episode. Save frozen embeddings but
do not fit any model.

**Control:** Eight training routes / four scenes selected before outcomes,
seed 100, released checkpoint, original sensors/controller, sliding=true,
effective tryout=false, high-level limit 15, original STOP and success rules.
Learned/forced STOP states do not enter rank-only regret; final-budget non-STOP
branches are not executed. Source/dataset/checkpoint hashes and unique logs are
under `results/graph_option_pilot/`.

**Result:** 63 decisions, 760 admissible branches, 63/63 native action identities
and selected execution sentinels pass, 218/218 reversed branches pass across
16 states, endpoint errors 0 m and all observation hashes match. Thirty final-
budget branches are excluded. All 144 episode metrics and 18 aggregates match
the untraced baseline, whose training-subset SR/SPL/nDTW is 100/95.62/89.21%.
Across 55 non-STOP states, mean raw local opportunity is 0.038215 m (four positive
states; two >=0.25 m), primitive-capped 0.001024 m, path-capped and jointly capped
0 m. Raw episode-cluster CI [0, 0.077089] m is exploratory. Historical ghosts add
no extra opportunity over current proposals plus selected action on this sample.

**Interpretation:** Fidelity gates pass, but the pilot does not support training
a new ranker. The baseline already succeeds on every sampled training episode;
longer action execution explains the material local progress differences.
This is not sufficient to infer that every unseen state lacks an opportunity.

**Next:** Apply the unchanged analysis to the previously frozen 66-route,
11-scene unseen set, as specified before this pilot. Do not tune or train on it.

## 2026-10-07 / GRAPH-OPTION-UNSEEN-001

**Question:** Does the training-pilot ceiling also hold on unseen scenes when
full graph actions, STOP, the decision horizon and execution costs are controlled?

**Change:** Applied the identical capture/probe/analysis code to the preselected
66-route / 11-scene unseen subset. Executed every admissible graph option and
reversed enumeration on two metadata-selected states per episode. Extended the
primary-paper audit from abstracts to method sections; no model was trained.

**Control:** Same released checkpoint, seed 100, sensors, default action generator,
native graph navigator/controller, sliding=true and effective tryout=false.
Actual STOP states are excluded from rank-only regret; final-budget non-STOP
actions are not executed. Config: `configs/GRAPH_OPTION_UNSEEN_001.json`.

**Result:** 600 states, 7,113 admissible branches; all 600 selected sentinels and
1,516 reverse-order checks pass with zero endpoint error and matching observation
hashes. Two hundred seven final-budget branches are excluded. Fresh control and
pre-hook archive match all 1,188 episode metrics exactly. Native subset SR/SPL/
nDTW stays 66.67/54.55/60.56%. On 534 non-STOP states, raw local gain is 1.284608 m;
primitive-capped 0.200184 m; path-capped 0.342688 m; jointly capped 0.148787 m
(88.42% below raw). Joint-capped scene-cluster CI [0.093913,0.204321] m; 59 states
have >=0.25 m gain across 28 episodes, and all 11 scene means are positive.
Ten of these selected actions collide, 49 do not. Post-hoc outcome stratification:
36/205 states in failed episodes and 23/329 in successful episodes pass the
0.25 m joint-capped threshold. All 36 tests, compilation and whitespace checks pass.

**Interpretation:** A limited local selection opportunity survives faithful
execution and cost controls on unseen scenes. Most unconstrained progress
opportunity is attributable to allowing more costly actions. This is a privileged
fixed-state analysis, not a closed-loop oracle, learned navigation improvement,
causal failure partition or proof of trainability. The zero training-pilot signal
requires broader training coverage before any fitting. Method-level related-work
overlap further narrows the original novelty premise (see method audit).

**Next:** Freeze broader training scenes/routes, collect identical labels, and
audit instruction/reference-route compatibility before fitting a minimal scorer.
Keep current NMS-selector NO-GO; focused graph-decision study CONDITIONAL GO.

**Execution notes:** The first noninterference check began before the control
finished; it failed because results were not yet present. Its dependent probe
failed preflight and executed no branches. Old attempts/logs remain intact.
Accepted runs are `noninterference_002/` and `probe_full002/`; no settings or gates
changed. Source snapshots and environment are under `graph_option_unseen/provenance/`.

**Final integrity audit:** Independently streamed every saved branch, including
reversed repeats: 978 training executions and 8,629 unseen executions, 9,607 in
total. Starting pose/goal distance/RNG, full action identity, admissibility and
coverage all pass. Start-position/goal-distance errors are zero. No partial or
failed probe contributes to the reported metrics.

## 2026-10-07 / GRAPH-OPTION-TRAIN64-001 and ROUTE-COMPATIBILITY-001 (started)

**Question:** Does cost-controlled local opportunity survive ordered route checks and broader training coverage?

**Change:** Freeze 16 hashed training scenes × four new routes, excluding all eight pilot routes; zero route overlap with either validation split. Define an exact ordered-prefix DTW diagnostic before inspecting route outcomes.

**Control:** Original checkpoint, seed 100, native sensors, STOP, controller and decision budget. Same capture/untraced/probe/integrity gates. No model fitting or validation tuning.

**Result:** Collection started. Twelve focused route/sampling tests pass, including shortcut, loop, reverse-motion, required-detour and incremental-DTW cases. Outcomes pending; no new navigation score claimed.

**Interpretation:** Need ordered reference consistency in addition to distance-to-goal opportunity. Proxy limitations are frozen in ROUTE_COMPATIBILITY_PROTOCOL.md.

**Next:** Finish paired control and full-option probes; reconstruct native full-path metrics before applying route gates.

## 2026-10-07 / GRAPH-OPTION-TRAIN64-001 (completed)

**Question:** Does the zero-signal training pilot persist with broader scene
and route coverage, while retaining exact controller/action identities?

**Change:** Before outcomes, selected 16 SHA256-ordered training scenes × four
routes, one instruction/route; excluded all eight pilot routes and verified zero
route overlap with val_seen/val_unseen. Executed the same complete-option pipeline
through a sequential driver that waits for and checks each prerequisite.

**Control:** Released checkpoint, seed 100, native sensors, sliding=true,
effective tryout=false, graph navigator, controller, STOP and 15-decision limit.
No fitting, failure-biased resampling or validation tuning. Config and unique
stdout/stderr manifests are under `results/graph_option_train64/`.

**Result:** 474 decisions, 5,090 admissible forward branches, 474/474 selected
sentinels and 1,359/1,359 reverse checks pass. Independent integrity passes all
6,449 branch executions including repeats; endpoint errors zero and checked
observations exact. Thirty-eight final-budget options excluded. All 1,152 episode
metrics and 18 aggregates equal untraced control; baseline subset SR/SPL/nDTW =
85.94/81.10/83.15%. On 410 non-STOP states, raw local gain 0.228170 m,
primitive-capped 0.065716 m, path-capped 0.091953 m, joint-capped 0.053452 m.
Joint-capped material states: 19, spanning ten routes and seven scenes.

**Interpretation:** Broader training contains some cost-controlled local signal,
unlike the eight-route pilot. It remains sparse and does not establish training
feasibility or a deployable navigation gain. Keep the original AAA NO-GO.

**Next:** Apply the already frozen ordered-reference gate before considering
these labels; do not fit a goal-only scorer.

## 2026-10-07 / ROUTE-COMPATIBILITY-001 (completed)

**Question:** Does cost-controlled local goal opportunity also satisfy an
ordered reference-route proxy, and is training signal sufficiently distributed?

**Change:** Frozen before route outcomes: exact common-prefix DTW, monotonic
baseline cursor, no smaller matched reference endpoint, and no worse cumulative
alignment to the same selected endpoint. Endpoint-only sensitivity declared in
advance. Apply offline to all original executed paths; no simulator reruns for
the route audit. Added 13 focused tests and representative geometry plots.

**Control:** Same dense GT locations as the evaluator; same state/options and
primitive/path caps as prior analysis. Native SR/SPL/nDTW unchanged. All 414
full-baseline path-length/fastdtw-nDTW/SDTW reconstructions across 138 episodes
match exactly. Every previous cost-only state result is preserved.

**Result:** Pilot8 remains zero. Training64: joint-capped 0.053452 m ->
route-gated 0.042087 m; 19 -> 14 material states, seven routes/five scenes.
Scene-cluster CI [0.005734,0.095675] m. Unseen66: 0.148787 -> 0.126962 m;
59 -> 49 material states over 26 routes/all 11 scenes, scene-cluster CI
[0.074038,0.179887] m. Median state gain zero. Post-hoc concentration: seven
training material states come from episode 6962, which contributes 61.26% of
summed statewise gains (not an episode improvement). Excluding it lowers the
training mean to 0.016881 m. Five training and 12 unseen retained alternatives
still have nonpositive absolute goal progress.

**Interpretation:** The geometric route check removes some misleading goal-only
opportunities, but a limited signal survives. Reference DTW is not semantic
instruction verification. Local relative improvement can remove necessary
backtracking; no alternative continuation has been tested. Training support is
too sparse/correlated to justify a larger model. Original AAA NO-GO; focused
label/graph-decision diagnosis CONDITIONAL GO; no scorer, RL or VLM added.

**Next:** Test one oracle action replacement followed by the unchanged native
navigator, using the frozen training-only seven-route schedule in
`configs/GRAPH_SINGLE_INTERVENTION_TRAIN7_001.json`. Keep all 64 routes and the
other 57 as controls, first run the interceptor disabled, preserve STOP/horizon.
Schedule is **planned, not executed**. Reject the local-label objective if the
apparent benefit disappears on continuation; do not promote it as a learned method.

**Attempts:** Sampler asset preflight initially used the wrong scene root; fixed
before output selection or simulation with the same metadata rule. The first
unseen route reconstruction failed on 14 path-length metrics due to float64 vs
native float32 arithmetic (max 4.126e-6 m); nDTW/SDTW already matched. Corrected
only reconstruction dtype, with no tolerance or gate change. Old attempts,
source snapshots and logs are preserved; accepted analyses use the v2 source.

## 2026-10-07 / GRAPH-SINGLE-INTERVENTION-TRAIN7-001 (started)

**Question:** Do the seven preselected cost-and-route-compatible local alternatives
actually improve final navigation under the original policy's continuation?

**Change:** Added a research-only single-action interceptor through existing
callbacks, preserving original ghost consumption and all subsequent choices.
Uses the schedule frozen in the preceding cycle; no route/state reselection.

**Control:** All 64 original training routes/order, seed 100, released checkpoint,
native sensors/controller/STOP/horizon 15. Run disabled and validate complete
metric/action equivalence before enabling the seven scheduled replacements.

**Result:** Ten interceptor tests pass; disabled 64-route evaluation launched.
No continuation result or learned-method gain is available yet.

**Interpretation:** This tests long-term validity of local labels, not a new
model or benchmark. The population was selected using earlier local outcomes.

**Next:** Require disabled noninterference, then run enabled continuation and
verify the other 57 routes plus every intervention prefix and execution.

## 2026-10-07 / GRAPH-SINGLE-INTERVENTION-TRAIN7-001 (completed)

**Question:** Do the seven frozen cost-and-route-compatible local replacements
improve final navigation under actual native continuation?

**Change:** Execute one privileged graph-index replacement per scheduled route,
then let the original policy continue. Retain all 64 original training routes
and their order. Analyze all seven paired outcomes, including deterioration.

**Control:** Released checkpoint, seed 100, sensors, controller, graph updates,
native STOP, 3 m success distance and 15-decision horizon unchanged. Disabled
interceptor matches 474 complete traces, 1,152 episode metrics and 18 aggregates
exactly. Enabled run changes exactly seven actions; 445 unchanged traces and
1,026 metrics from the other 57 routes match. All seven alternative executions
reproduce the isolated probes with zero endpoint error and matching primitive,
collision, observation and RNG records. Thirteen focused tests pass.

**Result:** Targeted seven: success 4/7 -> 4/7; SPL 46.0484 -> 51.6054%; nDTW
63.7303 -> 67.9689%; SDTW 46.9931 -> 51.6513%. Mean final goal distance worsens
3.4026 -> 4.0180 m. Primitive count totals 674 -> 711 (+37), despite an immediate
replacement saving of 17 primitives. No failed route is rescued. Five nDTW gains,
two declines. Episode 6962 / step 10 gains 3.5544 m locally but its continuation
uses 65 more primitives and ends 4.9852 m farther from goal. Across all 64 routes,
SR stays 85.9375%, SPL gains 0.6078 pp and nDTW gains 0.4636 pp. Independent native
reconstruction of 384 metrics across all 64 routes matches exactly.

**Interpretation:** Some trajectory-efficiency benefit survives continuation,
but one-step cost controls do not guarantee recovery or whole-episode savings.
Seven post-selected training routes/five scenes/one seed are descriptive oracle
evidence, not a trained-method result or held-out improvement. Original NMS AAA:
NO-GO. Narrow graph-decision diagnosis: CONDITIONAL GO. Do not train or integrate
a scorer on these labels yet. H=2 already reverses the two harmful local gains,
but that observation is explicitly post-hoc.

**Next:** Prospectively compare one-step and short native-continuation labels
on new training routes with complete primitive/path accounting, retaining all
negative cases. No new sample, experiment, learned model, RL or VLM was run in
this cycle. See `SINGLE_INTERVENTION_REPORT.md` and
`results/single_intervention_train7/` for accepted results and provenance.

## 2026-10-07 / CONTINUATION-LABEL-TRAIN64-001 (registered)

**Question:** Does a frozen H=2 progress-and-cost check improve the reliability of H=1 labels on new training scenes?

**Change:** Exclude all earlier training diagnostic scenes; predeclare 16 scenes / 64 routes, one eligible intervention per route, H=2 gain >=0.25 m with no extra primitive/path cost.

**Control:** Existing full-option/route gates, checkpoint, seed 100, sensors, native STOP/controller and horizon 15.

**Result:** Protocol/config saved before sampling and new outcomes; no results yet.

**Interpretation:** Prospective label confirmation only, not exhaustive H=2 action search or a learned method.

**Next:** Freeze metadata sample, run baseline/control/probes, then freeze the intervention schedule before continuation.

### CONTINUATION-LABEL-TRAIN64-001 / measurement gates

**Question:** Can the prospective sample and generalized auditing preserve the old measurement contract?

**Change:** Select 64 routes over 16 new-to-diagnostics training scenes, excluding all 19 previously inspected training scenes; support variable intervention counts in the research auditor.

**Control:** No baseline core, config thresholds, candidate policy or simulator change.

**Result:** 1,152 baseline episode metrics and 18 aggregates exactly match untraced control. Nine new sampling/window/label tests plus 13 interceptor/audit tests pass. Reauditing the old seven-route experiment reproduces all earlier gates and results exactly (only the explanatory scope string changes).

**Interpretation:** Measurement/sampling prerequisites pass; these results do not establish H=2 usefulness. Branch collection and independent integrity remain required before continuation.

**Next:** Automatically freeze the event schedule after the complete-option route audit, then run paired actual continuations.

### CONTINUATION-LABEL-TRAIN64-001 / full-option stage completed

**Question:** Does the previously observed route-compatible local signal survive coverage of new training scenes?

**Change:** Apply the unchanged calibrated full-option and route gates to the frozen 64-route / 16-scene sample.

**Control:** All thresholds, checkpoint, native policy/controller/STOP and costs remain fixed.

**Result:** 469 sentinels, 1,365 reversed-option checks and 6,699 independent branch-integrity checks pass; 192 native metric reconstructions exact. Across 405 non-STOP states, joint-cost mean gain 0.025680 m falls to 0.008900 m after the route gate. Six material states occur in five routes / three scenes. All five routes enter the frozen one-intervention schedule; no additional route was sampled.

**Interpretation:** Support is sparse and below the registered scene-coverage gate for learning. Local maxima still cannot establish eventual efficiency.

**Next:** Automatically complete the already scheduled disabled/enabled continuations and evaluate the fixed H=2 rule without tuning it.

## 2026-10-07 / CONTINUATION-LABEL-TRAIN64-001 (completed)

**Question:** Does the frozen H=2 progress-and-cost rule improve the reliability of H=1 labels on new training scenes?

**Change:** Five frozen H=1 interventions, followed by actual native continuations; apply the preregistered H=2 rule, then automatically execute its two-event mixed schedule after a second disabled control.

**Control:** Original checkpoint, seed, sensors, controller, native STOP and horizon 15. Five-event run preserves 59 control routes / 1,062 metrics; two-event run preserves 62 / 1,116. All alternatives and prefixes pass. Both enabled runs independently reconstruct 384 final metrics exactly; 469 mixed-run full traces equal offline composition. Twenty-two focused tests pass.

**Result:** On the same five eligible routes, baseline / H1 / H2 success remains 5/5; SPL 66.0905 / 73.9205 / 69.6394%; nDTW 68.0487 / 77.0138 / 70.6196%; total primitives 692 / 478 / 692. H2 rejects adverse 10121 (+32 primitives, nDTW -2.549 pp) but also useful 1773 (-244, +31.293 pp) and 1374 (-2, +3.227 pp). One of two H2 acceptances is terminal by H2.

**Interpretation:** H2 has some descriptive safety benefit but rejects delayed efficiency; positives cover only three scenes and all baseline successes. No recovery, learned, held-out, significance or novelty claim. NO-GO for direct short-window label distillation; narrow full-return diagnosis remains CONDITIONAL GO. Original NMS AAA remains NO-GO.

**Next:** Automatically decompose whole-episode costs and check whether unequal primitive budgets explain these window errors; do not retune H on these cases.

## 2026-10-07 / CONTINUATION-COST-DECOMPOSITION-001 (completed)

**Question:** Where do immediate versus final execution-cost differences arise?

**Change:** Post-hoc accounting of all seven earlier and five prospective events, retaining batch separation and same-ghost-ID target drift.

**Control:** Read only accepted actual traces; no new episode selection, threshold, action, controller or STOP change. Primitive and collision deltas tie exactly to native metrics in every case.

**Result:** Old seven: -17 immediate, +26 later navigation, +28 STOP = +37 primitives. New five: -39 immediate, -105 later navigation, -70 STOP = -214. Route 1773 contributes -244 alone; other four together +30. Its baseline forced STOP uses 71 primitives through four graph nodes versus one primitive after altered continuation. None of the new five reselects the displaced original ghost ID.

**Interpretation:** Benefits and harms can emerge much later and are concentrated. Neither local savings nor one repeated-frontier symptom explains all final outcomes. These overlapping symptoms do not establish causal failure classes.

**Next:** Calibrate navmesh goal distances, then inspect the same actual trajectories under equal baseline-derived primitive budgets.

## 2026-10-07 / EQUAL-PRIMITIVE-CONTINUATION-001 (completed)

**Question:** Does duration normalization alone resolve misleading fixed-H2 labels?

**Change:** Post-hoc comparison at each baseline H2 window's primitive count, using retained actual intermediate poses; early STOP is absorbing offline.

**Control:** First reproduce all 456 pre/post native geodesic distances exactly; same goals, navmeshes and original 0.25 m / path thresholds. No additional simulator actions or executable truncated options.

**Result:** Three decisions flip in the earlier batch and two in the new batch. New accepted IDs become 1374/7581; 1773 is still rejected. Old harmful 9045 is now accepted. Budget-normalized windows therefore retain false acceptance/rejection in these cases.

**Interpretation:** This exploratory check does not validate a new selector or horizon. Equal primitive count also does not measure wall-clock cost. No model trained and no core navigation source changed.

**Next:** Design outcome-blind state/candidate sampling with complete native-continuation returns; local gain cannot determine eligibility. See FULL_RETURN_LABEL_NEXT_GATE.md; that prospective design is not executed yet.

## 2026-10-07 / FULL-RETURN-LABEL-TRAIN16-001 (completed)

**Question:** Do outcome-blind alternative graph actions produce reliable full-episode labels for a deployable adaptive selector?

**Change:** Freeze one native non-STOP state per route on 16 routes from eight fresh-to-diagnostics training scenes. Run two distinct valid alternatives to native STOP: highest nonselected graph logit and a seeded graph-ID choice.

**Control:** Checkpoint, seed 100, simulator, sensors, controller, STOP, 15-decision limit, evaluator and environment provenance. Disabled controls, prefix/action/RNG checks, 1,429 forward branches, 305 reverse checks and one independent 96-metric reconstruction per enabled schedule (192 total) all pass.

**Result:** Across 32 alternatives, highest-logit has 1 dominating, 10 mixed and 5 dominated; seeded graph-ID has 0 dominating, 8 mixed and 8 dominated. The single dominating event covers one scene and rescues route 2508. Highest-logit mean SPL/nDTW changes are -15.23/-12.08 pp with +15.5 primitives; seeded changes are -28.77/-24.27 pp with +63.0 primitives. The rescued route is locally worse at H=1/H=2, showing delayed full-return effects.

**Interpretation:** The current pilot provides one useful route but adverse aggregate behavior and only one dominating scene. It cannot support selector fitting, scalar reward design, RL or VLM assistance. Fitting from this full-return graph-choice pilot is NO-GO; original NMS AAA and short-window labels remain NO-GO.

**Next:** Census terminal/recovery symptoms on existing calibrated baseline traces, keeping cohorts separate. Do not resample this pilot or tune a selector to its single rescue.


## 2026-10-07 / FULL-RETURN-LABEL-TRAIN16-001-DESCRIPTIVE (completed)

**Question:** Are mean changes driven by uncertain scene effects, and what terminal symptom occurs in the sole dominating event?

**Change:** Offline 5,000 paired scene-bootstrap draws (seed 20261007) and endpoint-neighborhood / terminal-cost census; no extra simulator rollout.

**Control:** Accepted actual trajectories and native metrics only; eight equal-sized scene clusters; preserve every positive, mixed and negative alternative. No model or threshold adjustment.

**Result:** Top-logit nDTW delta 95% interval [-24.57,-1.35] pp and primitive interval [+2.25,+28.50]; seeded [-35.93,-13.24] pp and [+37.81,+88.56]. Baseline 2508 reaches 0.1669 m then departs and stops at 4.1721 m; top alternative eventually stops at 0.1669 m. Neither alternative schedule has an endpoint-visit-then-failure. Two baseline failures and every newly lost success remain in the JSON.

**Interpretation:** This describes delayed stopping/continuation interactions, not an identified STOP cause or deployable goal oracle. Eight clusters / one seed give limited uncertainty estimates. Attempt 001 failed on a missing raw-trace success key; corrected attempt 002 uses audited native success, preserving logs and failed source.

**Next:** Check how broadly terminal/recovery symptoms occur across existing calibrated cohorts before defining a new causal intervention.

## 2026-10-07 / TERMINAL-RECOVERY-CENSUS-001 (completed)

**Question:** How common are endpoint-observable arrival-then-departure, never-arrived failure and forced-return symptoms in already calibrated baseline traces?

**Change:** Offline census over the accepted 64-route training capture and 16-route full-return capture at the unchanged 3 m success threshold.

**Control:** Read-only baseline traces and native episode metrics; no rollout, action, STOP, threshold or model change.

**Result:** Training64: 55/64 successes, 9 failures; all 9 failures never had an endpoint inside 3 m, 3 used forced STOP and all 3 failed. Full-return16: 14/16 successes, 2 failures; one arrived-then-left (2508), one never arrived, one forced STOP and it succeeded.

**Interpretation:** Endpoint symptoms vary by cohort. The 2508 delayed rescue is real but rare in this sample; forced STOP is not uniformly associated with failure. These are diagnostic labels with between-decision blind spots, not causal failure categories.

**Next:** Keep STOP, recovery and graph selection jointly scoped in any future study; do not train a component from this census alone.

## 2026-10-07 / TERMINAL-RECOVERY-CENSUS-002 (completed)

**Question:** Does the endpoint symptom generalize to calibration validation cohorts?

**Change:** Read-only census of accepted val_seen64 and val_unseen66 option-calibration captures, with their native `oracle_success` metrics checked separately.

**Control:** No rollout, intervention, threshold or model change; capture and metric tags remain paired (`001` for val_seen, `002` for val_unseen).

**Result:** val_seen64 has 13 failures, 4 with an endpoint within 3 m, 9 without; val_unseen66 has 22 failures, 1 with an endpoint within 3 m, 21 without. Native oracle-success flags among failures are 4 and 2 respectively, showing endpoint sampling misses intermediate proximity.

**Interpretation:** Arrival-then-departure exists but is not the dominant endpoint symptom in these cohorts. It cannot justify a STOP-only model or causal claim.

**Next:** Preserve this as a measurement limitation and keep graph selection, continuation and STOP coupled in future diagnostics.

## 2026-10-07 / FULL-RETURN-FEATURE-AUDIT-001 (completed)

**Question:** Can deployed pre-action graph fields explain the observed full-return tradeoffs without adding a model?

**Change:** Offline audit of native logit gap, logit entropy, commanded graph polyline, back-path length and current-proposal flag against the frozen 32 alternative outcomes.

**Control:** Same frozen states, candidates, source hashes and six-metric dominance rule. No simulator, fitting, outcome-based selection, threshold search or validation access.

**Result:** Top-minus-seeded paired success/SPL/nDTW scene intervals cross zero. Pooled commanded-polyline deltas correlate with final path/actions at 0.877/0.851 and with SPL/nDTW at -0.615/-0.775; native logit gap correlates with path/actions at 0.680/0.718 but is unstable across candidate subsets.

**Interpretation:** Geometry mainly exposes expected cost and candidate-source confounding. It does not provide a safe quality predictor or selector label. NO-GO for feature-based fitting on this sample.

**Next:** If continuing graph-choice work, freeze a new held-out feature contract and evaluate prospectively; do not tune the current 32 outcomes.

## 2026-10-07 / FULL-RETURN-FEATURE-COST-001 (completed)

**Question:** Is the observed geometry/cost association only immediate execution cost, or does it continue into later navigation and STOP?

**Change:** Exact post-hoc primitive/collision decomposition for both accepted full-return schedules.

**Control:** Frozen traces and native metrics; no rollout, model, threshold or candidate change.

**Result:** Top-logit added 248 primitives: 109 immediate, 139 later navigation, 0 later STOP. Seeded added 1,008: 330 immediate, 643 later navigation, 35 later STOP. Commanded-polyline difference versus later-navigation primitive delta is 0.629 (top) and 0.700 (seeded).

**Interpretation:** A one-step cost label misses continuation cost, but the correlation is confounded by candidate geometry and later policy behavior. No causal graph-ranking or STOP claim.

**Next:** Do not add a cost loss yet; a prospective held-out feature contract must first show quality signal under full-return evaluation.

## 2026-10-07 / FULL-RETURN-BRANCH-ALIGNMENT-001 (completed)

**Question:** Do the full-return alternatives also look locally useful at their selected state?

**Change:** Join every scheduled alternative to its exact forward branch probe and rank it by one-step goal progress and progress per primitive.

**Control:** Existing 16 states, 32 candidates, branch-integrity gates and accepted full-return outcomes; no rollout or fitting.

**Result:** All 32 alternatives have lower one-step progress than their native selected action. Top-logit mean delta is -1.724 m with mean progress rank 3.19; seeded is -3.998 m with mean rank 5.69. One top-logit alternative is locally worse yet finally dominates: route 2508.

**Interpretation:** The pilot shortlist has no locally positive alternative examples. This limits learning support but does not prove the whole action space lacks useful candidates. Short-window labels remain invalid for this sample.

**Next:** Audit native STOP branch feasibility and interior primitive arrivals before deciding whether STOP deserves a separate diagnostic.

## 2026-10-07 / NATIVE-STOP-FEASIBILITY-001 (completed)

**Question:** Can an unchanged native STOP branch have succeeded earlier on a failed route?

**Change:** Read-only audit of all accepted forward index-0 branches in five calibrated cohorts; compare pre-decision proximity, native STOP return distance, final metrics and historical back-path. Native success uses <=3 m.

**Control:** Existing probe/integrity/noninterference gates; exact action/RNG/pose and selected STOP replay checks. No simulator rollout or rule change.

**Result:** Train64: 0/9 failed routes have a successful native STOP branch. Continuation train64: 2/8. Full-return16: 1/2 (2508). Unseen66: 1/22 (73). 84 endpoint geodesic reconstructions have max error 0. Primitive localization finds interior arrivals on 10587, 393, 2508, 1593 and 73; 1593 has no successful STOP endpoint.

**Interpretation:** STOP timing can rescue a few routes but is not a general explanation. Interior proximity is privileged and does not define a deployable interrupt. NO-GO for STOP-only modeling or reviving AAA from this evidence.

**Next:** Keep STOP, low-level execution and graph selection coupled. Any future STOP work needs a deployable observability audit and independent held-out full-return evaluation.

## 2026-10-07 / PRIMITIVE-ARRIVAL-LOCALIZATION-001 (completed)

**Question:** Does oracle arrival without a successful decision-boundary STOP occur during ordinary movement or a terminal return?

**Change:** Inspect all five baseline failures with native oracle_success across the fixed cohorts, using recorded primitive poses and the dataset/navmesh from the matching split and scene.

**Control:** No new actions, rendering, policy calls or training. First reconstruct 84 pre/post goal distances; maximum discrepancy 0 m. Native success threshold remains <=3 m.

**Result:** Unseen1593 crosses into 3 m at primitive48 (option4, zero-based), minimum2.915239 m; only primitives48-50 are inside, all in ordinary act=4 motion. Its nearest decision boundary is3.107427 m and final error3.206254 m. Native STOP branches all fail on this route. The other four oracle-arrival failures have earlier successful STOP opportunities.

**Interpretation:** One real interior-only arrival symptom is measured, not a learned interruption rule or a success guarantee. Its margin is only0.084761 m inside the native threshold. Adaptive trajectory length already has literature precedent; no novelty or standalone contribution is supported.

**Next:** Keep the AAA/selector learning NO-GO. Before any temporal abstraction prototype, review deployable intermediate observations and published adaptive truncation mechanisms; do not fit to this route or resample for more positives.

## 2026-10-07 / INTERRUPTIBLE-EXECUTION-AUDIT-001 (completed)

**Question:** Can the one interior-arrival route justify mid-option re-observation and interruption in the existing ETPNav executor?

**Change:** Static audit of environment control, observation return, graph ghost consumption and literature overlap; no baseline core mutation or simulator run. Research audit files and a static checker were added.

**Control:** Existing action contract, `sim.step_without_obs` ordinary execution, high-level loop and graph update ordering. Thirteen static source checks pass after correcting a preflight that incorrectly required inherited methods in the tracing subclass.

**Result:** Non-video primitive control returns no observation to the high-level policy; the front-node render is overwritten by the final ghost-segment render before return. Ghost deletion and `prev_vp` update occur before `envs.step`. Mid-option interruption would require explicit commit/cancel semantics for pending graph/control intent while preserving executed physics, costs, collisions and RNG. Beyond Waypoints truncates candidates before execution; it does not establish the same online mid-option mechanism. Revised static metadata are in `analysis_002`; prior output is preserved. Cycle provenance is under `research/results/interruptible_execution_audit/archive_001/`.

**Interpretation:** The interior-arrival symptom identifies an implementation constraint, not a validated research contribution. A mid-option STOP head would be an incomplete and potentially unfair redesign under the current graph transaction. The checker is static evidence only; it does not prove dynamic semantics.

**Next:** Keep interruptible execution NO-GO. If revisited, first specify the transaction and sensing budget, then perform a literature/action-identity audit before any prototype or training.


## 2026-10-07 / CRITICAL-FULL-RETURN-TRAIN64-001 (completed)

**Question:** At failure-critical states, can any existing native graph action rescue complete-episode return, and can event-triggered re-observation/replanning rescue execution symptoms?

**Change:** User-directed pivot: stop coarse/default/fine learning. Enumerate all 509 admissible native actions at 46 registered critical states in all nine train64 failures (eight scenes). Replay from reset, substitute one action, then run native continuation. Four collision cuts use sensing-only and two graph-consumption semantics.

**Control:** Same checkpoint, sensors, seed100, sliding=true, effective tryout=false, native STOP and 15-decision horizon. Nine source-identical dataset records; every prefix, selected-action control and final metric audited. No fitted model, RL or VLM.

**Result:** SR rescue by non-STOP replacement on 6/9 routes; 4/9 in three scenes also avoid nDTW loss. Termination replacement rescues 2/9, overlapping the six. Four interruption events rescue 0/4 under either ghost semantic. Sensing-only exactly matches baseline. 530 actual rollouts; 4,240 reconstructed metrics, 4,709 exact option/prefix checks and 52,012 exact primitive records; five boundary tests pass.

**Interpretation:** CONDITIONAL GO for registered long-horizon graph-choice investigation; not yet a route-quality-aware majority or learned gain. NO-GO for interrupt-policy training from four fixed cuts. Three routes remain unresolved, not proven proposal failures. The retired legacy schedule adapter failed cohort validation before rollout; its attempts are marked SUPERSEDED. Smoke001 had a metric-key serialization mismatch, smoke002 was manually interrupted, smoke003 passed.

**Next:** Apply the identical state/choice/interrupt rules to the previously inspected unseen66 diagnostic failures, keeping their results separate and forbidding use as fitting labels. Protocol: CRITICAL_CAUSAL_UNSEEN_PROTOCOL.md.


## 2026-10-07 / CRITICAL-FULL-RETURN-UNSEEN66-001 (completed)

**Question:** Does the failure-critical graph-choice/interruption evidence survive across the previously inspected 11-scene unseen66 diagnostic cohort?

**Change:** Applied the frozen train protocol to all 22 baseline failures: 71 critical states, 985 admissible actions, one replacement followed by native continuation, and four first collision cuts with sensing-only/consume/retain arms.

**Control:** Same checkpoint, seed100, sensors, native STOP, 15-decision horizon, collision accounting and source-identical subset records. No labels entered training and no benchmark aggregate was changed.

**Result:** 7/22 SR rescues by native non-STOP replacement; 6/22 also avoid nDTW degradation. Termination replacement rescues 6/22 (overlapping). Four interrupt cuts rescue 0/4; sensing-only exact. 8,152 metric reconstructions, 9,688 option/prefix checks and 127,881 primitive records pass independently.

**Interpretation:** Cross-scene confirmation makes graph-choice opportunity the leading observed mechanism, but 6/22 is not a majority and 14 routes remain unresolved. Interrupt remains NO-GO for learning. The subset was previously inspected and is not an untouched held-out benchmark.

**Next:** Keep coarse/default/fine and interrupt-policy training stopped. Register a training-only, route-quality-aware long-horizon graph value/preference feasibility study only after fixing outcome-blind features, continuation-cost accounting and a genuinely held-out scene protocol. Do not use unseen results as labels.

## 2026-10-07 / CRITICAL-GRAPH-VALUE-FRESH-001 (completed)

**Question:** Does a newly registered, six-scene training failure cohort contain
enough route-quality-aware native graph-action preference signal to justify one
held-out confirmation?

**Change:** Frozen 10 failures from 16 new training scenes, enumerated every
admissible native graph action at 47 critical states, executed 996 full native
continuations, and tested four preselected interrupt cuts. No model controlled
navigation.

**Control:** Same checkpoint, seed 100, sensors, graph masks, controller,
native STOP and 15-decision horizon. Controls, action identity, primitive
prefixes, RNG, collisions and final metrics were independently audited.

**Result:** 996/996 action cases passed. Nine of ten routes have a graph-choice
rescue; seven have SR rescue with nDTW nondegradation across five scenes.
Termination rescues occur on six routes and overlap. Interrupt rescues are 2/4,
but only one is route-quality clean. The audit passes 10,868 exact option/prefix
checks and 144,979 primitive checks. A corrected outcome-blind pairwise model
has 5,668 strict pairs and leave-one-scene-out accuracy 0.9386 versus 0.8520
for native logit.

**Interpretation:** This is **CONDITIONAL GO** for a separately registered,
scene-grouped full-return confirmation. Pair accuracy is feasibility evidence;
predicted actions were not executed. Counts overlap and cannot be called a
causal majority or benchmark gain. Interrupt learning remains NO-GO.

**Correction:** The first feasibility artifact indexed STOP probability against
the unfiltered option list and omitted the current-to-first back-path segment.
`feasibility_001.json` is retained as superseded audit history;
`feasibility_002.json` uses the corrected schema. New unit tests cover both
cases. Rollout rescue counts are unchanged.

**Next:** Register and execute one independent six-scene confirmation with the
corrected feature contract. Compare frozen ranker, native logit and random
choice without policy retraining. Require route-quality rescue in at least two
scenes while preserving STOP and primitive budget; otherwise stop graph-value
fitting and investigate proposal coverage.

## 2026-10-07 / GRAPH-VALUE-CONFIRMATION-001-SCENE-POOL-AUDIT (completed)

**Question:** Is a genuinely independent six-scene, 96-route confirmation
cohort available in the local R2R/RxR data after the registered exclusions?

**Change:** Audited the metadata-only scene pool before writing any confirmation
dataset. Counted unique scene IDs and trajectories from the exact source and
hashed every exclusion manifest.

**Control:** No episode outcomes, reference routes, simulator states or labels
were read. No dataset sample was written and no model was trained.

**Result:** The R2R train source has 61 scenes. Existing diagnostic exclusions
cover 59, leaving two scenes with four routes total. This is below the
pre-registered six-scene/96-route gate. RxR train uses the same 59 MP3D scene
IDs, so changing language annotations would not create scene-independent
evidence.

**Interpretation:** **NO-GO for the clean independent confirmation with the
current local scene pool.** Reusing the 59 covered scenes or validation labels
would invalidate the intended gate. This is a data-availability limitation,
not evidence for or against graph ranking.

**Next:** Expand the scene/data pool, or explicitly register a weaker
episode-level cross-scene-overlap study with its limitations. Keep the ranker
out of navigation and keep RL/PPO/VLM and interrupt-policy work paused until a
valid confirmation cohort exists.

## 2026-10-07 / GRAPH-VALUE-CROSS-COHORT-TRANSFER-001 (completed)

**Question:** Does the corrected fresh-cohort preference model transfer to the
earlier 16-state, eight-scene full-return cohort?

**Change:** Fit only on the fresh 47-state cohort using schema v2, then score the
separate two-alternative confirmation states offline. No target outcomes were
used as features and no predicted action was executed.

**Control:** Native graph logit, exact action identity, same strict dominance
definition and the unchanged confirmation records.

**Result:** Learned pair accuracy is 0.8421, exactly equal to native logit
0.8421 over 19 strict pairs (`transfer_002.json`). The earlier train-new-to-
train16 transfer remains 0.8947 versus 0.8421, but it uses a different
training cohort and is only feasibility evidence.

**Interpretation:** The fresh in-cohort 0.9386 pair accuracy does not establish
cross-cohort generalization. This strengthens the need for a truly independent
confirmation and rules out treating the current ranker as deployable.

**Next:** Do not fit more features or tune thresholds on existing outcomes.
Obtain new scene data or explicitly downgrade the next study to an
episode-level overlap analysis with wider uncertainty.


## 2026-10-07 / GRAPH-VALUE-ROUTE-EXECUTION-002

Question:
Does a frozen full-return graph preference yield real navigation gains on routes
excluded from prior research diagnostics, rather than just offline pair accuracy?

Change:
Correct route exclusions (invalid pilot 12/96 overlaps preserved), capture 96 new
routes in eight scenes, census 561 actions at 57 failure-critical states, then
execute one first-disagreement action from an old-cohort ridge-10 scorer. Three
random controls use identical eligible states and native-action abstention.

Control:
Released checkpoint, native STOP, 15 decisions, masks, sensors, sliding/controller,
seed 100 and all 96 routes are identical. Every disabled hook exactly matches
native; every enabled branch/prefix and all final metric reconstructions pass.
Training uses only prior 47-state/996-action/5,668-pair fresh census, not target labels.

Result:
Census: 4/7 graph SR opportunities, 3/7 SR+nDTW, 0/4 interrupt rescues.
Learned: 3 rescued/0 lost, SR +3.125 pp, SPL +2.302 pp, nDTW +0.956 pp,
primitive count −1.677/route. Random seeds 20261021/22/23: SR 0/−1.042/0 pp,
SPL −5.223/−6.111/−5.792 pp, nDTW −4.115/−4.332/−3.684 pp.
Scene bootstrap learned-native SR CI +1.042 to +6.250 pp; SPL/nDTW/cost CIs cross zero.

Interpretation:
CONDITIONAL GO for prospective replication. Point gates pass; stable multi-seed
or benchmark gains are not established. Target census was inspected before this
schedule was written, so this is exploratory. Primitive savings and gains may
be explained by execution-distance preference. State features cancel in this
linear scorer. Route 3161 rescued at step 3 despite no late census rescue.
Incomplete critical smoke/full stdout/stderr archival is explicitly documented;
foundation/intervention logs and every case result/trace are retained.

Next:
Prospectively freeze a fresh route-disjoint sample and compare the unchanged
full scorer with a graph-logit/rank plus distance-only preference fit on the old
training cohort. No target tuning, RL/VLM, density selector or interrupt model.


## 2026-10-07 / GRAPH-VALUE-PROSPECTIVE-REPLICATION-001 — registration

Question:
Does the frozen first-disagreement scorer replicate on new research-held-out
routes, and does its full feature set add value beyond graph scores and execution
distance at the same decision time?

Change:
New metadata-only sample, seed 20261024: 96 routes in 8 scenes, excluding 1,584
registered route keys including both prior 96-route cohorts. FULL weights are
byte-identical to the old model. COST fits four features on the same 5,668 old
strict-dominance pairs. Six arms: FULL, COST-own, COST-matched at FULL states,
and three matched random seeds 20261031/32/33. Native is the paired baseline.

Control:
R2R released checkpoint, seed 100, sensors, sliding/controller, masks, native
STOP and 15-decision horizon. All 96 routes retained. Models/protocol/source
hashes frozen before target capture and committed as 6fb0482. No target labels
used in fitting; no resampling after native outcomes. Scene overlap allowed.

Result:
Sampling audit passed with zero declared-route overlap; 3 new focused tests pass.
Navigation outcomes pending; do not treat registration as a positive result.

Interpretation:
This is a prospective follow-up to the exploratory execution gain, still on
baseline training scenes/routes. COST uses the same full-return labels, so the
contrast tests feature sufficiency, not whether long-horizon labels are necessary.

Next:
Complete all predeclared fidelity gates and six actual-navigation arms; apply
the frozen point and matched-feature gates without tuning or changing the sample.


## 2026-10-08 / GRAPH-VALUE-PROSPECTIVE-REPLICATION-001 — completed

Question:
Does the frozen first-disagreement result replicate, and do full features add
value over graph scores plus execution distance at matched timing?

Change:
Executed all six prospectively frozen arms on 96 new declared-research-held-out
routes in eight overlapping training scenes. FULL remains byte-identical;
COST uses four features fitted on old labels. No target fitting or resampling.

Control:
Same checkpoint, simulator seed 100, sensors, masks, controller/sliding, STOP
and 15-decision horizon. All 96 routes retained. Six disabled/enabled audits
pass; 3,456 independent metric reconstructions and 838 cross-arm identical-action
checks pass. Original random31 worker-startup BrokenPipeError produced no
navigation records; preserved and retried under continuation_001 amendment,
with identical registered scientific settings and unchanged execution HEAD.

Result:
FULL rescues 4, loses 1; SR/SPL/nDTW +3.125/+1.771/+0.938 pp;
primitives +1.260/route. All four primary scene CIs include zero.
COST-own: same SR gain, +3.427 primitives/route. COST-matched: SR +2.083 pp,
+1.021 primitives/route. Random SR changes 0/−1.042/−1.042 pp; all worsen
SPL/nDTW/cost. FULL-vs-COST-matched added-feature gate fails.
Post hoc: changed options −133 primitives, continuation +254, total +121;
14 routes have cheaper replacements but costlier complete episodes.

Interpretation:
NO-GO for scaling this frozen first-disagreement recipe: the prospective mean
primitive gate fails. Positive SR point changes and superiority to random are
preserved, but do not override the gate or establish benchmark generalization.
The data do not show that extra full-model features are necessary, that interrupts
would solve the problem, or that proposal coverage is the dominant failure.

Next:
Before further learning, separately register a development-only diagnostic of
pre-action evidence for harmful early substitutions using full continuation
cost, native-success routes and abstention. Do not tune on this replication,
expand this recipe, pool it post hoc into a positive claim, or add RL/VLM.


## 2026-10-08 / PREFERENCE-CONTINUATION-MECHANISM-001

Question:
What follows cheaper local replacements whose complete navigation costs more:
execution symptoms, later graph navigation, displaced-target reselection, or
STOP return execution? Can existing structural signals distinguish the harm?

Change:
Retrospective specification fixed before this derived census, after old outcomes
were known. Analyze all 27 exploratory and 29 replication FULL events; retain
74 COST-own and 23 COST-matched events as overlapping secondary controls.
Use previous critical-census execution thresholds and 0.25 m stable-target drift.
An explicitly post hoc attribution additionally decomposes the frozen pair score.

Control:
No model fit, weight/threshold tuning, candidate/action change, or new rollout.
Existing seed-100 checkpoint/controller/STOP/15-decision traces reused. Protected
replication outcomes remain excluded from fitting. All 153 actual action/prefix
and cost/collision/path ties pass; four focused tests pass. Config/source/input
hashes and durable analysis/test/attribution/plot stdout/stderr saved. Analysis
execution Git HEAD is 72f78fb; new tool contents are pinned by source hashes.

Result:
FULL cost reversals 10/27 and 14/29; replacement symptoms only 2/10 and 2/14;
stable next reselections 4/10 and 7/14. All 56 alternatives are current proposals
with no back-path, 53 shorten final target segments. Geometry gives the largest
positive frozen-score contribution in 49/56 choices and 23/24 reversals.
Earlier total primitive delta −161 includes −155 from route 10199. Cost split:
earlier immediate −106, later navigation +36, later STOP −91; replication
immediate −133, later navigation +200, later STOP +54. No aggregate pooled claim.

Interpretation:
NO-GO remains for the failed frozen recipe. Cheap local actions do not establish
cheap complete returns; repeated-target and execution symptoms are neither a
complete causal partition nor demonstrated remedies. Score attribution is
algebraic, not a causal ablation. Saturated structural flags cannot distinguish
these treated choices, but other untested features are not ruled out.

Next:
The specified development-only training-support audit compares strict-pair
accuracy with native-relative full-return choices on the old 47-state archive,
including mixed/tied and STOP-involving pairs; no fit or new labels. It has not
run. In-sample results cannot overturn the failed prospective navigation gate.


## 2026-10-08 / PREFERENCE-TRAINING-SUPPORT-001 — completed

Question:
Does strict all-pair accuracy support the actual native-versus-chosen decisions,
or is the apparent success concentrated in comparisons irrelevant to the frozen
override rule?

Change:
Execute the previously specified postmortem on the old 47-state/996-action
fitting archive. Classify all pairs, preserve STOP/masks/abstention, and extract
selected full returns. Add an explicitly post hoc native-preferred versus
alternative-preferred direction split of the audited strict move pairs.

Control:
No fitting, new labels, simulator episodes or replication-outcome reads.
Unchanged frozen FULL weights/normalization and original action tie-break.
Executable/input hashes registered at analysis HEAD 6100821. Original simulation
seed 100, checkpoint, sensors/controller, STOP and horizon are preserved in the
source provenance. All 47 native returns match baseline; six tests pass.
Independent vector-label/masked-sort verification passes 12,159 pairs, 47
choices and 564 return components, sharing only frozen feature extraction.

Result:
12,159 unordered pairs: 5,668 strict, 6,491 mixed, zero ties. All-pair accuracy
94.34% vs logit 85.20%; native-move 403/418 vs logit 404/418. All-pair net +518
correct comparisons = +404 non-native move pairs +115 STOP pairs −1 native-move
pair. Only 14 native-move pairs prefer the alternative; frozen model identifies
0/14, across 12 states / six routes / four scenes. Choices: 10 protected STOP,
35 native abstentions, two overrides. One is strictly worse (+59 primitives),
the other mixed (SR rescued, nDTW −9.175 pp, primitives +13). No successful
native continuation appears in this failure-selected fitting population.

Interpretation:
NO-GO remains for this recipe. Retire all-pair accuracy as a sufficient development
gate. Its improvement is not evidence of detecting native-relative benefits;
53.38% mixed comparisons are not learned strict pairs, including one actual
override. These in-sample correlated-state counts are not new navigation metrics
or held-out prediction performance, and cannot establish the remedy.

Next:
Close this recipe as an unsupported method candidate. Before any new training,
require an independently justified mechanism and a native-relative full-return
evaluation with abstention and successful-route controls. Do not launch new
labels, losses, RL/VLM, interrupt or proposal training on this audit alone.

## 2026-10-08 / INTERRUPT-COVERAGE-001

Question: Did the first-event/four-route interrupt pilot cover the available
collision timing opportunities, and do successful native routes face the same exposure?

Change: Deterministic enumeration of all eligible interior ghost cuts on old
train64 and fresh64 captures. No rollout, model or returned-action labels added.

Control: Original threshold and first-event eligibility rules, seed100, training
cohorts, native checkpoint/physics and source metrics. Prospective/validation
outcomes excluded; successful nonempty-back-path phases marked unknown.

Result: Failed routes: 156 cuts / 56 options / 11 eligible routes, of which only
8 cuts/options/routes historically tested. Successful routes: at least 128 cuts /
43 options / 27 routes; three move phases unknown. 217 native-control checks,
188 empty-path phase checks, 7 tests and independent 878-option/284-cut/two-plan
verification pass. Attempt001 STOP-phase assertion failure retained; attempt002
corrects only that assertion, not eligibility or population.

Interpretation: Timing opportunity was undersampled; symptom alone does not justify
interrupting successful routes. No causal benefit inferred and no model gate passed.

Next: Bounded fresh-cohort timing oracle over all 175 known cuts, with matched
sensing, both pending-ghost semantics and 19 native route controls.

## 2026-10-08 / INTERRUPT-TIMING-ORACLE-001

Question: Does expanding collision cut timing reveal route-quality rescue, and
what harm occurs on successful native routes?

Change: All 175 known eligible fresh64 cuts, each followed by original navigator
under sense-only / interrupt-consume / interrupt-retain, plus 19 native controls.
25-rollout smoke precedes 544 complete main rollouts. No learning or new predictor.

Control: Plan/protocol committed in30bc545 before execution; release checkpoint,
seed100, sensors, sliding, effective tryout, physical prefixes, STOP and 15 decisions
unchanged. Fresh64 is old training data; unknown-phase success options excluded.

Result: Consume timing oracle quality-rescues4/7 failed routes in3 scenes, with
3/7 also primitive-cost-capped. Retain rescues3/7 in2 scenes,2/7 cost-capped.
First-event consume preserves all12 native successes but lowers nDTW on8/12;
retain loses1/12 success. Both have any-cut success harm on2/12. Full verifier
passes4590 native/prefix records,350 actual cuts/panoramas,4352 return components,
7118 path continuities. Five legacy and four return-gate tests pass.

Interpretation: Registered gate passes for independent confirmation only. There
is a bounded mid-option replan opportunity, but no safe trigger, execution-majority
claim or benchmark improvement. All-cut oracle is privileged and correlated.

Next: Independently registered route sample, comparing event cuts with outcome-blind
within-option timing controls, native abstention and successful-route controls.
Do not fit a trigger or select thresholds from this sweep.
