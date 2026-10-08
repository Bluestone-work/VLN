# Experiment Table

All runs use Habitat seed 100 unless stated otherwise. Full runs use the
released R2R IL checkpoint, two RTX 4090 GPUs, eight environments per process,
`IL.back_algo=control`, and success distance 3 m.

| ID | Split/episodes | Abstraction | SR | SPL | nDTW | Status | Artifacts |
| --- | --- | --- | ---: | ---: | ---: | --- | --- |
| BASELINE-001 | val_seen / 1 | default | 100.00% | 64.694% | 83.369% | passed smoke | `research/results/baseline/aaa_baseline_smoke4.stdout` |
| BASELINE-FULL | val_unseen / 1839 | default | 57.096% | 49.101% | 62.305% | exact reference reproduction | `research/results/baseline/r2r_current_full_result.json` |
| FIXED-002-COARSE | val_seen / 8 | coarse | 50.00% | 32.310% | 41.355% | diagnostic | `research/results/action_abstraction/coarse_8.jsonl` |
| FIXED-002-DEFAULT | val_seen / 8 | default | 87.50% | 68.921% | 70.863% | diagnostic | `research/results/action_abstraction/default_8.jsonl` |
| FIXED-002-FINE | val_seen / 8 | fine | 62.50% | 39.987% | 49.726% | diagnostic | `research/results/action_abstraction/fine_8.jsonl` |
| FIXED-003-COARSE | val_unseen / 1839 | coarse | 54.214% | 45.474% | 59.650% | full fixed comparison | `research/results/action_abstraction/coarse_full_result.json` |
| FIXED-003-DEFAULT | val_unseen / 1839 | default | 57.096% | 49.101% | 62.305% | full fixed comparison | `research/results/action_abstraction/default_full_result.json` |
| FIXED-003-FINE | val_unseen / 1839 | fine | 57.531% | 47.260% | 60.045% | full fixed comparison | `research/results/action_abstraction/fine_full_result.json` |
| ORACLE-001 | val_seen / 8 (73 decisions) | privileged matched-state coarse/default/fine | — | — | — | run; candidate-level upper bound | `research/results/action_abstraction/oracle_valseen8_recorded_summary.json` |
| ORACLE-002 | val_seen / 64 (511 decisions) | privileged matched-state coarse/default/fine | 79.69%* | 70.16%* | 73.61%* | candidate-level upper bound; fine-default +0.285 m | `research/results/action_abstraction/oracle_valseen64_summary.json` |
| ORACLE-003 | val_seen / 64 (511 matched decisions) | default + oracle + diagnostics | 79.69%* | 70.16%* | 73.61%* | selection/execution gap measured | `research/results/action_abstraction/oracle_valseen64_selection_gap.json` |
| GRAPHSEL-001 | val_seen / 64 (511 matched decisions) | default + candidate-to-graph mapping | 79.69%* | 70.16%* | 73.61%* | graph selection gap 0.471 m; execution gap 0.052 m | `research/results/action_abstraction/graph_selection_valseen64_analysis.json` |
| GRAPHSEL-ORACLE-001 | val_seen / 64 (404 oracle decisions) | default + privileged graph selection | 100.00% | 98.26% | 92.58% | analysis-only intervention; 22.3% action overrides | `research/results/action_abstraction/graph_selection_oracle_valseen64_summary.json` |
| GRAPHSEL-CONTROL-RANK | val_seen / 64 (484 oracle decisions) | default + privileged rank-only selection | 96.88% | 88.24% | 88.46% | learned STOP preserved; 23.97% overrides | `research/results/action_abstraction/graph_rank_only_valseen64.stdout` |
| GRAPHSEL-CONTROL-STOP | val_seen / 64 (529 oracle decisions) | default + privileged STOP-only selection | 81.25% | 73.72% | 73.47% | learned ranking preserved; 16.45% overrides | `research/results/action_abstraction/graph_stop_only_valseen64_retry.stdout` |
| FIXED-VALSEEN-COARSE-64 | val_seen / 64 | coarse fixed | 62.50% | 54.20% | 66.29% | matched fixed comparison | `research/results/action_abstraction/fixed_coarse_valseen64.stdout` |
| FIXED-VALSEEN-DEFAULT-64 | val_seen / 64 | default fixed | 79.69% | 70.16% | 73.61% | matched fixed comparison | `research/results/action_abstraction/oracle_valseen64.stdout` |
| FIXED-VALSEEN-FINE-64 | val_seen / 64 | fine fixed | 71.88% | 58.83% | 68.19% | matched fixed comparison | `research/results/action_abstraction/fixed_fine_valseen64.stdout` |
| RANKATTR-001 | val_seen / 64 (511 decisions) | offline graph/heatmap ranking attribution | — | — | — | graph logit Spearman 0.531; heatmap 0.197 | `research/results/action_abstraction/candidate_rankers_valseen64.json` |
| RANKER-TRAIN-001 | train 64 → val_seen 64 | offline ridge candidate ranker | — | — | — | 54.99% top-1, 0.537 m regret; below graph logit | `research/results/action_abstraction/linear_ranker_train64_val64.json` |
| RANKER-PAIRWISE-001 | train 64 → val_seen 64 | offline pairwise ridge ranker | — | — | — | 54.60% top-1, 0.535 m regret; below graph logit | `research/results/action_abstraction/pairwise_ranker_train64_val64.json` |
| RANKER-EMBED-002 | train 320 (2 scenes) → val_seen 64 | offline frozen GraphMap embedding rankers | — | — | — | pairwise 57.34% top-1, 0.507 m regret; CI crosses 0 | `research/results/action_abstraction/embedding_ranker_train320_val64.json` |
| RANKER-EMBED-003 | train 960 (6 scenes) → val_seen 64 | offline frozen GraphMap embedding rankers | — | — | — | pointwise 58.71% / 0.510 m, pairwise 57.14% / 0.511 m; paired CIs cross 0 | `research/results/action_abstraction/embedding_ranker_train960_val64.json` |
| RANKER-EMBED-004 | train 960 (6 scenes) → val_unseen 64 (1 scene) | independent offline frozen GraphMap embedding validation | — | — | — | pairwise 63.78% / 0.347 m vs graph 62.37% / 0.368 m; historical decision CI `[-.052,+.010]` m | `research/results/action_abstraction/embedding_ranker_train960_valunseen64.json` |
| GRAPH-ALIAS-001 | train 960 + val_seen 64 | GraphMap candidate aliasing diagnostic | — | — | — | default aliases 49.4% train states; no-merge 12.4% matched val states but SPL/nDTW lower | `research/results/action_abstraction/graph_aliasing_train960.json` |
| GRAPH-ALIAS-CONTROL | val_seen / 64 | default vs `MODEL.merge_ghost=false` | 79.69% | 69.65% | 73.01% | no-merge control; default was 79.69/70.16/73.61, matched ghost count 28.50 vs 21.53125 | `research/results/action_abstraction/graph_no_merge_valseen64.stdout` |
| RANKER-VALIDITY-001 | same train960 / val_seen64 / unseen64 logs | action identity, route/episode-cluster CI, graph dedup sensitivity | — | — | — | historical metrics reproduced; pairwise episode CIs seen [-.08892,+.05327], unseen [-.05232,+.00929] m; 3,937 conflicting-label pairs | `research/results/ranker_validity/validity_001_rowwise_20261007/metrics.json` |
| OPTION-CALIBRATION-001 | val_seen / 64 / 4 scenes | default, traced versus untraced + two replay modes | 79.69% | 70.16% | 73.61% | both replay modes 511/511 pass, endpoint error 0; 1,152 episode metrics exact | `research/results/option_calibration/replay_full001/summary.json`, `noninterference_001/summary.json` |
| OPTION-CALIBRATION-002 | balanced val_unseen / 66 / 11 scenes | default, traced versus untraced + two replay modes | 66.67% | 54.55% | 60.56% | both replay modes 600/600 pass, endpoint error 0; 1,188 episode metrics exact; diagnostic subset only | `research/results/option_calibration/replay_full002/summary.json`, `noninterference_002/summary.json` |
| GRAPH-OPTION-PILOT-001 | train / 8 / 4 scenes | frozen default capture + all full-option branches | 100.00%* | 95.62%* | 89.21%* | 760 branches; 63 sentinels and 218 order checks pass; local gain .038215 m, path-capped 0 | `research/results/graph_option_pilot/analysis_001/summary.json` |
| GRAPH-OPTION-UNSEEN-001 | balanced val_unseen / 66 / 11 scenes | frozen default + all full-option branches, STOP/budget preserved | 66.67%* | 54.55%* | 60.56%* | 7,113 branches, 600 sentinels + 1,516 order checks pass; raw local gain 1.284608 m, joint-capped .148787 m; no learned gain | `research/results/graph_option_unseen/analysis_001/summary.json` |
| GRAPH-OPTION-TRAIN64-001 | train / 64 new routes / 16 scenes | frozen default, traced vs untraced | 85.94%* | 81.10%* | 83.15%* | 5,090 branches; 474 sentinels + 1,359 order checks pass; cost-capped .053452 m; 1,152 baseline metrics exact | `research/results/graph_option_train64/noninterference_001/summary.json` |
| ROUTE-COMPATIBILITY-001 | unseen66 + pilot8 + training64 | offline ordered-reference prefix gate, same joint execution-cost caps | — | — | — | unseen .126962 m / 49 material states; training64 .042087 m / 14; pilot 0; 414 native reconstruction metrics exact | `research/results/route_compatibility_001/unseen66_v2/summary.json` |
| GRAPH-SINGLE-INTERVENTION-TRAIN7-001 | original train64 / seven treated routes in five scenes | one frozen oracle choice then native continuation | 85.94%† | 81.71%† | 83.61%† | treated SR 4/7 unchanged; nDTW 5 up / 2 down; +37 primitives; other 57 exact | `research/results/single_intervention_train7/enabled_audit_001/summary.json` |
| COMPAT-ORACLE-001 | val_seen / 1 | default, oracle disabled | 100.00% | 64.694% | 83.369% | post-instrumentation smoke passed | `research/results/action_abstraction/postoracle_smoke.stdout` |
| CONTINUATION-LABEL-TRAIN64-001 | train / 64 new routes, 16 new scenes | unchanged baseline control | 87.500%* | 81.471%* | 81.578%* | baseline/control and full-option gates exact | `research/results/continuation_label_train64/` |
| CONTINUATION-LABEL-TRAIN64-001-H1 | same train / 64; 5 interventions | single H1 oracle intervention per eligible route | 87.500%‡ | 82.082%‡ | 82.278%‡ | executed; 59 routes exact; native metrics reconstructed | `research/results/continuation_label_train64/enabled_audit_001/` |
| CONTINUATION-LABEL-TRAIN64-001-H2 | same train / 64; 2 interventions | registered H2-confirmed oracle schedule | 87.500%‡ | 81.748%‡ | 81.778%‡ | separately executed; all composed traces exact | `research/results/continuation_label_train64/h2_enabled_audit_001/` |
| CONTINUATION-COST-DECOMPOSITION-001 | old 7 + new 5, kept separate | post-hoc complete cost accounting | — | — | — | all primitive/collision tie-outs pass | `research/results/continuation_label_train64/cost_decomposition_001/` |
| EQUAL-PRIMITIVE-CONTINUATION-001 | same 12 treated routes, separate batches | post-hoc budget-normalized labels | — | — | — | 456 geodesic checks exact; no new rollout | `research/results/continuation_label_train64/equal_primitive_001/` |
| FULL-RETURN-LABEL-TRAIN16-001 | train / 16 routes / 8 scenes | two outcome-blind full native continuations per state | 87.50%* | 84.15%* | 90.00%* | 32 alternatives: 1 dominates, 13 dominated, 18 mixed; NO-GO for fitting; all controls/reconstructions pass | `research/FULL_RETURN_LABEL_REPORT.md`, `research/results/full_return_label_train16/full_return_analysis_001/` |
| FULL-RETURN-FEATURE-AUDIT-001 | train / 16 routes / 8 scenes | post-hoc frozen graph-feature associations | — | — | — | cost geometry correlates with final cost, quality intervals cross zero; NO-GO for fitting | `research/FULL_RETURN_FEATURE_AUDIT_REPORT.md`, `research/results/full_return_feature_audit/analysis_001/` |
| FULL-RETURN-FEATURE-COST-001 | train / 16 routes / 8 scenes | post-hoc immediate/continuation cost decomposition | — | — | — | top +248 primitives (109 immediate/139 later); seeded +1008 (330/643/35); accounting exact | `research/FULL_RETURN_FEATURE_AUDIT_REPORT.md`, `research/results/full_return_feature_audit/cost_001/` |
| FULL-RETURN-BRANCH-ALIGNMENT-001 | train / 16 routes / 8 scenes | post-hoc one-step branch vs complete return | — | — | — | all 32 scheduled alternatives locally below native action; delayed rescue retained; no fitting | `research/FULL_RETURN_BRANCH_ALIGNMENT_REPORT.md`, `research/results/full_return_branch_alignment/analysis_001/` |
| NATIVE-STOP-FEASIBILITY-001 | train/validation diagnostic cohorts / 218 routes | post-hoc native STOP branches and primitive arrivals | — | — | — | successful STOP opportunity in 0/9 train64 failures, 1/22 unseen66 failures; no STOP-only model | `research/NATIVE_STOP_FEASIBILITY_REPORT.md`, `research/results/native_stop_feasibility/analysis_001/` |
| PRIMITIVE-ARRIVAL-LOCALIZATION-001 | five oracle-arrival baseline failures, cohorts separate | archived primitive poses and original navmeshes | — | — | — | 84 endpoint distances exact; unseen1593 has only three interior near-goal primitives | `research/results/native_stop_feasibility/primitive_arrival_001/` |
| INTERRUPTIBLE-EXECUTION-AUDIT-001 | released ETPNav code / existing cohorts | read-only execution/observability and literature audit | — | — | — | 13 static source anchors pass; front-node render is overwritten before high-level return; graph transaction required; NO-GO prototype | `research/INTERRUPTIBLE_EXECUTION_AUDIT.md`, `research/results/interruptible_execution_audit/analysis_002/`, `archive_001/` |


| Experiment | Population | Change | SR | SPL | nDTW | Result | Artifact |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CRITICAL-FULL-RETURN-TRAIN64-001 | train64 / all 9 failures / 8 scenes | exhaustive 509 native actions at 46 critical states, full native continuation | — | — | — | 6/9 SR rescues; 4/9 also preserve nDTW; termination rescue 2/9 overlaps; 3 unresolved | `research/CRITICAL_CAUSAL_REPORT.md`, `research/results/critical_causal_train64/analysis_001/` |
| CRITICAL-INTERRUPT-TRAIN4-001 | four failed training routes | matched sensing + ghost consume/retain interruption at frozen collision cuts | — | — | — | 0/4 rescued in both arms; matched sensing exact; no learned model | `research/results/critical_causal_train64/full_001/interrupt_plan.json` |


| Experiment | Population | Change | SR | SPL | nDTW | Result | Artifact |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CRITICAL-FULL-RETURN-UNSEEN66-001 | val_unseen diagnostic / 22 failures in 10 scenes (parent cohort: 11 scenes) | exhaustive 985 native actions at 71 critical states, full native continuation | — | — | — | 7/22 SR rescues; 6/22 also preserve nDTW; termination 6/22 overlaps; 14 unresolved | `research/CRITICAL_CAUSAL_REPORT.md`, `research/results/critical_causal_unseen/analysis_001/` |
| CRITICAL-INTERRUPT-UNSEEN4-001 | four unseen diagnostic failures | matched sensing + ghost consume/retain interruption at frozen collision cuts | — | — | — | 0/4 rescued; sensing exact; no labels/training | `research/results/critical_causal_unseen/full_001/interrupt_plan.json` |

The full comparison and paired scene-bootstrap intervals are in
`research/results/action_abstraction/full_comparison_summary.json` and
`research/REPORT.md`.

For ORACLE-002/003, starred SR/SPL/nDTW are the unchanged fixed-default
rollout metrics; the privileged oracle did not control actions.

OPTION-CALIBRATION runs use one RTX 4090 / one worker, seed 100, sliding=true
and hence effective tryout=false. Reported SR/SPL/nDTW belong to the unchanged
baseline, not replay episodes or a new policy. The unseen66 diagnostic subset
is balanced over scenes/routes and must not replace the standard full benchmark.

GRAPH-OPTION scores marked with an asterisk are the frozen baseline rollout,
not a learned/oracle controller score. Branch comparisons use privileged local
goal progress and do not constitute counterfactual episode SR/SPL/nDTW.

† SINGLE-INTERVENTION values are actual privileged-intervention results over
all 64 preselected **training** routes, not the untouched baseline, a learned
method or a validation benchmark. Seven routes are treated; the other 57 remain
exactly unchanged. Treated SR remains 4/7; treated SPL/nDTW improve by 5.557/4.239
pp, but two failures worsen and total primitive count rises by 37. All seven
paired outcomes appear in `SINGLE_INTERVENTION_REPORT.md`.

‡ These are executed privileged interventions on 64 new-to-diagnostics training
routes, not learned or held-out scores. Among five eligible baseline-successful
routes, SR stays 5/5. H1 saves 214 primitives but one route saves 244; H2 retains
two interventions and no aggregate primitive saving. Coverage is only three
treated scenes. The original NMS AAA and direct short-window distillation remain
NO-GO; see `CONTINUATION_LABEL_REPORT.md`.





* FULL-RETURN values are the unchanged 16-route baseline aggregate, repeated for context; they are not a learned or adaptive-policy score. Alternative aggregates decrease SPL/nDTW and increase primitive cost. Dominance is descriptive, not an exhaustive oracle upper bound.

| Experiment | Population | Change | SR | SPL | nDTW | Result | Artifact |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CRITICAL-GRAPH-VALUE-FRESH-001 | train / 64 routes / 16 new scenes; 10 failures | exhaustive 996 native actions at 47 critical states; full native continuation | — | — | — | 9/10 graph-choice rescue; 7/10 SR+nDTW rescue across 5 scenes; 6/10 termination overlap; no model trained | `research/GRAPH_VALUE_FEASIBILITY_REPORT.md`, `research/results/critical_graph_value_fresh/analysis_001/summary.json` |
| GRAPH-VALUE-FEASIBILITY-FRESH-002 | same fresh cohort / six scenes | corrected frozen-feature scene-grouped pairwise feasibility | — | — | — | 5,668 strict pairs; learned 0.9386 vs native logit 0.8520; predictions not executed | `research/results/critical_graph_value_fresh/feasibility_002.json` |
| GRAPH-VALUE-FEATURE-CORRECTION-001 | same fresh cohort + prior train-new cohort | corrected STOP-probability indexing and back-path cost extraction | — | — | — | rollout results unchanged; unit tests pass; prior JSON retained as superseded audit | `research/tools/test_graph_value_features.py` |
| GRAPH-VALUE-CONFIRMATION-POOL-AUDIT-001 | local R2R/RxR train pool | pre-sampling scene/route availability audit | — | — | — | 59/61 R2R scenes already excluded; 2 scenes/4 routes remain; no dataset written; independent confirmation NO-GO with current pool | `research/results/graph_value_confirmation_train96/scene_pool_audit_001.json` |
| GRAPH-VALUE-CROSS-COHORT-TRANSFER-001 | fresh 47-state train cohort -> prior 16-state / 8-scene cohort | corrected schema v2 offline transfer | — | — | — | learned 0.8421 = native logit 0.8421 over 19 strict pairs; no predicted action executed | `research/results/critical_graph_value_fresh/transfer_002.json` |

| Experiment | Population | Change | SR | SPL | nDTW | Result | Artifact |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GRAPH-VALUE-ROUTE-SAMPLING-001 | 96 train routes | retrospective exclusion audit | — | — | — | INVALID: 12/96 overlap with historical ranker; pilot only | `sampling_001/INVALID_OVERLAP_AUDIT.json` |
| GRAPH-VALUE-ROUTE-BASELINE-002 | 96 routes / 8 overlapping train scenes | corrected route-disjoint sample, native trace/control | 92.7083 | 86.9455 | 83.7453 | exact 1,728 episode and 18 aggregate comparisons | `GRAPH_VALUE_ROUTE_EXECUTION_REPORT.md` |
| CRITICAL-GRAPH-VALUE-ROUTE-DISJOINT-002 | all 7 failures / 5 scenes | 57 critical states / 561 native full returns | — | — | — | 4/7 SR rescue, 3/7 SR+nDTW; interrupt 0/4; independent audit passed | `confirmation_002/full_return_analysis_001/summary.json` |
| GRAPH-VALUE-ROUTE-EXECUTION-002 | all 96 routes, 27 changed | frozen old-cohort linear scorer, one early disagreement | 95.8333 | 89.2480 | 84.7011 | 3 rescues, 0 lost; −1.677 primitives/route; exploratory, not benchmark | `confirmation_002/paired_analysis_001/summary.json` |
| GRAPH-VALUE-ROUTE-RANDOM-002 | same 27 eligible states, 3 seeds | uniform native-admissible actions including abstention | Δ 0/−1.042/0 pp | Δ −5.223/−6.111/−5.792 pp | Δ −4.115/−4.332/−3.684 pp | all-route controls; +12.792/+12.448/+12.490 primitives/route | `confirmation_002/paired_analysis_001/summary.json` |


| Experiment | Population | Change | SR | SPL | nDTW | Result | Artifact |
| --- | --- | --- | ---: | ---: | ---: | --- | --- |
| GV-PROSPECTIVE-NATIVE-001 | 96 train routes / 8 overlapping scenes | preregistered research-route holdout | 91.6667 | 84.4428 | 84.8572 | 56.3125 primitives/route; exact baseline control | `GRAPH_VALUE_PROSPECTIVE_REPLICATION_REPORT.md` |
| GV-PROSPECTIVE-FULL-001 | all 96 routes | 29 changes; frozen one-action rule | 94.7917 | 86.2138 | 85.7952 | 4 rescued / 1 lost; primitive delta +1.2604/route | `results/graph_value_prospective_replication_001/paired_analysis_001/summary.json` |
| GV-PROSPECTIVE-COST_OWN-001 | all 96 routes | 74 changes; frozen one-action rule | 94.7917 | 86.6087 | 86.1980 | 4 rescued / 1 lost; primitive delta +3.4271/route | `results/graph_value_prospective_replication_001/paired_analysis_001/summary.json` |
| GV-PROSPECTIVE-COST_MATCHED-001 | all 96 routes | 23 changes; frozen one-action rule | 93.7500 | 85.4720 | 85.9446 | 3 rescued / 1 lost; primitive delta +1.0208/route | `results/graph_value_prospective_replication_001/paired_analysis_001/summary.json` |
| GV-PROSPECTIVE-RANDOM_20261031-001 | all 96 routes | 28 changes; frozen one-action rule | 91.6667 | 78.3281 | 78.9223 | 3 rescued / 3 lost; primitive delta +19.2917/route | `results/graph_value_prospective_replication_001/paired_analysis_001/summary.json` |
| GV-PROSPECTIVE-RANDOM_20261032-001 | all 96 routes | 26 changes; frozen one-action rule | 90.6250 | 76.2502 | 79.0493 | 1 rescued / 2 lost; primitive delta +17.4375/route | `results/graph_value_prospective_replication_001/paired_analysis_001/summary.json` |
| GV-PROSPECTIVE-RANDOM_20261033-001 | all 96 routes | 25 changes; frozen one-action rule | 90.6250 | 77.5448 | 78.7259 | 2 rescued / 3 lost; primitive delta +19.9792/route | `results/graph_value_prospective_replication_001/paired_analysis_001/summary.json` |
| GV-PROSPECTIVE-GATE-001 | all registered comparisons | unchanged point and feature gates | — | — | — | NO-GO for scaling: primitive gate fails; added-feature gate fails; all fidelity audits pass | `GRAPH_VALUE_PROSPECTIVE_REPLICATION_REPORT.md` |
| GV-PROSPECTIVE-COST-ACCOUNTING-001 | 29 FULL changes / all 96 routes | post hoc exact-prefix cost decomposition | — | — | — | option −133, continuation +254, total +121 primitives; 14 cheaper-option/costlier-episode routes | `results/graph_value_prospective_replication_001/route_diagnostics_001/summary.json` |


| Experiment | Population | Change | SR | SPL | nDTW | Result | Artifact |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PREFERENCE-CONTINUATION-MECHANISM-001 | 27 + 29 FULL events; 97 overlapping COST events | retrospective exact-prefix decomposition, no rollout | unchanged | unchanged | unchanged | 153 checks pass; FULL reversals 10/27 and 14/29, immediate symptoms 2/10 and 2/14 | `PREFERENCE_CONTINUATION_MECHANISM_REPORT.md` |
| PREFERENCE-SCORE-ATTRIBUTION-001 | all 56 FULL choices | exact frozen score decomposition, no ablation | unchanged | unchanged | unchanged | geometry largest positive group in 49/56 choices and 23/24 cost reversals; no policy fitted | `results/preference_continuation_mechanism_001/attribution_001/summary.json` |
| PREFERENCE-TRAINING-SUPPORT-001 | old 47-state / 996-action development census | specified only, not run | — | — | — | planned in-sample postmortem; cannot grant training GO | `PREFERENCE_TRAINING_SUPPORT_PROTOCOL.md` |


| Experiment | Population | Change | SR | SPL | nDTW | Result | Artifact |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PREFERENCE-TRAINING-SUPPORT-001 (completed) | 47 old fitting states / 996 actions / 10 failure routes | frozen model and native-relative full-return audit; no fit | not episode-policy metrics | — | — | strict all-pair 94.34% vs 85.20%, native-move 403/418 vs 404/418; two overrides: strict harm and mixed return | `PREFERENCE_TRAINING_SUPPORT_REPORT.md` |
| PREFERENCE-NATIVE-DIRECTION-001 | 418 audited strict native-move pairs | post hoc direction/class-balance split | — | — | — | native preferred 404, alternative preferred 14; model identifies 0/14 native improvements | `results/preference_training_support_001/directional_support_001/summary.json` |
| PREFERENCE-SUPPORT-VERIFICATION-001 | same archived fitting data | independent vector inequalities and masked sorting | — | — | — | 12,159 labels, 47 choices, 564 metrics all pass; six unit tests pass | `results/preference_training_support_001/verification_001/summary.json` |

| INTERRUPT-COVERAGE-001 (attempt002) | Two old training cohorts, seed100 | Offline eligibility enumeration; no new rollouts/model | Failed:156 cuts/56 options/11 routes,8 tested; success:>=128 cuts/43 options/27 routes | 7 tests;878 options/284 cuts/2 plans verified | CONDITIONAL GO for bounded timing oracle only; no interrupt learning | `INTERRUPT_COVERAGE_REPORT.md` |
