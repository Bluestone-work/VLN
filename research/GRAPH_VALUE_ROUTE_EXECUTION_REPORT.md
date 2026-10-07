# Route-disjoint graph preference execution study — 2026-10-07

## Current decision

**CONDITIONAL GO for a new prospective replication; no benchmark or stable generalization claim.** The frozen linear scorer rescued 3/7 native failures and lost 0/89 native successes on 96 training routes. It met the declared point-estimate gate and outperformed all three matched random controls. SPL/nDTW and primitive-cost intervals against native baseline still include zero. Do not restart adaptive coarse/default/fine or add RL/VLM.

## Sampling validity

Sampling 001 is invalid confirmation: 12/96 routes overlap the old ranker_embedding_train960 diagnostics, all in gTV8FGcVJC9. It is retained as a pipeline pilot, with the overlap audit and all attempts preserved. The corrected sampler canonicalizes scene prefixes and resolves episode IDs to whole routes, excluding sibling instructions together. Sampling 002 excludes all old pilot routes, five prior cohorts, four ranker logs and both validation splits: 1,488 unique excluded route keys; 96 selected routes, 8 scenes, zero overlap against these declared sources. The source, output and exclusion hashes were independently checked.

Route disjointness is relative to research diagnostic labels. Scenes overlap and the released baseline checkpoint was trained on R2R train. The native baseline and census were inspected before this follow-up ranker schedule was frozen. Therefore this is an exploratory transfer/execution study, not untouched prospective validation. The old strict scene-independent protocol reference in the frozen capture config is explicitly superseded for this study by GRAPH_VALUE_ROUTE_INTERVENTION_PROTOCOL.md; the historical file/config is not rewritten.

## Native baseline and causal census

Native baseline SR/SPL/nDTW = 92.7083/86.9455/83.7453%; NE 1.6708 m; path 9.3162 m; 58.125 primitives and 7.5 high-level decisions per episode. Trace on/off matched exactly for 1,728 per-episode and 18 aggregate metric comparisons.

The option probe executed 7,829 forward and 1,911 reversed-order single options at 720 decisions; these 9,740 branches are **not full-episode continuations**. All action/RNG/coverage checks passed.

The failure-critical census separately executed all 561 actions at 57 critical states across all 7 failures in 5 scenes, followed by native navigation through STOP or the unchanged 15-decision limit. Independent audit: 5,631 exact option/prefix checks, 67,409 primitive records, eight interrupt prefix/cost checks; all passed. Full-return reconstruction made 4,640 metric comparisons.

| Opportunity | Routes | Scope |
| --- | ---: | --- |
| Alternative non-STOP graph choice reaches success | 4/7 | 4 scenes |
| Above, also nondecreasing nDTW | 3/7 | 3 scenes; not a majority |
| Termination change can rescue | 2/7 | Overlaps other rescue sets |
| First-event interrupt oracle rescue | 0/4 | Two ghost consume/retain arms per tested route |
| No rescue under registered interventions | 3/7 | Does not prove proposal failure |

Interrupt cuts were preselected first observable ghost-execution events, not exhaustive cut times. Re-observation-only controls were exact. These results provide no positive justification for scaling an interrupt policy.

## Actual frozen scorer interventions

The scorer uses schema-v2 existing graph features, fixed ridge=10, and only the prior fresh training census (47 states, 996 actions, 5,668 strict dominance pairs). Target labels are absent from fitting and schedule selection. At the first native non-STOP decision where its top admissible action disagrees, replace one action; afterwards the original navigator controls everything. Native STOP, masks, sensors, checkpoint, ghost consumption, simulator seed=100 and horizon stay unchanged. This is an offline exact-state intervention harness, not yet an online released policy.

27/96 routes are changed; 69 remain exact native controls. Random arms use the same 27 eligible states, sample uniformly from the identical admissible non-STOP set including native, and can abstain. Seeds 20261021/22/23 produce 23/20/25 changed routes. All 96 routes remain in every denominator. Every arm passed its own disabled control, enabled prefix/action/sensor/RNG audit, and independent six-metric reconstruction (576 comparisons per arm).

| Arm | SR % | SPL % | nDTW % | Primitives/route | Rescued / harmed |
| --- | ---: | ---: | ---: | ---: | ---: |
| Native | 92.7083 | 86.9455 | 83.7453 | 58.125 | — |
| learned | 95.8333 | 89.2480 | 84.7011 | 56.448 | 3 / 0 |
| random_20261021 | 92.7083 | 81.7226 | 79.6299 | 70.917 | 2 / 2 |
| random_20261022 | 91.6667 | 80.8344 | 79.4133 | 70.573 | 0 / 1 |
| random_20261023 | 92.7083 | 81.1540 | 80.0613 | 70.615 | 1 / 1 |

| Learned − native | Mean change | Scene bootstrap 95% interval |
| --- | ---: | --- |
| SR (pp) | +3.125 | [+1.042, +6.250] |
| SPL (pp) | +2.302 | [-0.603, +5.621] |
| nDTW (pp) | +0.956 | [-1.198, +3.368] |
| Primitives | -1.677 | [-6.365, +2.146] |

Bootstrap: 8 scene clusters, 5,000 replicates, seed 20261012; exploratory intervals. Three random seeds are control choices, not three independent training seeds. Compared with the mean random arm, learned SR is +3.472 pp (95% scene CI +0.694 to +7.292 pp); SPL +8.011 pp; nDTW +5.000 pp; primitives −14.253 per route.

## Concrete rescues and harms

| Episode | Decision step (zero-based) | ΔSR | ΔnDTW (pp) | Δ primitives |
| --- | ---: | ---: | ---: | ---: |
| 3161 | 3 | +1 | +45.733 | −82 |
| 3398 | 7 | +1 | +20.750 | 0 |
| 10199 | 4 | +1 | +79.661 | −155 |

All three rescues come from different scenes and preserve nDTW. But among the 27 treated routes, 13 lose nDTW and 10 lose SPL; 12 use more primitives. Three large rescues must not hide these harms. Mean high-level decisions increase by 0.073 per route despite fewer primitives. No online end-to-end inference-time gain has been measured.

Route 3161 has no rescue in its census states 13/14 but is rescued by a step-3 choice. This exposes a limitation of failure-critical timing: missing late native-action rescue is not evidence of proposal failure. These are overlapping intervention opportunities, not a disjoint causal partition. Do not pool old and new cohorts to turn this cohort’s 3/7 quality rescue rate into a confirmation majority.

## Interpretation and next experiment

Learning an execution-cost-sensitive preference is now worth a targeted replication, but the full mechanism remains unresolved. The largest-magnitude standardized weight penalizes back-path length; state-level entropy, STOP probability and candidate count cancel in linear pairwise scoring. Thus this study cannot claim uncertainty/instruction conditioning or uniquely establish long-horizon reasoning.

The single most informative next experiment is a fresh route-disjoint replication comparing this unchanged frozen scorer against a reduced graph-logit/rank plus execution-distance scorer fitted only on the same old training corpus. Freeze sampling, weights, first-disagreement timing and controls before any new target outcomes. See GRAPH_VALUE_ROUTE_NEXT_GATE.md. Do not tune on this target, broaden model capacity, or add RL/VLM.

## Reproducibility and limitations

Artifacts are under `research/results/graph_value_route_disjoint_confirmation_train96/`. Read sampling_001/INVALID_OVERLAP_AUDIT.json; sampling_002/ROUTE_DISJOINT_AUDIT_002.json; confirmation_002/{full_return_analysis_001,independent_audit_001,paired_analysis_001}/summary.json and intervention_cycle_001/*_enabled_audit/paired_episodes.json. Models, schedules, configs, source hashes and logs are retained. Execution HEAD was 9a9a7ae98bf61138da8e12da6fdce00269c3f82b with recorded dirty-source diffs; the later archive commit is not misrepresented as execution HEAD.

Foundation and intervention cycles save separate stdout/stderr. Critical smoke/full001 were launched without separate complete stdout/stderr archives; native logs, manifests, per-case traces and result JSONL exist, and this omission is retained in cycle_limitations_001.json. Legacy intervention hook flags still say privileged/no-model-trained; schedule/audit metadata correctly identify learned action choice in the offline harness. No core navigation code changed this cycle.

Validation this cycle: 16 route-related tests and 13 intervention/audit tests passed; bytecode compile and diff checks passed. The invalid pilot and superseded schedule_ranker_001 are preserved and excluded from accepted results.
