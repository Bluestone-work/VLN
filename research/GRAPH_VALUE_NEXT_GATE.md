# Long-horizon graph value / preference next gate

## 2026-10-07 — route-disjoint execution update

**CONDITIONAL GO for a prospective replication**, not a benchmark improvement.
The invalid first 96-route sample had 12 historical diagnostic overlaps and is
retained only as a pilot. The replacement 96-route / 8-scene training sample
passes the declared-route exclusion audit. Its complete census has 561 native
full returns at 57 critical states: 4/7 SR rescue opportunities, 3/7 also preserve
nDTW, 0/4 interrupt rescues, three unresolved routes. Independently audited.

A scorer trained only on the prior fresh 47-state census was actually executed
with one first-disagreement action per route: 27 changes, 3 rescues, 0 success
losses across all 96 routes. SR/SPL/nDTW change = **+3.125/+2.302/+0.956 pp**;
primitive count **−1.677/route**. Three matched random seeds have SR changes
0/−1.042/0 pp and all worsen mean SPL/nDTW/cost. All disabled/enabled and metric
reconstruction gates pass. SPL/nDTW/cost confidence intervals vs native still
cross zero; 13 treated routes lose nDTW. This is scene-overlapping, one-seed,
exploratory training follow-up after target census inspection, not untouched
validation. Late critical-state non-rescue missed an earlier step-3 rescue on
route 3161. Next: unchanged frozen scorer vs a simple cost-aware scorer on a
new prospectively frozen route sample; no adaptive density, RL, VLM or interrupt
policy training. Full evidence: `GRAPH_VALUE_ROUTE_EXECUTION_REPORT.md`;
protocol: `GRAPH_VALUE_ROUTE_INTERVENTION_PROTOCOL.md`; next gate:
`GRAPH_VALUE_ROUTE_NEXT_GATE.md`.

Decision after `CRITICAL-FULL-RETURN-TRAIN64-001` and
`CRITICAL-FULL-RETURN-UNSEEN66-001`: **CONDITIONAL GO for a feasibility study; do not train a deployable selector yet.** Route-quality-aware native-choice opportunities occur on 10/31 failures across the two retrospective cohorts; that is the leading mechanism, but not a strict majority. SR-only rescues are 13/31 and can hide nDTW degradation. Termination opportunities are 8/31 and interrupt rescues 0/8. Seventeen routes remain unresolved.

## Required next study

Use a newly registered training failure cohort for development, with scene-grouped splits and all unseen labels held out. For that prospective cohort, freeze features before reading continuation labels:

- current graph action logit and normalized rank;
- candidate bearing, distance, commanded polyline and back-path length;
- current proposal versus historical ghost flag;
- policy entropy and STOP probability;
- collision/revisit/history counters available to the deployed navigator;
- instruction embedding and candidate visual/topological embeddings already available in the baseline.

Do not use goal distance, reference route, oracle arrival, future collision or full-return outcome as a deployed feature. Do not use unseen66 labels to choose features, thresholds or loss weights.

The target is a route-quality-aware continuation utility, not one-step goal progress. For an action replacement, record success, SPL, nDTW, path length, primitive cost and native termination outcome after the unchanged navigator continues. Keep action identity and graph masks exact. A pair is eligible only when both alternatives are admissible at the same frozen state. Report all utility dimensions and an unresolved/mixed label; do not collapse a harmful nDTW change into a success label.

## Go / no-go gates

1. Leave-one-scene-out training on the nine existing failures is only a feasibility probe: the current data provide eligible route-quality pairs in three scenes, so this gate is not yet passed. A fold with no positive pair is reported as unsupported, not oversampled.
2. A frozen linear/logistic preference baseline must beat native logit on the held-out training scene fold in paired utility regret without using future features. This is feasibility evidence only.
3. The predictor must be evaluated on a separately sampled new training cohort with at least six scenes, using the same full-return intervention protocol.
4. It must preserve native STOP and primitive budget. Report route-level rescue, nDTW nondegradation, SPL, primitive cost and confidence intervals.
5. If fewer than two new scenes show route-quality rescue, stop. Do not add RL, PPO, VLM or a waypoint predictor change.

If the gates pass, compare a frozen preference ranker against native logit and a random selector with no policy retraining. Only then consider execution-aware joint fine-tuning. If gates fail, keep adaptive abstraction and graph-value learning NO-GO and investigate proposal coverage with an independent reachable waypoint oracle. This gate deliberately postpones RL/VLM.

## Fresh cohort update (2026-10-07)

The registered development cohort `CRITICAL-GRAPH-VALUE-FRESH-001` is now
complete. It covers 47 critical states and 996/996 native full-return action
cases from 10 failures in six new training scenes. Nine routes have some
non-STOP rescue and seven have a rescue with SR and nondecreasing nDTW. These
are overlapping oracle opportunities, not deployable results or a causal
majority. Four interrupt routes were tested; only one has a clean route-quality
rescue, so no interrupt policy is trained.

The corrected frozen feature analysis has 5,668 strict dominance pairs. Its
leave-one-scene-out pair accuracy is 0.9386 versus 0.8520 for native graph
logit. The previous `feasibility_001.json` is retained because it is part of the
audit trail; `feasibility_002.json` corrects STOP-probability indexing and
back-path cost extraction. The correction does not alter any rollout or rescue
count. Pair accuracy is still only a feasibility signal because predictions
were never executed.

The gate is therefore **CONDITIONAL GO** for one independently registered,
scene-grouped full-return confirmation. The confirmation must use the corrected
feature schema, preserve native STOP and primitive budget, and report route
quality by scene. Do not tune on unseen outcomes, insert the ranker into the
benchmark, or start RL/VLM work before that gate.
