# Long-horizon graph value / preference next gate

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
