# Directional semantic waypoint memory validation

HEAD: `32deb0b4772edf7f3f992f55a624ee700714d2a5`

## Causal dataset

states: 10
actions: 70 alternatives (80 including 10 native sentinels)
routes: 2 episodes
scenes: 1
BENEFIT: 2
HARM: 68
UNCERTAIN: 0

The exact native/alternative full-return table is
`research/results/native_single_action_full_return_003/causal_pairs/pairs.jsonl`.
All 37 disabled sentinel schedules reproduced the original 34-row source trace
and evaluator metrics exactly. The two episodes are in one scene, so the
required scene-disjoint probe is not estimable from this causal cohort.

## Representation

F0 native: native action logit and native graph/action identity.
F1 geometry: action type, current-node-to-ghost bearing, distance and sector.
F2 directional semantic: current visited node's observed 360° semantic landmark
and relation support in the action's nearest 30° sector.
F3 instruction-grounded semantic: the same sector record plus frozen ordered
instruction evidence and phrase/relation confidence.

All semantic inputs come from the current node's pre-action 360° observation.
Alternative endpoint images occur only in the causal audit, never as online
features. Goal distance, reference path and final outcome occur only as labels.

## Causal prediction

AUROC: **NOT ESTIMABLE** (scene-disjoint split has one scene)
AUPRC: **NOT ESTIMABLE** (scene-disjoint split has one scene)
pairwise accuracy: **NOT ESTIMABLE** (scene-disjoint split has one scene)
benefit precision: **NOT ESTIMABLE**
harm precision: **NOT ESTIMABLE**

The native logit orders the alternative below the selected native action in 68/70
rows and therefore agrees with the fixed label on 68/70 rows (0.9714) in this
same-scene descriptive audit. This is not a held-out probe result. The semantic
extractor has zero instruction evidence/confidence on these rows, so F3 has no
validated independent score to fit.

## Complementarity

Using the descriptive native-logit ordering, native predicts HARM on 68 rows and
is wrong on the 2 BENEFIT rows. F3 abstains (zero evidence) on all 70 rows.

```text
native correct / semantic correct: 0 (no semantic prediction)
native correct / semantic wrong: 68 (semantic abstention; no rescue signal)
native wrong / semantic correct: 0
native wrong / semantic wrong: 2 (semantic abstention)
```

These counts are not a scene-disjoint classifier confusion matrix. They record
the required native-error audit without converting abstention into a positive
semantic claim. The larger 44-episode semantic validation independently reports
native-correct/semantic-correct 0, native-correct/semantic-wrong 33,
native-wrong/semantic-correct 0 and native-wrong/semantic-wrong 11 for the same
zero-confidence extractor.

## Semantic description quality

object precision: **NOT MEASURED** (no held-out human object labels)
direction accuracy: **NOT MEASURED** (no held-out bearing labels)
relation accuracy: **NOT MEASURED** (no held-out relation labels)
hallucination rate: **NOT MEASURED** (no held-out annotation protocol)

The frozen Qwen2-VL extractor emits structured fields and abstains with zero
confidence in the audited 44-episode run. Generic CLIP similarity is retained as
a probe only and cannot fill these quality metrics.

## Shadow deployment

trajectory unchanged? **NOT RUN**. The offline scene-disjoint GO gate failed, so
no shadow policy was started and no action, graph state or controller was changed.

## Online intervention

native-success retention: **NOT RUN**
rescued failures: **NOT RUN**
destroyed successes: **NOT RUN**
risk-coverage: **NOT RUN**

No A0/A1/A2/A3 paired benchmark was run because the causal probe gate did not
pass.

```text
A0: released ETPNav — NOT RUN
A1: geometry-only side memory — NOT RUN
A2: directional semantic memory — NOT RUN
A3: instruction-grounded directional semantic memory — NOT RUN
```

## Conclusion

Does Directional Semantic Waypoint Memory provide complementary deployable signal?
**NO-GO / UNRESOLVED FOR DEPLOYMENT.** Exact causal joining now works, but the
available causal cohort has one scene and the frozen textual extractor supplies
no calibrated instruction-grounded signal. The required complementarity and
safety conditions therefore do not hold.

GO / NO-GO: **NO-GO**.

If GO: not applicable.
If NO-GO: the scene-disjoint causal probe and held-out description-quality gate
failed; semantic evidence cannot yet predict long-term BENEFIT/HARM beyond native
ETP memory.

Next single highest-value experiment: repeat exact native single-action
full-return collection for the frozen 44-episode, 11-scene native-success
cohort, then run the preregistered F0/F1/F2/F3 scene-disjoint probe with a
held-out viewpoint annotation audit. Do not start shadow or intervention before
both gates pass.
