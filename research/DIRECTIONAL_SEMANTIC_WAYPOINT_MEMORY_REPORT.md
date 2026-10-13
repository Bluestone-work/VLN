# Directional semantic waypoint memory validation

HEAD: `2963a0f33d757250c709d401ef0f7309ada71b84`

Causal dataset: `research/results/native_single_action_full_return_44_002/exact_audit_003/`
Directional join: `research/results/native_single_action_full_return_44_002/directional_pairs_002/`
Probe: `research/results/native_single_action_full_return_44_002/probe_002/summary.json`

Causal dataset:
states: 369 alternative-bearing states (373 source decision states)
actions: 3,878 alternatives
routes: 44
scenes: 11
BENEFIT: 430
HARM: 3,447
UNCERTAIN: 1 (excluded from classifier fitting)

Representation:
F0 native: native logits, action identity/geometry controls and the frozen 768d
native action embedding.
F1 geometry: F0 plus local bearing, sine/cosine bearing, distance and STOP flag.
F2 directional semantic: F0 plus current-node observed semantic landmark/relation
fields in the nearest 30° local sector.
F3 instruction-grounded directional semantic: F2 plus sector-filtered phrase and
relation evidence from the instruction-conditioned extractor.
A STOP action has no sector; it receives an explicit semantic abstention.
All bearings are computed by inverse-rotating the ghost displacement by the
pre-action quaternion. Synthetic front/left/right and rotated-agent tests pass.

Causal prediction (leave-one-scene-out, fixed unweighted logistic probe):

| arm | AUROC | AUPRC | within-state pairwise | benefit precision | harm precision |
|---|---:|---:|---:|---:|---:|
| F0 | 0.714 | 0.209 | 0.779 | 0.243 | 0.893 |
| F0+F1 | 0.719 | 0.211 | 0.782 | 0.262 | 0.893 |
| F0+F2 | 0.720 | 0.216 | 0.778 | 0.278 | 0.893 |
| F0+F3 | 0.720 | 0.216 | 0.778 | 0.278 | 0.893 |
| F3 without native | 0.383 | 0.113 | 0.543 | 0.000 | 0.889 |
| F0+F1+F3 | 0.726 | 0.218 | 0.783 | 0.278 | 0.893 |

Scene bootstrap on fixed OOF predictions:

- F0+F3 minus F0 AUROC 95% CI: **[-0.0016, 0.0130]**; pairwise CI
  **[-0.0036, 0.0026]**.
- F0+F1+F3 minus F0+F1 AUROC 95% CI: **[-0.0014, 0.0132]**.
- F0+F3 and F0+F2 are identical on this extractor because all instruction
  evidence is zero.

Complementarity:

| definition | native correct / semantic correct | native correct / semantic wrong | native wrong / semantic correct | native wrong / semantic wrong |
|---|---:|---:|---:|---:|
| F3 without native | 3,366 | 26 | 80 | 405 |
| F0+F3 | 3,388 | 4 | 16 | 469 |

On native-wrong rows, F0+F3 AUROC is 0.032; semantic-only is 0.319. The
semantic-only result is below chance and does not supply a reliable rescue signal.
The combined model changes little relative to native and its bootstrap interval
includes no improvement.

Semantic description quality:

- source rows: 373; parse errors: 3
- object/direction/relation precision: **NOT MEASURED** (no held-out annotations)
- all 373 global confidences are 0
- all 373 rows abstain under the confidence gate; 3,877 causal alternatives
  therefore have no deployable semantic confidence
- 389 emitted landmarks and 381 relations have zero confidence; 760 instruction
  evidence entries have no bearing and zero score/confidence

The extractor is a frozen Qwen2-VL structured hypothesis generator. Its generic
text output is retained for an outcome-blind diagnostic only. F2 is a visual-field
ablation of an instruction-conditioned extractor, not an independent visual-only
semantic model.

Shadow deployment:
Does semantic branch alter native trajectory? **NOT RUN**. The offline GO gate
failed, so no shadow branch was started.

Online intervention:
native-success retention: NOT RUN
rescued failures: NOT RUN
destroyed successes: NOT RUN
risk-coverage: NOT RUN

A0/A1/A2/A3 paired benchmark: **NOT RUN** because the offline complementarity
and semantic-quality gates failed.

Conclusion:
Does Directional Semantic Waypoint Memory provide complementary deployable signal?
**NO**.

GO / NO-GO: **NO-GO**. Exact causal outcomes and strict semantic joins pass, but
explicit directional semantic memory adds no stable scene-disjoint signal beyond
native ETP features. The textual representation itself is uncalibrated and
abstains on every row. Shadow and online intervention are therefore prohibited.

If GO: not applicable.

If NO-GO: the complementary-signal and representation-quality hypotheses failed;
keep this direction offline and do not insert semantic nodes into the native
Graph Transformer.

Next single highest-value experiment: obtain held-out viewpoint annotations and
regenerate a genuinely confidence-calibrated, instruction-grounded extractor on
the same 44-route pre-action capture; rerun only the offline causal probe before
considering shadow deployment.
