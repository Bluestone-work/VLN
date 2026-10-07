# Ranker validity audit — 2026-10-07

## Decision

Keep the current NMS-based Adaptive Action Abstraction formulation at **NO-GO
for learned-selector development**. Keep graph ranking at **CONDITIONAL GO for
measurement calibration only**. Neither a learned adaptive selector nor a
deployed ranker is supported by these experiments. This does not reject every
possible action abstraction.

## What was actually sampled

| Extraction | Episodes | Routes | Scenes | Raw decisions | Eligible decisions |
| --- | ---: | ---: | ---: | ---: | ---: |
| train960 | 960 | 328 | 6 | 7,813 | 7,803 |
| val_seen64 | 64 | 23 | 4 | 511 | 511 |
| val_unseen64 | 64 | 46 | **1** | 497 | 497 |

The unseen extraction covers only `zsNo4HB9uLZ`. It is held out from training,
but is not evidence of multi-scene generalization. Train scenes are also
imbalanced: episode counts are 147, 210, 192, 202, 196 and 13. Sequentially
taking the first N episodes does not produce a scene-balanced sample.
There is no train/validation overlap on `(scene_id, trajectory_id)`.

## Reproduction and uncertainty

The historical row-wise scoring calculation was reproduced exactly for both
pointwise/pairwise regret and top-1 on both validation sets. No hyperparameter
search or new trajectory intervention was performed. The original CIs resampled
individual decisions; those decisions share episodes and often routes.

Paired regret differences are learned minus graph-logit regret; negative is
better. All intervals below use 5,000 whole-cluster resamples with seed 20261007
and preserve the decision-weighted estimand.

| Validation | Scorer | Delta (m) | Episode-cluster 95% CI | Route-cluster 95% CI |
| --- | --- | ---: | --- | --- |
| val_seen64 | pointwise | -0.01207 | [-0.04097, +0.01425] | [-0.04546, +0.01950] |
| val_seen64 | pairwise | -0.01082 | [-0.08892, +0.05327] | [-0.07429, +0.05020] |
| val_unseen64 | pointwise | +0.01737 | [-0.02182, +0.05815] | [-0.02037, +0.05895] |
| val_unseen64 | pairwise | -0.02079 | [-0.05232, +0.00929] | [-0.05238, +0.01153] |

Four-scene bootstrap estimates for val_seen are exploratory. An unseen
scene-level interval is deliberately **not reported** because there is one
scene. Bootstrap seeds are not independent training/evaluation seeds.

## Action identity and label mismatch

1. All candidate counts, indices, wrapped angles, distances, scene IDs and
   pre-decision goal distances match between paired diagnostic/oracle records.
   Duplicate decision keys are rejected, not silently overwritten.
2. Of 29,710 eligible train candidate rows, there are only 25,930 unique
   state/graph actions. There are 3,937 pairs with **identical embeddings and
   logits but different progress labels**, 8.77% of 44,887 legacy training
   pairs. These zero-feature-difference pairs cannot be fitted by a scorer of
   that graph embedding. This is a label-contract issue, not evidence that a
   larger model is necessary.
3. The direct probe in `cand_dist_to_goal` rotates the agent continuously then
   moves it forward. The actual graph option executes the merged ghost
   position via a front node/back path, quantized turns and possible tryout.
   They are not the same action. Restoring the agent pose alone does not certify
   full worker/task/RNG rollback.
4. Existing labels also cover current proposals only; valid historical ghosts
   and learned STOP are outside the offline ranker comparison. Thus local
   candidate top-1 is not full graph-action policy accuracy.

Geometric evidence from selected actions: among 426 val_seen ghost decisions
represented by current candidates, 75 (17.6%) have a selected ghost target more
than 10 cm from **every** corresponding raw candidate position (maximum
24.6 cm). On unseen, this is 86/415 (20.7%), maximum 32.5 cm. These are target
offsets, not collision or execution-failure rates.

Consequently the earlier 0.471 m selection / 0.052 m execution decomposition
is a **proxy decomposition**, not a causal separation of ranking and controller
failure. The measured rank-only oracle SR 96.88% versus baseline 79.69% remains
an actual privileged rollout result, but neither that intervention nor the
local candidate oracle is a certified optimal upper bound on full graph actions.

## Sensitivity to ties and graph aggregation

The original baseline graph-logit regret changes from 0.521685 to 0.511697 m on
val_seen solely by choosing the last rather than first tied candidate row;
its graph action has not changed. On unseen it changes from 0.367837 to
0.377520 m. Candidate-level labels can therefore move the score without any
change in graph policy behavior.

Aggregating aliased labels by mean/min/max removes duplicate graph entries,
but these aggregates are only sensitivity conventions, not true graph-option
outcomes. Under all three conventions, both validation sets still have pairwise
episode-cluster delta intervals crossing zero. No positive method claim follows.

A first audit run used a batched BLAS score that perturbed machine-precision
alias ties and did not exactly reproduce historical candidate metrics. It is
retained as `validity_001_20261007`, **superseded and excluded**. The accepted
run `validity_001_rowwise_20261007` uses the original row-wise dot product and
matches all eight historical metric anchors exactly.

## Reporting correction

The matched 64-episode baseline ghost count is **21.53125**, not 23.85.
23.8499 belongs to the full 1,839-episode unseen run. The matched no-merge
comparison is therefore 21.53125 -> 28.50 ghosts (+32.37%), SR 79.69% unchanged,
SPL 70.1608% -> 69.6514%, nDTW 73.6074% -> 73.0117%. Previous language mixing
these populations is corrected in the current report/table; original artifacts
remain unchanged.

## Next measurement, before another model

Pre-register a small controller-consistent label calibration, preserving the
fixed default abstraction and frozen navigator. Capture each actual graph option
(ghost mean, front/back path, tryout flag), pre-state and RNG; evaluate it in an
isolated worker using the same controller. First show that probing the selected
option reproduces its real endpoint, primitive count and collision outcomes,
and that the analysis-disabled baseline still matches. Do not collect labels
for all actions until this gate passes. Include historical ghosts and identify
STOP/backtrack separately. Goal/reference data remain privileged labels only.

Then collect a preselected, scene-balanced train/held-out set and assess route-
and scene-cluster uncertainty without tuning on validation. Only stable offline
gains with faithful labels justify a small paired navigation intervention.

## Artifacts and provenance

- Config: `research/configs/RANKER_VALIDITY_001.json`.
- Accepted metrics and frozen scores: `research/results/ranker_validity/validity_001_rowwise_20261007/`.
- Command: `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/wj/miniconda3/envs/etpnav_legacy/bin/python research/tools/audit_ranker_validity.py --config research/configs/RANKER_VALIDITY_001.json --output-dir research/results/ranker_validity/validity_001_rowwise_20261007`.
- Source/checkpoint/dataset hashes, Python/NumPy versions and git commit are in
  the run manifest. The working tree is dirty; hashes captured now cannot
  reconstruct missing historical uncommitted snapshots.
- stdout/stderr are adjacent to the run directory. No old result was replaced.
- Geometry audit: `research/results/ranker_validity/selected_target_geometry_20261007.json`.
