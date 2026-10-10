# Dense Context Only (A2-MASK) Protocol

## Objective

Measure closed-loop harm caused by inserting dense ghosts into the graph context while preventing every dense action from execution. This isolates representation contamination from direct dense-action selection.

## Frozen arms and cohort

- A0: native graph, native scoring, native executable actions.
- A1: dense native-union graph, full dense scoring, native and dense executable actions.
- A2-ISO: dense graph, isolated-native scores for native actions and full scores for dense actions, all executable.
- A2-MASK: dense graph and full dense scoring, but final executable logits are masked to native actions after SAP scoring.

All arms use the frozen 44-route val-unseen native-success control cohort, checkpoint `data/logs/checkpoints/release_r2r/ckpt.iter12000.pth`, seed 100, native controller, STOP mechanism, `ALLOW_SLIDING=True`, evaluator, success threshold, and decision budget.

## Mask placement and validity

The model first encodes `G_native + G_dense` and computes full SAP logits. Only then are indices outside `gmap_native_masks` set to negative infinity for policy selection. The native mask includes STOP, current native ghosts, and historical native ghosts as emitted by GraphMap. Dense nodes remain present in encoder attention and full-logit diagnostics.

The capture must record full-context logits, native mask, selected effective action, and action provenance. A2-MASK is valid only if every captured effective action is native and dense executable count is zero.

## Metrics and inference

Report SR, SPL, nDTW, SDTW, path length, primitive actions, collisions, changed routes, and destroyed native-success route IDs for all four arms. Report scene-cluster bootstrap (10,000 resamples) CIs for A2-MASK-A0, A2-MASK-A1, and A2-ISO-A1 on SR, SPL, nDTW, primitive count, and collisions. No threshold or evaluator tuning is allowed.

## Decision rule

- If A2-MASK remains materially below A0, representation contamination is independently harmful and a shadow candidate architecture is required.
- If A2-MASK is near A0 while A1 is harmful, direct dense ranking/override dominates.
- If both A2-MASK and A2-ISO only partially recover A0, H1 and H2 are coupled.
