# Dense Action A2 Control Experiment

## Question

This experiment tests whether isolating native graph scoring from newly inserted dense ghosts reduces the damage caused by dense action expansion. It separates representation contamination (H1) from dense-action ranking/calibration failure (H2).

The cohort is the frozen outcome-blind val-unseen control replication: 44 routes, 11 scenes, checkpoint `ckpt.iter12000.pth`, seed 100, native controller, STOP mechanism, evaluator, and decision budget unchanged. A0 and A1 results are the previously completed paired replication; A1-capture and A2 were run on the identical dataset.

## Arms

- **A0**: released native graph action set and native argmax.
- **A1-capture**: dense native-union action set and unchanged full-graph argmax, with an isolated native graph encoder captured for diagnostics only.
- **A2**: dense native-union action set; native actions receive logits from the isolated native graph encoder, while dense actions retain full-graph logits. This is opt-in via `ACTION_ABSTRACTION.DENSE_ISOLATION_MODE=a2`.

No model was trained and no threshold, evaluator, controller, horizon, or success definition was changed.

## Navigation results

| arm | SR | SPL | nDTW | path length | collisions |
|---|---:|---:|---:|---:|---:|
| A0 | 44/44 = 1.000 | 0.8182 | 0.7451 | 11.074 | 0.107 |
| A1-capture | 29/44 = 0.659 | 0.5620 | 0.6728 | 13.475 | 0.135 |
| A2 | 33/44 = 0.750 | 0.5862 | 0.6482 | 12.711 | 0.210 |

A1-capture exactly reproduces the prior A1 control metrics, establishing that diagnostics are non-interfering. Relative to A1, A2 rescues 4 routes (`1778`, `413`, `14`, `382`) and changes 4 success outcomes in the positive direction. Relative to A0, A2 still destroys 11 native successes, versus 15 destroyed by A1. A2 therefore reduces harm by 4 routes but does not make dense expansion safe. Its mean nDTW is lower than both A0 and A1, and its collision rate is higher than both.

The paired success changes are:

- A1 minus A0: `-15` routes, mean nDTW delta `-0.0722`, mean SPL delta `-0.2562`.
- A2 minus A0: `-11` routes, mean nDTW delta `-0.0969`, mean SPL delta `-0.2321`.
- A2 minus A1: `+4` routes, mean nDTW delta `-0.0246`, mean SPL delta `+0.0242`.

Scene-cluster bootstrap (10,000 resamples over 11 scenes) gives SR-delta intervals:

- A1 minus A0: `[-0.456, -0.220]`.
- A2 minus A0: `[-0.409, -0.132]`.
- A2 minus A1: `[0.000, 0.164]`.

The A2 improvement is not driven by a single scene, but the interval touching zero means this is mitigation evidence, not a validated safe intervention.

## Representation and calibration diagnostics

The A1-capture run produced 301 matched decision rows and 3,915 native option comparisons. Compared with full-graph scoring, isolated-native scoring had:

- mean absolute native logit shift: `1.6682` (median `1.4690`, p95 `3.7817`);
- mean native embedding L2 shift: `6.8026` (median `5.7876`, p95 `13.3083`).

These shifts are large relative to the action logits used for argmax and directly confirm H1: adding dense ghosts changes native representations even before a dense action is selected. A2 removes this contamination from native-action logits, but dense actions still use a score distribution produced in the dense graph context. The remaining 11 destroyed native successes therefore provide direct evidence for H2/OOD calibration risk after H1 is partially controlled.

## Decision

**NO-GO for unconditional dense expansion.** A2 is a useful diagnostic and partial harm-reduction mechanism, but it is not a safety gate: native-success destruction remains 11/44 on this frozen unseen cohort, with degraded nDTW and increased collisions.

Do not train a ranker, add g3D-LF, or start PPO/GRPO from this result. The next experiment should be representation-level calibration analysis of dense-vs-native score distributions on the frozen A0/A1/A2 traces, with no policy intervention and no validation-driven threshold tuning.

## Reproduction

Configs:

- `research/configs/DENSE_A1_ISOLATION_CAPTURE_CONTROL_VAL_UNSEEN_001.json`
- `research/configs/DENSE_A2_CONTROL_VAL_UNSEEN_001.json`

Outputs:

- `research/results/dense_a1_isolation_capture_control_val_unseen_001/run2/`
- `research/results/dense_a2_control_val_unseen_001/run/`
- `data/logs/eval_results/dense_a1_isolation_capture_control_val_unseen_001/`
- `data/logs/eval_results/dense_a2_control_val_unseen_001/`

The runs were executed in `etpnav_legacy` and retain manifests, source hashes, route dataset path, config, trace, stdout, and stderr.
