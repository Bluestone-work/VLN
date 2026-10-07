# ETPNav Baseline Report

Date: 2026-10-06  
Code commit at audit: `1c1a794`  
Environment: `etpnav_legacy` (Python 3.6.13, PyTorch 1.9.1+cu111, Habitat-Lab/Sim 0.1.7 headless EGL)  
Hardware: 2 x NVIDIA GeForce RTX 4090 for the published reference; smoke run used GPU 0.
Random seed: Habitat default `SEED=100` (no seed override).

## Reproduction controls

The published reference uses the released ETPNav R2R checkpoint
`data/logs/checkpoints/release_r2r/ckpt.iter12000.pth` and language checkpoint
`data/pretrained/ETP/mlm.sap_r2r/ckpts/model_step_82500.pt`. The original full
reference was evaluated on R2R-CE `val_unseen`, 1,839 episodes, with
`IL.back_algo=control`, `ALLOW_SLIDING=True`, 2 distributed processes and 8
environments per process. The exact effective command is preserved in
`results/r2r_baseline_command.txt`; the machine-readable reference is copied to
`research/results/baseline/r2r_release_reference.json`.

The first current-cycle smoke run used the same checkpoint and language model,
single GPU, one environment, `val_seen`, one episode, `RL_TOPO.ENABLED=False`,
and `IL.back_algo=control`. Command and stdout are in
`research/results/baseline/SMOKE_COMMAND.txt` and
`research/results/baseline/aaa_baseline_smoke2.stdout`. The evaluation completed
without changing the success threshold, sensor setup, or simulator version.

## Reference full validation results

| Dataset/split | Episodes | SR | SPL | nDTW | sDTW | NE | Path length | Primitive steps | Collisions | Ghost count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| R2R-CE val_unseen | 1839 | 0.570962 | 0.491014 | 0.623051 | 0.467859 | 4.734436 | 11.569253 | 80.274063 | 0.127652 | 23.849920 |
| RxR-CE val_unseen | 11006 | 0.554334 | 0.451361 | 0.623533 | 0.456984 | 5.728692 | 18.065586 | 166.566330 | 0.363841 | 33.551247 |

Existing full evaluation artifacts are:

- R2R: `data/logs/eval_results/release_r2r/stats_ckpt_59_val_unseen.json`
- RxR: `data/logs/eval_results/release_rxr/stats_ckpt_97_val_unseen.json`
- raw logs: `logs/r2r_full_baseline.log`, `logs/rxr_full_baseline.log`

The full reference does not contain waypoint candidate counts or inference
latency. Those are being collected by the opt-in action diagnostics rather than
retroactively inferred from episode averages.

## Current-code full reproduction

The current code was rerun after the diagnostic extension on the identical
R2R-CE `val_unseen` protocol. Its result is archived at
`research/results/baseline/r2r_current_full_result.json`, with stdout in
`research/results/baseline/aaa_baseline_full_current.stdout` and the exact
command in `research/results/baseline/R2R_FULL_CURRENT_COMMAND.txt`.

The standard metrics match the released reference to approximately `5e-7` or
better for every field (the JSON files use different floating-point precision):

```text
success          0.5709624887
SPL              0.4910142971
nDTW             0.6230507386
sDTW             0.4678593546
distance         4.7344364961
path length      11.5692529678
primitive steps  80.2740631104
collisions       0.1276516169
ghost count      23.8499202728
```

This is an exact behavior check for the default action abstraction under the
current code, rather than a reuse of the older result file.

## Current-cycle smoke result

| Dataset/split | Episodes | SR | SPL | nDTW | sDTW | NE | Path length | Primitive steps | High-level decisions | Collisions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| R2R-CE val_seen | 1 | 1.000000 | 0.646941 | 0.833686 | 0.833686 | 1.157351 | 7.042108 | 59 | 6 | 0.000000 |

The smoke output is numerically identical to the existing one-episode
`smoke_r2r` result for the standard metrics. The high-level decision count is
now recorded by the evaluator as 6 for this run; the older smoke artifact did
not record that field.

## Reproduction discrepancy and repair

The first smoke attempt failed before evaluation because a released IL
checkpoint's optimizer state did not contain the newly added optional critic
and residual-head parameter groups. This is a checkpoint compatibility issue,
not a metric discrepancy. In baseline mode the optional heads are now frozen,
and incompatible optimizer state is skipped with a warning; model state loading
remains strict=False as required for the released checkpoint. A second smoke
run then completed and matched the prior metrics.

## Baseline conclusion

The ETPNav baseline is reproducible on the local legacy environment. Full
R2R/RxR reference metrics are available, and a current-code smoke evaluation
passes. The next comparison can therefore vary only waypoint proposal density
while keeping the navigator, graph, controller, checkpoint, split, success
definition and simulator fixed.


## 2026-10-07 measurement-validity follow-up

The full baseline/fixed navigation metrics remain unchanged. The new ranker
validity audit operates offline and makes no core policy changes. Its small
unseen extraction covers one scene only; do not confuse it with the full
1,839-episode unseen baseline. The matched seen64 default ghost count is
21.53125 (the full unseen mean is 23.8499). Candidate-probe and full-controller
outcomes differ in contract; consult `RANKER_VALIDITY_AUDIT.md` before treating
selection/execution proxy gaps as causal failure rates.

## 2026-10-07 full-option tracing controls

Two paired evaluations explicitly test noninterference of the opt-in full-option
tracer, using the released checkpoint, one RTX 4090/worker, seed 100 and unchanged
sensors/controller/metrics. Effective tryout=false because sliding=true.

| Sample | SR | SPL | nDTW | Traced versus untraced |
| --- | ---: | ---: | ---: | --- |
| val_seen64 / 4 scenes | 79.6875% | 70.1608% | 73.6074% | all 1,152 episode metrics exact |
| balanced val_unseen66 / 11 scenes | 66.6667% | 54.5468% | 60.5601% | all 1,188 episode metrics exact |

All 18 aggregate metrics per sample match exactly too. These are baseline
controls, not improved agents. The unseen66 subset was selected from unchanged
val_unseen data using six distinct routes per scene and one instruction per route;
its score must not replace or be compared directly to the full benchmark score.
The source/selection manifest and paired result copies are under
`research/results/option_calibration/`. Full replay results and limitations are
in `OPTION_CALIBRATION_REPORT.md`; baseline inference latency is not estimated
from instrumented runs.

## Full graph-option capture control

The new optional capture trainer passes exact uninstrumented controls on train8
(144 episode metrics) and the frozen unseen66 subset (1,188 episode metrics).
Unseen capture results also match the previously archived pre-hook baseline
file byte-for-byte. All original sensors, checkpoint, masks and controller
settings remain in the recorded protocol. See `GRAPH_OPTION_REPORT.md` and
`results/graph_option_unseen/provenance/historical_baseline_check.json`.
## Expanded training control and native-path reconstruction (2026-10-07)

`GRAPH-OPTION-TRAIN64-001` samples 64 new training routes over 16 scenes by
metadata hashing, excluding pilot routes and checking zero route overlap with
both validation splits. Released checkpoint, simulator seed 100, sensors,
native controller, sliding, STOP gate and high-level horizon are unchanged.

Traced capture and the separately completed untraced control match all **1,152
per-episode metrics exactly**, plus all 18 aggregates. Native training-subset
SR/SPL/nDTW/SDTW = **85.9375 / 81.1023 / 83.1456 / 76.4019%**; mean goal distance
1.8534 m, path length 9.1224 m, 7.40625 high-level decisions and 57.265625 recorded
primitive steps. This is a preselected diagnostic training subset, not a new
method score or standard validation result. Results and source hashes are in
`results/graph_option_train64/noninterference_001/` and the capture/control
manifests; the sequential driver archives separate stdout/stderr for each stage.

The ordered-route audit independently reconstructs full original paths from
saved primitive poses. On pilot8 plus unseen66, native float32 path length,
fastdtw nDTW and SDTW match exactly across all 222 comparisons. The first unseen
attempt exposed a float64-vs-float32 reconstruction discrepancy in path length;
correcting arithmetic restored exact equality without changing tolerances or
the evaluator. See `ROUTE_COMPATIBILITY_REPORT.md` for preserved attempts.

Expanded training64 now also passes native path reconstruction: all 192 added
path-length/nDTW/SDTW checks are exactly equal. Total across pilot8, unseen66 and
training64 is 414 comparisons / 138 episodes. Full graph-option probe and
independent integrity checks also complete; no evaluation settings change.

## Single-intervention control and independent metric check (2026-10-07)

`GRAPH-SINGLE-INTERVENTION-TRAIN7-001` first runs the new research interceptor
disabled on the same 64 training routes in their original order. All 474 full
trace records, 1,152 per-episode metrics and 18 aggregates match the archived
capture exactly. The seven scheduled states are encountered with zero action
replacements. This verifies the new hook; it is not a baseline score change.

The subsequent enabled oracle experiment changes exactly seven actions. All
57 unaffected routes retain their 1,026 episode metrics exactly, and all 445
unchanged traces (unaffected episodes plus pre-intervention prefixes) match.
The released checkpoint, native sensors/controller, seed 100, success distance
3 m, native STOP and 15-decision horizon are preserved. No evaluation threshold
or horizon is adjusted to favor an intervention.

Independent reconstruction from the enabled run's retained primitive paths
and native goal-distance records matches **384/384 metrics exactly** across
64 episodes: goal distance, success, SPL, path length, nDTW and SDTW. It uses the
native float32 path arithmetic, dense reference path and fastdtw. This validates
the reported intervention metrics without treating instrumentation time as
baseline inference latency. Accepted controls and reconstructions are under
`results/single_intervention_train7/{disabled_audit_001,enabled_audit_001,native_reconstruction_001}/`.
Paired outcome interpretation belongs in `SINGLE_INTERVENTION_REPORT.md`;
privileged training-subset scores must not replace the standard baseline.

## Prospective continuation sample control (2026-10-07)

`CONTINUATION-LABEL-TRAIN64-001` uses 64 metadata-selected training routes in
16 scenes, excluding the 19 scenes inspected by earlier training diagnostics.
Validation route overlap is zero; these scenes are new to the diagnosis, not
held out from released-checkpoint training. Baseline SR/SPL/nDTW/SDTW =
87.5000 / 81.4706 / 81.5775 / 74.6881%. Mean final goal distance 1.6619 m,
path length 9.8834 m, primitive count 61.5, high-level decisions 7.328125.

Capture/untraced control matches 1,152 episode metrics and 18 aggregates exactly.
Disabled one-action hooks match all 469 complete traces before each enabled
schedule. The five-event H=1 run leaves 59 routes / 1,062 metrics unchanged;
the two-event H=2-confirmed run leaves 62 routes / 1,116 metrics unchanged.
Independent final-metric reconstruction passes 384 exact comparisons for each
enabled run. A separate full-trace comparison verifies all 469 H=2 mixed-run
records against the earlier offline composition. Released checkpoint hash,
seed 100, sensors, controller, sliding, STOP, 3 m success distance and horizon 15
are unchanged. See `CONTINUATION_LABEL_REPORT.md` for intervention outcomes;
they do not replace this control baseline or the standard validation scores.
