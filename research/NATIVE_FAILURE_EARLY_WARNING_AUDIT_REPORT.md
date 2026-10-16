# Native failure early-warning audit

HEAD: `4364ddf`.

This was a development probe for a different question from waypoint
selection: can pre-action native state predict whether the current route will
eventually fail? It did not modify ETPNav or run an intervention.

## Corrected data flow

The first implementation accidentally filled success-cohort pre-action counters
with zero. That was fixed by rebuilding both success and failure features from
their original graph and trace captures. The corrected source contains:

- 369 states from 44 native-success episodes;
- 71 states from 22 native-failure episodes;
- 440 states, 66 episodes and 11 scenes in total.

Features are pre-action graph/controller fields: high-level step, graph size,
visited/ghost counts, candidate count, native selected logit and margin,
policy entropy, STOP flag, native back-path length, pre-action primitive count
and pre-action collision count. Final route success is the label only.

## Observed development metrics

| Window | AUROC | AUPRC | Failure precision | Failure recall |
| --- | ---: | ---: | ---: | ---: |
| All captured states | 0.9615 | 0.9890 | 0.9721 | 0.9431 |
| High-level step ≤ 3 | 0.9285 | 0.9937 | 0.9830 | 0.9830 |

These numbers are **not valid evidence of an early-warning capability**. The
failure cohort was selected post hoc from critical states after native failure
had already been observed. Its mean high-level step is 7.92 versus 4.13 for
the success cohort. Only 12 failure states occur at step ≤3, so the apparent
early separation is supported by a very small, selected subset. The cohort
also does not contain all pre-action states from the failure episodes.

## Decision

**INVALID FOR MODEL OR DEPLOYMENT CLAIMS.** The high AUROC is a sampling
artifact until a fresh metadata-selected route cohort is run end to end and
all native states are retained before reading final outcomes. No risk gate,
threshold, intervention or shadow deployment is justified by this probe.

The next valid experiment is a registered fresh native baseline capture with
scene-disjoint route groups, including both successes and failures without
critical-state filtering. Feature extraction must be frozen before outcomes
are read; evaluation should report route-level calibration and early-step
recall at fixed false-alarm rates. If a warning signal survives that design,
it can be evaluated as an abstention-only shadow branch. If it does not, the
failure-recovery direction should be closed.

Artifacts:

- `research/tools/probe_native_failure_early_warning.py`
- `research/results/native_failure_early_warning_probe_002/summary.json`
- `research/results/native_failure_early_warning_probe_002/rows.jsonl`
