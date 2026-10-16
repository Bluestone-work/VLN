# Native failure opportunity census

HEAD: `d1c4057`.

This report audits an existing exact full-return development cohort. The cohort
was selected after native failures and critical states had already been found,
so it is **outcome-enriched** and is not an unbiased validation benchmark. It
is used only to decide whether native failures contain causal waypoint rescue
opportunities.

## Source and label audit

Source cases:
`research/results/critical_causal_unseen/full_001/cases/`.

- 22 native-failure episodes;
- 71 critical decision states;
- 985 exact single-action full-return branches;
- 10 scenes;
- every baseline state has `success = 0`;
- all branches passed the existing full-return validity checks.

For each state, every branch is compared with the native continuation using the
fixed lexicographic order: success, nDTW, SDTW, shorter path, fewer primitive
steps, and fewer collisions. No feature is fitted and no model is trained.

## Opportunity result

| Quantity | Count |
| --- | ---: |
| States with at least one BENEFIT alternative | 50 / 71 (70.4%) |
| States with no BENEFIT alternative | 21 / 71 (29.6%) |
| States with a success-rescuing alternative | 18 / 71 (25.4%) |
| States with quality-only improvement | 32 / 71 |
| Episodes with any BENEFIT state | 20 / 22 |
| Episodes with a success rescue | 8 / 22 |
| BENEFIT branches | 275 / 985 |
| HARM branches | 639 / 985 |
| Equal-to-native branches | 71 / 985 |

The 18 success-rescue states are distributed across eight episodes. The
opportunity is therefore not confined to one route, although the cohort is too
selected to support a benchmark claim. One rescue state is a native STOP state;
the remaining rescue opportunities are non-STOP waypoint choices.

## Interpretation

This census rejects the explanation that native failures are mostly caused by
the absence of a better admissible waypoint. In this development cohort,
better candidates are present in roughly seven of ten critical states, and
success-rescuing candidates occur in eight episodes.

It also does not validate a semantic waypoint method. The earlier RGB-D,
semantic, endpoint-latent, graph/history and temporal-history probes were run
on the native-success cohort or scene-disjoint diagnostic splits and did not
provide stable candidate-level residual signal. The present result says that
there is causal room for improvement, while the useful signal remains
unidentified by the tested representations.

## Decision and next experiment

**GO for a failure-focused diagnostic; NO-GO for online intervention.**

The next experiment should collect exact full-return pairs on a fresh,
pre-registered failure-enriched cohort with the same native checkpoint and
controller, while keeping a separate outcome-blind holdout. It should test a
small **safe abstention / recovery gate** against the following pre-action
signals:

- native action margin and STOP uncertainty;
- collision and deviation history already available before the action;
- graph proposal age and backtracking status;
- instruction-progress state from the existing language/trajectory context.

The gate may abstain and leave the native action unchanged. It must not invent
new waypoints, alter the Graph Transformer, or use endpoint observations. A
fresh scene-disjoint cohort is required before any claim of rescue or any
online shadow deployment.

Artifacts:

- `research/tools/analyze_native_failure_opportunity.py`
- `research/results/native_failure_opportunity_census_003/summary.json`
- `research/results/native_failure_opportunity_census_003/states.jsonl`
