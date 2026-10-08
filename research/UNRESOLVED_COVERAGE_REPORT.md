# Unresolved failure coverage census

Date: 2026-10-08  
Experiment: `UNRESOLVED-COVERAGE-CENSUS-001`  
Status: post-hoc diagnostic; no rollout, fitting, or policy change.

## Question

When exhaustive native graph-action returns fail to rescue a route, is the immediate explanation that no usable proposal/action exists?

## Data and method

The census reuses three already completed exhaustive full-return cohorts:

| Cohort | Unresolved routes | Critical states | Full native action returns |
| --- | ---: | ---: | ---: |
| train64 | 3 | 7 | 112 |
| unseen66 diagnostic | 14 | 42 | 622 |
| fresh train64 | 1 | 7 | 175 |
| **Total** | **18** | **56** | **909** |

For every unresolved route, the analysis counts valid native action branches, non-STOP branches, distinct ghost targets at each critical state, native primitive collision events, and forced STOP. These are action-availability and symptom measurements. They do not use a reference route to label a proposal as correct, and they do not identify proposal failure causally.

## Result

Every unresolved route had multiple available native non-STOP branches:

| Statistic over 18 unresolved routes | Value |
| --- | ---: |
| routes with at least one non-STOP branch | **18/18** |
| minimum number of non-STOP branches on a route | **11** |
| maximum number of distinct ghost targets at one critical state | **8–29** depending on route |
| routes with native forced STOP | 4/18 |
| routes with at least two native primitive collision events | 7/18 |

The route-level records are in `results/unresolved_coverage_census_001/summary.json`.

## Interpretation

This rejects the narrow proposal-empty hypothesis for the unresolved cohort. The failures are not caused by an empty action list: the native graph exposes many alternatives. The result does **not** prove proposal coverage is perfect. A correct direction could still be missing because the waypoint predictor, NMS, ghost merging, or action horizon omits it. It also does not separate ranking, execution, recovery, and termination.

The unresolved routes are heterogeneous. Some are collision-heavy or budget-limited, while others have few collisions and terminate early without a successful alternative. Therefore a single trigger or a single proposal-density change is unlikely to explain all failures.

## Decision

- Do not modify the waypoint predictor based on this census.
- Do not label unresolved routes as proposal failures.
- Keep proposal coverage as an open hypothesis that requires a denser or independently generated candidate set at matched states.
- The leading deployable problem remains native graph-action selection with long-horizon continuation value and safe abstention.
- Termination/recovery should be analyzed separately on the small subset with forced STOP or interior near-goal arrival.

No model was trained and no evaluation benchmark was changed.
