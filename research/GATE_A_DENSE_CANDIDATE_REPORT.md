# Gate A: Dense-Candidate Full-Return Oracle

## Question

Does a denser candidate set contain useful actions that are absent from the
native ETPNav graph candidate set? This is a privileged opportunity audit, not a
deployable method evaluation.

## Frozen protocol

A0 is the unchanged native ETPNav graph candidate set. A1 is constructed from
the same frozen `[120, 12]` waypoint heatmap as `A0 union dense-NMS`, using
`max_predictions=12` and `sigma=(4,3)`. A0 uses the native `max_predictions=5`,
`sigma=(7,5)` pass. The union guarantees `A0 subset A1`; no route, goal,
reference path, or full-return outcome is read while generating candidates.
A2 was not run because reducing ghost merging changes persistent graph identity
and would no longer be a clean candidate-only comparison.

Every A1-new branch starts from the captured pose, graph prefix, simulator RNG,
checkpoint, controller, STOP logic, sensors, and 15-decision budget. It runs
the full native controller option and then the native navigator to termination.
The branch traces passed prefix, RNG, primitive, path, metric reconstruction, and
native sentinel gates. All results are marked `privileged_analysis_only`.

## Results

The combined audited cohorts contain 31 unresolved routes, 86 non-STOP critical
states, and 18 scenes. A0 contributes 1,093 native branches; A1 adds 667 new
branches. Mean candidate count rises from 12.71 to 20.47 per critical state.

| Full-return rescue ceiling | A0 native | A1 dense union |
| --- | ---: | ---: |
| strict dominance | 18/31 routes | 20/31 routes |
| quality rescue (failure to success, nDTW non-decrease) | 10/31 | 13/31 |
| cost-capped quality rescue | 4/31 | 6/31 |

A1 adds three quality-rescue routes in three scenes (`8343`, `1052`, `1584`)
and two cost-capped routes in two scenes. Ten routes already have an A0 rescue
that the native policy did not select. Eighteen routes still have no A1 quality
rescue. New-candidate primitive counts have a broad state-level range (32 to
374), so the opportunity is expensive and not uniformly reliable.

## Decision

**GO for proposal-coverage oracle opportunity, with a narrow scope.** The
additional rescue ceiling appears in multiple routes and scenes in both train
and held-out unseen cohorts. This does not authorize waypoint-predictor
training or claim a navigation-policy gain. The next question is whether an
execution-before-action signal can selectively identify native-relative
interventions; if that fails, proposal expansion should stop as well.

Artifacts:

- `research/results/gate_a_dense_candidate_combined/analysis_001/summary.json`
- `research/results/gate_a_dense_candidate_combined/analysis_001/per_route.csv`
- `research/results/gate_a_dense_candidate_combined/analysis_001/gate_a_rescue_ceiling.png`
- `research/results/gate_a_dense_candidate_002/` (train v2, graph-state-corrected)
- `research/results/gate_a_dense_candidate_unseen/` (unseen replication)
