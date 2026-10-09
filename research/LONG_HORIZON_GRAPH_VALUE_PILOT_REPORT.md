# Full-return Graph Preference Pilot

Date: 2026-10-09  
Experiment: `LONG-HORIZON-GRAPH-VALUE-PILOT-001`

## Scope

This pilot checks the native action contract and serialization boundary before
any new rollout or model fitting. It reads the accepted 16-route capture under
`research/results/full_return_label_train16/`; that capture is a historical
training diagnostic, not held-out validation. No outcome was used to select the
eight pilot states, and no processed label is used by the pilot.

## Checks

The pilot inspected eight states and 83 admissible native actions: 23 current
ghost actions, 52 historical ghost actions and eight STOP actions. Every
admissible option had exactly one forward branch trace, with unique native
identity `(index, graph_id, action_type, current_proposal)`. Reverse calibration
records were excluded from the forward uniqueness map because they intentionally
repeat the same action in a fresh worker.

The source capture reports 132/132 selected-action sentinels, 305/305 reverse
order checks, zero endpoint error, 132/132 and 305/305 sensor matches, complete
admissible-option coverage, and baseline non-interference. Its independent
integrity audit reports 1,734 branches (1,429 forward, 305 reverse), exact
action/RNG identity, maximum start position error 0 m, maximum goal-distance
sentinel error 0 m, 164 STOP branches and 54,544 primitive events.

The existing disabled control independently compares 288 per-episode metrics
and 18 aggregates with maximum absolute delta 0.0. These are fidelity checks on
the existing capture, not a new benchmark run.

## Serialization and leakage

`research/tools/graph_value_pilot.py` writes `pilot_dataset.jsonl` with one
state row, the complete action set, stable identity, deployable feature view,
raw option dictionary and raw native branch trace. It rejects duplicate native
identity, missing forward branches, mismatched episode identity and forbidden
feature keys. The generated manifest records source hashes and reports zero
deployable-feature leakage. A separate check parsed 1,018 full-return result
rows and verified that each has raw `metrics` and `baseline_metrics`; those
rows are not joined to pilot states because their case keys are from a different
critical census.

The pilot intentionally retains immediate branch fields such as `progress_m`
only inside `raw_option`/`raw_native_branch_trace`; they are not in the
deployable feature view and are not labels. Full-return final metrics are not
invented when a branch capture does not contain them.

## Decision

**Pilot PASS for native identity, state restoration sentinels, action coverage,
raw-trace retention and feature leakage checks.** This does not pass the
scientific feasibility gate. The source population is historical and training
only, the pilot has no scene-disjoint validation, and it does not yet produce
the required outcome-blind full-return preference dataset. Therefore no B0/B1/B2
training or online intervention is started in this cycle.

The next execution must use a newly frozen scene-balanced route sample, retain
complete continuation outcomes for every admissible native action, and freeze
the protocol before reading those outcomes. Existing in-cohort feasibility
numbers remain descriptive and cannot be promoted to a GO claim.

## Artifacts

* `research/tools/graph_value_pilot.py`
* `research/configs/LONG_HORIZON_GRAPH_VALUE_PILOT_001.json`
* `research/results/long_horizon_graph_value_pilot_003/manifest.json`
* `research/results/long_horizon_graph_value_pilot_003/pilot_dataset.jsonl`
