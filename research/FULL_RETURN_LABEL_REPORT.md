# Outcome-blind full-return graph decision study

Date: 2026-10-07  
Experiment: `FULL-RETURN-LABEL-TRAIN16-001`  
Decision: **NO-GO for fitting an abstraction or graph-action selector on this pilot**

## Question

Do outcome-blind alternatives at the same native graph state produce enough
reliable, full-episode variation to justify learning a deployable action
abstraction or graph-choice controller?

## Protocol and controls

The frozen sample contains 16 R2R-CE training routes from eight scenes that had
not appeared in the earlier diagnostics, with no validation-route overlap. One
non-STOP state was selected per route from metadata, masks and native graph
logits. Goal distance, reference route, branch results and final metrics were
excluded from selection. Two distinct valid alternatives were frozen at each
state: the highest nonselected native logit and a seeded graph-ID alternative.

Both alternatives were run to the native STOP or unchanged 15-decision limit.
Checkpoint `data/logs/checkpoints/release_r2r/ckpt.iter12000.pth`, seed 100, sensors, controller, STOP logic,
simulator and evaluator were kept fixed. Disabled controls, prefix/action/RNG
checks, independent branch integrity and native final metric reconstruction all
passed. There were 132 captured decisions, 1,429 forward branches and 305
reverse checks. Each enabled schedule reconstructed 96 final metric values with
zero discrepancy.

The launch environment intentionally preserved unset `EGL_PLATFORM` and
`CUDA_VISIBLE_DEVICES`, as recorded in the capture manifest. Earlier failed
attempts remain archived: an environment-mismatched attempt was rejected by the
provenance gate, and a directory-tag collision stopped attempt 002 before any
enabled run. Neither is used as evidence.

## Full-return result

Dominance requires no degradation in success, SPL, nDTW, final goal error, path
length or primitive count, with at least one strict improvement. No scalar
reward was fitted.

| Alternative | Dominates | Mixed | Dominated | Rescue / loss | Mean SPL change | Mean nDTW change | Mean primitive change |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| highest nonselected logit | 1 | 10 | 5 | 1 / 1 | -15.23 pp | -12.08 pp | +15.5 |
| seeded graph-ID | 0 | 8 | 8 | 1 / 2 | -28.77 pp | -24.27 pp | +63.0 |

The single dominating event is route `2508` in scene `mJXqzFtmKg4`. It rescues a
baseline failure and uses two fewer primitive actions. This is one scene and one
route. The top-logit alternative rescues that route, but the seeded alternative
at the same state is mixed and uses 107 more primitives. Thus the state alone
does not identify a safe alternative.

The top-logit schedule's scene-cluster bootstrap intervals were [-30.13, +1.61]
SPL points, [-24.57, -1.35] nDTW points, and [+2.25, +28.50] primitives. The
seeded schedule intervals were [-46.93, -12.29], [-35.93, -13.24], and
[+37.81, +88.56], respectively. These intervals describe this eight-scene
diagnostic sample; they are not a benchmark confidence claim.

Short windows would have missed the only dominating alternative: after H=1 and
H=2 it is 1.0190 m and 1.4952 m farther from the goal than the baseline at the
matched windows. Both agents are still making absolute progress. The alternative
then stops at 0.1669 m; the baseline reaches 0.1669 m first, continues away and
stops at 4.1721 m. This is further evidence against the current short-window
label rule, not proof that every short-horizon predictor must fail.

## Failure observations

The changed candidate often altered later graph choices and stopping cost. For
example, the seeded alternative causes large detours on routes `6235`, `7537`
and `8891`, including a 36-primitive terminal return on `8891`. The baseline has one endpoint-neighborhood-visit-then-failure (2508); neither
alternative schedule has that symptom. Baseline failure 6235 never has an
endpoint inside 3 m. Endpoint visits alone cannot identify a STOP cause. These are descriptive
symptoms, not privileged causal labels.

These cases motivate inspecting graph selection, continuation and STOP jointly;
they do not establish a dominant causal failure category. It does not
show that changing waypoint proposal granularity is the missing capability.

## Decision and next gate

**NO-GO:** do not train an abstraction policy, ranker, RL objective or VLM
feature from this pilot. Support is one dominating scene, below the registered
six-scene gate; both candidate schedules reduce aggregate SPL/nDTW and increase
primitive cost. This does not claim that every adaptive action representation is
impossible. It rejects the current evidence base as sufficient justification.

The next useful research action is a descriptive census of terminal/recovery
symptoms on already calibrated baseline traces, with training and validation
cohorts kept separate. It must distinguish arrival-then-departure, never-arrived
STOP, and forced historical-node return before proposing a new intervention.
This is a diagnostic pivot, not authorization to bypass the failed learning gate
by sampling again. No model is trained in this cycle.

Artifacts: `full_return_analysis_001/summary.json`,
`full_return_analysis_001/paired_table.md`, `descriptive_002/summary.json`,
and `descriptive_002/full_return_diagnostics.png`, and `research/results/terminal_recovery_census_001/summary.json`.


## Scope, reproducibility and implementation record

These are graph-choice interventions within the **same default abstraction**,
not adaptive coarse/default/fine switching. Candidate selection is deployable in
its inputs; final-return analysis, full-option probes and labels use privileged
information. Two candidates at one state are neither exhaustive search nor an
upper bound. Thirty-one non-dominating events are not thirty-one useless actions:
18 are genuine mixed tradeoffs. The strict six-metric gate may reject reasonable
preferences, but we do not change it after seeing these outcomes. Eight training
scenes / one seed cannot establish held-out benefit. No new novelty claim is made;
see `LITERATURE_METHOD_AUDIT_20261007.md` for known overlap and outstanding checks.

Git: `1c1a794148a774940d59a12dc601ea51febf3d7e` (pre-existing tracked dirty tree).
Checkpoint SHA256: `4e70d2a1b4a6cfb32158901b7300cbc2be657641f16c9a17f35ca8570470d8b3`.
One worker, simulator/Torch GPU 0 on an RTX 4090; Python 3.6 legacy environment.
State/sample seed 20261007, second-alternative seed 20261008, simulator seed 100.
No latency comparison is made across the environment difference from prior cycles.
The frozen `FULL_RETURN_TRAIN16_PROTOCOL.md`, config, schedules and earlier
`FULL_RETURN_LABEL_NEXT_GATE.md` remain immutable; their design is now completed.

Research-only fixes: sample asset validation changed `.glb` is_dir to is_file;
state hash gained its missing step field and second-candidate duplication was
removed before schedules were frozen; intervention runner now accepts the exact
configured positive population instead of hard-coded 64; secondary graph analysis
falls back to the existing 0.25 m material-gain threshold when absent in config.
Driver output tags now apply to every directory. Attempts 001/002 retain their
original running-looking status files and error logs; they are failed attempts,
not live jobs. Attempt 003 is the accepted complete continuation.

Descriptive follow-up attempt 001 failed before creating results because raw
trace post_metrics does not include the native success key. Attempt 002 reads
native success from the audited episode table; its source and predecessor are
archived. The full-return classifier rejects missing/nonfinite metrics and
noninteger primitive deltas. Four frozen selection tests, thirteen interceptor /
audit tests, and five full-return tests pass (22 total). Source compilation and
git diff whitespace checks pass. No baseline core source was edited in this cycle.
