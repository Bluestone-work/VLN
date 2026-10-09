# Dense Action A2 Representation-Isolation Smoke

Date: 2026-10-09

## Implementation

The opt-in `ACTION_ABSTRACTION.DENSE_ISOLATION_MODE` supports `off`,
`capture`, and `a2`. The default remains `off`. In dense mode, GraphMap emits a
native mask that keeps STOP, real nodes and native ghosts while excluding newly
inserted dense ghosts. The navigation model runs a second graph encoder with
that mask. `capture` records isolated native embeddings/logits while retaining
the A1 action path; `a2` replaces native action logits with isolated values and
keeps full-graph values for dense actions.

## Smoke population and checks

The same first val-unseen episode (`1778`) was run with one RTX 4090, seed 100,
released checkpoint, native controller, STOP and evaluator. A0 produced four
decisions and exactly matched the archived replication action sequence
`[4, 6, 12, 0]`. A0 capture contains no isolation fields. This is the minimal
baseline non-interference check for the code change.

The A1 `capture` arm retained the dense action path and produced 12/12 rows with
isolated fields. At those states the isolated native logits differed from the
full A1 logits by mean absolute `1.92698` and maximum `5.60102`; 255 dense or
masked entries were serialized as JSON `null`, while all native finite values
were retained. This confirms the isolation branch is active and the observed
contamination is measurable online, but one episode is not a policy result.

The A2 action arm also completed one episode and produced 12/12 isolated rows.
It ended with SR 0 on this single route, while A0 ended with SR 1; this is a
smoke-path observation only and must not be treated as an intervention estimate.

## Decision

**Instrumentation smoke PASS; scientific gate remains open.** The baseline
default is unchanged, A1 capture can measure representation isolation, and A2
can execute the isolated-native/full-dense composition. Do not interpret the
single-route A2 result as evidence of improvement or harm. The next experiment
is a pre-registered, scene-balanced paired A0/A1-capture/A2 rollout with native
success controls and the existing candidate-nesting, sensor, RNG and metric
reconstruction audits. g3D-LF remains deferred until that representation
control is evaluated.

Artifacts:

* `research/configs/DENSE_A0_SMOKE_001.json`
* `research/configs/DENSE_A1_ISOLATION_CAPTURE_SMOKE_001.json`
* `research/configs/DENSE_A2_SMOKE_001.json`
* `research/results/dense_a0_smoke_001/run/`
* `research/results/dense_a1_isolation_capture_smoke_001/run/`
* `research/results/dense_a2_smoke_001/run3/`
