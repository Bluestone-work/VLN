# Dense Action OOD Next Protocol

Registered after `DENSE-OOD-CALIBRATION-AUDIT-001`, 2026-10-09.

## Purpose

Test whether A1 harm is caused by GraphMap/SAP representation contamination,
new-action OOD ranking, or both. This is an instrumentation and causal
comparison, not a new selector.

## Required arms

* A0: released native candidates, native graph encoder and SAP argmax.
* A1: current A0-union-dense candidates, native graph encoder and SAP argmax.
* A2: same A1 candidate set and unchanged controller, but compute the native
  action scores from an isolated native subgraph and compute dense-action scores
  in a separately marked branch. A2 must preserve native action identity and
  must not feed dense nodes back into native-node message passing.

A2 is opt-in and diagnostic. Baseline default behavior is unchanged when its
flag is disabled.

## Measurements

At every matched prefix before the first action divergence, record native-node
embedding and logit deltas between A0/A1, dense-node score distributions,
native-vs-dense score percentile, graph attention/message-passing input sizes,
selected action identity and calibration summaries. Do not read or use final
outcome to define a feature or threshold.

The primary contamination statistic is the native action embedding L2 shift and
native logit shift from A0 to A1. The primary OOD statistic is dense action score
percentile relative to the A0 native score distribution at the same state.

## Fidelity gates

Disabled A2 must reproduce A0 per-episode metrics and all matched prefix action,
sensor, RNG and STOP sentinels. A1/A2 candidate nesting must hold. The same
checkpoint, seed, controller, evaluator and decision budget are mandatory.

## Decision rule

If A2 materially reduces native embedding/logit shift while avoiding the A1
native-success destruction, representation contamination is the first target.
If native scores remain stable under A2 but dense score percentiles are
miscalibrated, run a separately frozen OOD calibration diagnostic. If both
remain problematic, stop dense expansion. Do not introduce g3D-LF until A2
shows that native representation isolation is insufficient and a future-semantic
question is explicitly registered.
