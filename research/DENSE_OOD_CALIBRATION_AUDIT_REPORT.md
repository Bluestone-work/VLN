# Dense Action OOD / Calibration Audit

Date: 2026-10-09  
Experiment: `DENSE-OOD-CALIBRATION-AUDIT-001`

## Question

Why does A0-union-dense A1 rescue some unresolved routes while destroying native
successes? The audit separates native representation contamination from a pure
new-action ranking/calibration failure.

## Frozen comparison

The accepted 99-route A0/A1 native-policy replication was reused read-only. It
contains 55 train controls and 44 val-unseen controls, with the original
checkpoint, seed, controller, STOP rule and 15-decision budget. For each route,
only matched prefixes before the first selected-action divergence were used.
No continuation outcome, goal/reference geometry or label was used in the
comparison. A0 and A1 graph captures were joined by episode and high-level
step; native options were joined by rounded action target identity.

## Results

| Quantity | All controls | Train | Val-unseen |
|---|---:|---:|---:|
| Matched prefix states | 154 | 82 | 72 |
| States with selected-action divergence | 97 | 54 | 43 |
| Native option comparisons | 915 | 482 | 433 |
| Mean absolute native logit shift | 0.5993 | 0.5920 | 0.6075 |
| Mean native embedding L2 shift | 2.6256 | 2.6274 | 2.6237 |
| Max native embedding L2 shift | 16.7613 | — | — |

The native representation is therefore not invariant to adding dense actions.
At the first divergence, native action embeddings and logits have already
changed substantially in the A1 graph. This directly supports H1
`Representation Contamination`; a comparison based only on whether the new
dense action wins would miss this mechanism.

The shift is directionally larger on destroyed native-success routes than on
retained routes: mean native embedding L2 shift `2.7747` versus `2.5985`, and
mean absolute native logit shift `0.6348` versus `0.5929`. This is descriptive,
because the destroyed group has only 18 routes and matched states are weighted
by decisions, but it is consistent with contamination contributing to harm.

The effect is present in both train and val-unseen prefixes, so it is not a
single-scene artifact. It also means H2 is not an adequate sole explanation:
the dense candidates are not merely miscalibrated additions to an unchanged
native score distribution. A1 changes the native score function itself while
also exposing OOD dense actions.

## Decision

**NO-GO for unconditional dense expansion remains confirmed.** The next
technical priority is an A2 representation-isolation control: expose the same
dense action set but compute native action embeddings/logits from the native
subgraph, then score dense actions in a separately identified branch. This must
be an opt-in diagnostic with A0 non-interference and matched prefix audits; it
must not change the released baseline or train a selector.

Only after A2 establishes whether native isolation removes the observed harm
should a dense-action OOD calibration study be attempted. g3D-LF is deferred:
the current evidence identifies a graph representation confound before it
establishes missing future semantics.

## Artifacts

* `research/tools/analyze_dense_ood_calibration.py`
* `research/results/dense_ood_calibration_audit_002/summary.json`
* `research/results/dense_ood_calibration_audit_002/matched_states.jsonl`
