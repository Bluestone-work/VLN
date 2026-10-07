# Native STOP feasibility and primitive-arrival diagnostic

Date: 2026-10-07  
Status: post-hoc diagnostic; no policy or STOP rule changed

## Question

When a baseline route fails, is there evidence that the unchanged native STOP
action could have succeeded at an earlier graph decision? Separately, can an
episode enter the 3 m success neighborhood during a low-level controller option
without any recorded high-level decision endpoint entering it?

## Protocol

Five previously calibrated cohorts were kept separate: train pilot 8, train64,
continuation train64, full-return train16 and unseen66. Every accepted forward
index-0 branch was checked against the same decision prefix, action dictionary,
pose, RNG and native STOP primitive. Native success uses the evaluator's
inclusive `distance <= 3 m` operator. Existing integrity, probe and baseline
noninterference gates passed before counting. No rollout, threshold search or
model fitting occurred.

For primitive localization, recorded primitive poses were evaluated with the
original scene navmesh and dataset goal. All 84 pre/post endpoint comparisons
reconstructed the native distances exactly (maximum error 0 m). This uses
privileged goal/navmesh information for diagnosis only.

## Results

| Cohort | Episodes | Failures | Failed routes with successful native STOP branch | Failed routes arriving only between decision boundaries |
| --- | ---: | ---: | ---: | ---: |
| train64 | 64 | 9 | 0 | 0 |
| continuation train64 | 64 | 8 | 2 | 0 |
| full-return train16 | 16 | 2 | 1 | 0 |
| unseen66 | 66 | 22 | 1 | 1 |

The train pilot has no failures. The two continuation routes are `10587` and
`393`; both enter the success neighborhood during an option and an earlier
native STOP branch would finish within 3 m. In full-return route `2508`, the
first primitive arrival occurs at episode primitive 26, while the baseline
continues and eventually stops at 4.1721 m. In unseen route `1593`, the first
interior arrival occurs at primitive 48 but no recorded or counterfactual native
STOP endpoint succeeds. Route `73` reaches the neighborhood at an option endpoint and has multiple
successful native STOP branches.

The route-level result is therefore mixed: native STOP timing can rescue a few
failures, but it cannot rescue nine train64 failures and 21 of the 22 unseen66 failures
under the tested native STOP actions. Of the 22 unseen failures, 20 never reach
the neighborhood according to native oracle_success; one reaches it only between
decisions (1593), and one has successful earlier native STOP branches (73). Interior arrival is
also not equivalent to a deployable STOP signal because the agent does not have
privileged geodesic goal distance at runtime.

## Decision

**NO-GO for developing a STOP-only model or reviving the AAA contribution from this sample.** STOP and
low-level execution should remain in the failure decomposition, but the evidence
does not justify replacing graph selection or waypoint abstraction with a STOP
model. A counterfactual opportunity is insufficient to identify a deployable trigger.
No resampling for more positive cases or model fitting is justified by these
results. The next gate, if this mechanism is pursued, is a novelty/observability
audit of interruptible action execution; adaptive trajectory truncation already
appears in Beyond Waypoints. No claim of a new method is made.

Machine-readable outputs:

- `research/results/native_stop_feasibility/analysis_001/summary.json`
- `research/results/native_stop_feasibility/primitive_arrival_001/summary.json`



## Proximity versus actual native STOP

In unseen66, nine near-goal decisions across eight routes would finish outside
3 m after executing their native STOP return; seventeen far-goal decisions across
three routes would finish inside 3 m. These are counterfactual states, including
baseline-successful routes, not additional failures. Thus current distance alone
cannot substitute for native STOP execution. The 1593 interior minimum is only
2.91524 m, compared with 3.10743 m at its nearest decision boundary and 3.20625 m
finally. Only three recorded primitive endpoints are inside 3 m. This marginal,
single-scene example is not broad support for changing action granularity.

All 1,738 baseline decisions across the five separately reported cohorts have an
accepted forward STOP branch. Existing raw branch hashes match their independent
integrity audits; selected terminal STOP reproduces baseline primitives exactly.
The native operator is <= 3 m, unlike the older endpoint census's strict < 3 m.
There are zero exact-3 m cases in the checked boundaries/STOP endpoints, so this
operator correction changes none of those observed classifications. The four
new unit tests specifically cover the inclusive boundary and false causal
attributions. Primitive localization uses dataset goals from the matching split
and scene, not a training GT route with a coincidentally equal episode ID.

The interior trace is available as `results/native_stop_feasibility/figures_001/interior_arrival.png` and PDF. This shows recorded poses, not a new rollout.
