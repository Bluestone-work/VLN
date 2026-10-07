# Single-action oracle intervention with native continuation

Date: 2026-10-07. Experiment: GRAPH-SINGLE-INTERVENTION-TRAIN7-001.
Status: **completed; all fidelity gates passed**. Original NMS-density AAA stays
NO-GO. A limited graph-decision efficiency study remains CONDITIONAL GO; direct
training on the current one-step labels is not justified.

## Question and frozen design

Can one locally better, no-more-expensive, reference-compatible graph action
improve eventual navigation when the original policy continues naturally?
The preceding cycle froze seven routes and one SHA256-selected state per route
in `configs/GRAPH_SINGLE_INTERVENTION_TRAIN7_001.json`. They were selected using
previous local oracle outcomes, not new continuation outcomes. All 64 original
training routes and their simulator order were retained. The other 57 are
non-intervention controls, not 57 extra independent treatment observations.

The hook changes only the scheduled non-STOP graph index before the original
builder updates the front node and consumes the selected ghost. Every subsequent
observation, graph update, action and STOP comes from actual native continuation.
No teleportation, baseline-suffix replay, extra observation, budget extension,
new encoder, learned scorer, RL or VLM is introduced. This is **privileged oracle
analysis**, never a deployable method. Checkpoint, simulator seed 100, sensors,
controller, success threshold and 15-decision horizon remain fixed.

The schedule file is immutable and retains its original `planned_not_executed`
field as a provenance snapshot. Actual execution status is recorded in the
result manifests, accepted audits and cycle summary.

## Fidelity gates

- Disabled interceptor: all **474 complete trace records**, all **1,152
  per-episode metrics** and 18 aggregate metrics equal the archived baseline.
- Enabled: exactly **seven** scheduled replacements, with graph IDs, logits,
  masks, typed positions and native actions checked before each event.
- All **445 unchanged full traces** (unaffected episodes and intervention
  prefixes) match exactly. All **1,026 per-episode metrics** from the other
  **57 routes** match exactly; none is changed or excluded.
- All seven executed alternatives match the prior isolated probes: endpoint
  error 0 m, primitive/collision sequences, RNG and observation hashes match.
- Original STOP gates and native ghost consumption pass throughout. Every
  episode ends under the original 15-decision horizon. Enabled run has 472
  high-level decisions versus 474 for the paired baseline.
- Thirteen focused interceptor/audit tests pass; compilation and whitespace
  checks pass. Existing tracked core source files are not modified in this cycle.
- Independent reconstruction from actual retained paths and native goal records
  reproduces **384/384 final metrics exactly** across all 64 routes: goal
  distance, success, SPL, path length, nDTW and SDTW. The unchanged native
  float32 path arithmetic, fastdtw and 3 m criterion are used.

## Paired final outcomes

| Metric | Targeted 7: baseline | Targeted 7: intervention | Paired delta |
| --- | ---: | ---: | ---: |
| SR (%) | 57.1429 | 57.1429 | +0.0000 |
| SPL (%) | 46.0484 | 51.6054 | +5.5570 |
| nDTW (%) | 63.7303 | 67.9689 | +4.2386 |
| SDTW (%) | 46.9931 | 51.6513 | +4.6582 |
| Final goal distance (m) | 3.4026 | 4.0180 | +0.6155 |
| Path length (m) | 15.2417 | 15.2167 | -0.0250 |
| Primitive actions | 96.2857 | 101.5714 | +5.2857 |
| High-level decisions | 10.0000 | 9.7143 | -0.2857 |

Percent-metric deltas are percentage points, not relative percent gains.
**No failed episode is rescued; no previously successful episode is lost.**
All four successful routes improve SPL and nDTW; among the three failed routes,
one improves nDTW while two worsen it. Five routes shorten their path and use
fewer primitives, but two regress sufficiently that total primitive count rises
from **674 to 711 (+37)**. Mean final goal error worsens by **0.6155 m**.
Mean total path length changes only -0.0250 m despite the SPL gain.

Across all 64 routes, SR remains **85.9375%**; SPL changes **81.1023 -> 81.7101%**
(+0.6078 pp), nDTW **83.1456 -> 83.6092%** (+0.4636 pp), and primitive count
57.2656 -> 57.8438 per episode. These are actual privileged intervention scores
on the selected training population, **not** a learned-model gain or validation
benchmark. Report the seven treated routes separately rather than diluting
failures with 57 unchanged controls.

## All seven cases

| Episode / step | SR before → after | SPL delta (pp) | nDTW delta (pp) | Path delta (m) | Primitive delta |
| --- | --- | ---: | ---: | ---: | ---: |
| 9045 / 7 | 0 → 0 | +0.000 | -14.798 | +2.250 | +17 |
| 8786 / 3 | 1 → 1 | +9.664 | +10.608 | -2.000 | -24 |
| 2408 / 6 | 1 → 1 | +11.942 | +13.443 | -1.500 | -6 |
| 8 / 2 | 1 → 1 | +13.073 | +4.372 | -1.750 | -11 |
| 10811 / 7 | 1 → 1 | +4.221 | +4.184 | -0.750 | -3 |
| 6962 / 10 | 0 → 0 | +0.000 | -1.228 | +3.825 | +65 |
| 875 / 6 | 0 → 0 | +0.000 | +13.089 | -0.250 | -1 |


Two concrete limitations of the one-step labels:

1. **Episode 9045 / step 7:** local gain +0.8181 m and immediate cost -2
   primitives / -0.7500 m. The alternative still initially moves away from the
   goal (-0.8197 m progress). Native continuation then moves farther away twice
   and stops. Final nDTW decreases 14.798 pp, path increases 2.2500 m and total
   primitive count rises by 17. The route fails in both runs.
2. **Episode 6962 / step 10:** local gain +3.5544 m, immediate cost -4 primitives
   and -0.1723 m. At step 11, the unchanged policy selects the displaced original
   ghost `g24`. At step 14, native forced STOP returns through two graph nodes
   to historical node `8`, costing 29 primitives versus one STOP primitive in
   the baseline. Final goal error changes **3.5230 -> 8.5082 m**, and the episode
   uses **65 more primitives**. This one tested state belongs to the route that
   dominated the training local-gain sum; the other states on that route have
   **not** all been intervention-tested and must not be labeled harmful by
   association.

Episode 875 improves nDTW by 13.089 pp and finishes 1.6729 m closer, but the
native policy still stops at 4.8004 m, outside the unchanged 3 m criterion.
This is a premature-stop symptom, not proof that overriding STOP would recover
success. No STOP intervention was performed.

## Post-hoc horizon observation

All seven immediate replacements satisfy both actual execution-cost caps.
Summed immediate primitive count drops by 17, yet complete episodes use 37
more primitives: later native continuation contributes the +54 difference.
One-option cost control does not establish episode-level cost control.

Examining H=1/2/3 high-level decision windows **after seeing outcomes**, the two
harmful routes already reverse their local goal advantage at H=2. This motivates
an independent test of short-continuation labels; it does not validate a new
labeling rule. Decision windows do not have equal primitive cost. STOP is treated
as absorbing only in this offline comparison, with no simulator/evaluator change.
The data and explicit post-hoc limitations are in `horizon_posthoc_001/`.

## Interpretation and decision

The local signal does not disappear completely: some trajectory-efficiency
improvements survive real native continuation. It is also **not a reliable
recovery label**: success stays 4/7, two failed routes deteriorate, and total
primitive cost rises. Do not advertise one-step cost bounds as full-episode
compute savings or use future success to filter training examples at deployment.

- **NO-GO:** original fixed-NMS adaptive selector remains unsupported.
- **CONDITIONAL GO:** narrow graph-decision / trajectory-efficiency diagnosis.
- **Do not train/integrate a new scorer on the current local labels yet.** Their
  sparsity, concentration and observed continuation reversals remain unresolved.

The next informative experiment is a prospectively frozen comparison of one-step
and short native-continuation labels on **new training routes**, with the same
candidate identities, STOP and total cost accounting. The H=2 observation here
is hypothesis-generating; do not tune on these seven outcomes and call it
independent confirmation. If these controls cannot predict useful continuations,
stop the label-distillation pivot rather than increasing model capacity. No
additional experiment or learned model is claimed complete in this report.

## Artifacts and scope

All files are under `results/single_intervention_train7/`:

- `disabled_001/`, `disabled_audit_001/`: no-intervention run and exact control.
- `enabled_001/`, `enabled_audit_001/`: native continuations, paired episode JSON,
  all execution gates and machine-readable aggregate results.
- `interpretation_001/`: complete seven-case table and best/worst nDTW trajectory
  plots (PNG/PDF), including original instructions. Walls are not rendered.
- `horizon_posthoc_001/`: explicit exploratory horizon and displaced-ghost audit.
- `native_reconstruction_001/`: independent final-metric reconstruction.
- `logs/`, `provenance/`: stdout/stderr, commands, environment/source hashes,
  frozen schedule, unchanged-core diff and tests.

Seven preselected training routes, five scenes, one simulator seed; no independent
learned-policy seeds, statistical-significance claim or causal six-way failure
partition. Direction of final goal distance and primitive cost must accompany
SPL/nDTW means. Full held-out R2R, novelty-code review and RxR transfer remain
outside this study.
