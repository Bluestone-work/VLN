# Ordered reference-route audit — 2026-10-07

Status: frozen unseen66, pilot8 and expanded training64 analyses complete.
Original NMS-density AAA remains NO-GO. No deployable
model, RL, VLM or navigation improvement is claimed.

## Question and measurement

Previous full-option probes found 0.148787 m mean local goal-progress opportunity
on 534 non-STOP unseen states when neither primitive count nor actual movement
path exceeds the selected action. Does that remain when respecting reference
route order? Protocol and tolerances were fixed before route outcomes in
`ROUTE_COMPATIBILITY_PROTOCOL.md` and `configs/ROUTE_COMPATIBILITY_001.json`.

Reconstruct the real baseline prefix from every original primitive pose.
Each alternative starts from that identical prefix and original controller
state. Use dense reference locations from the native evaluator and exact
Euclidean DTW for a separate ordered-prefix diagnostic. Keep a monotonic
baseline cursor. An alternative must match at least the selected reference
endpoint and have no larger cumulative DTW cost to the SAME selected endpoint.
This conservative gate is a geometric proxy; it is not semantic verification.
Original SR/SPL/fastdtw-nDTW and STOP rules are untouched.

## Completed unseen result

Frozen 66 routes, 11 unseen scenes, 534 eligible states:

| Same-state privileged comparison | Mean extra goal progress | States >=0.25 m | Episodes with such states |
| --- | ---: | ---: | ---: |
| Both execution-cost caps | 0.148787 m | 59 | 28 |
| Cost caps + route-endpoint-only sensitivity | 0.136866 m | 52 | 27 |
| Cost caps + primary matched-prefix route gate | **0.126962 m** | **49** | **26** |

All 11 scenes retain material states. Primary scene-cluster descriptive 95% CI:
[0.074038, 0.179887] m; episode-cluster CI [0.076090, 0.185295] m. Median gain is
zero. Of the original 59 material states, the goal-best candidate passes both
route conditions in 46; three others retain a different material alternative,
yielding 49 states. This is a local privileged opportunity, not an SR gain.

The original eight-route training pilot remains zero under both cost and route
controls. Accepted outputs are `results/route_compatibility_001/unseen66_v2/`
and `train_pilot8_v2/`.

## Expanded training result

Sixty-four new routes over 16 training scenes were selected before outcomes,
excluding the eight pilot routes and all validation routes. All 474 native
selected sentinels, 1,359 reversed-order comparisons and 6,449 branch-integrity
checks pass. The 5,090 forward branches include every admissible graph option;
38 final-budget branches are excluded, and actual STOP states stay excluded
from ranking. All 1,152 episode metrics match the untraced control exactly.

| Training64 comparison (410 non-STOP states) | Mean extra goal progress | States >=0.25 m | Episodes | Scenes |
| --- | ---: | ---: | ---: | ---: |
| Unconstrained local goal-best | 0.228170 m | 55 | — | — |
| Joint execution-cost caps | 0.053452 m | 19 | 10 | 7 |
| Cost + endpoint-only route sensitivity | 0.042316 m | 14 | 7 | 5 |
| Cost + primary matched-prefix route gate | **0.042087 m** | **14** | **7** | **5** |

Primary scene-cluster descriptive 95% CI [0.005734, 0.095675] m; episode-cluster
CI [0.007394, 0.096974] m. Median gain is zero. Training signal exists beyond the
original zero-signal pilot, but remains sparse: seven of the 14 material states
come from episode 6962. That episode contributes 61.26% of the sum of statewise
gains; excluding it descriptively lowers the mean to 0.016881 m. These mutually
exclusive branch gains must not be summed as an episode improvement.

Only nine of the 14 retained alternatives have positive absolute goal progress;
five material states occur in baseline-successful episodes. The evidence does
not support fitting a high-capacity scorer. Accepted outputs are
`results/route_compatibility_001/train64_v2/`, with post-hoc interpretation,
concentration and reproducible plots stored alongside it.

The selected action is always among the alternatives, so these local maximum
gains are nonnegative by construction. Their bootstrap intervals describe the
observed opportunity distribution; a positive interval is not a significance
test for a learned policy or evidence of end-to-end improvement.

## Interpretation checks and representative episodes

Post-hoc descriptive stratification of the 49 retained unseen states:

- Selected action has negative goal progress in 34 states.
- Alternative has positive absolute goal progress in 37; **12 alternatives
  still move away from the goal**, only less than the selected action.
- Nineteen occur in episodes the baseline ultimately succeeds on. Intervening
  may remove necessary recovery behavior; no successful continuation is known.
- The selected matched reference endpoint is the route start in 16 states and
  the route end in 10. These boundary cases limit the interpretation of a
  monotonic prefix-alignment proxy. Counts overlap; they are not failure labels.

Examples use zero-based high-level steps, full original controller actions:

| Episode / step | Observation | Meaning |
| --- | --- | --- |
| 221 / 10 | Selected progress -3.7070 m; alternative +1.7057 m, 39 vs 57 primitives and 4.5 vs 6.0 m movement. Route gate passes. | A local opportunity survives both controls; no continuation evaluated. |
| 382 / 0 | Goal-only alternative gains 1.8310 m versus selected, but matched reference endpoint regresses from 5 to 0; primary gain becomes zero. | Goal-distance improvement alone can conflict with the ordered reference. |
| 268 / 8 | Retained gain 5.2438 m, but alternative absolute progress is -0.0862 m. Baseline episode succeeds. | Relative improvement is not necessarily forward motion or an episode benefit. |

Reproducible horizontal/elevation plots and original instructions are in
`results/route_compatibility_001/unseen66_interpretation/` (PNG and PDF).
Plots do not render walls or prove landmark compliance.

## Fidelity and attempts

Every baseline path reconstructs exactly: native path length, fastdtw-nDTW and
SDTW all have zero difference across 138 episodes (414 metric comparisons).
Every eligible selected action is an exact path/DP sentinel. All 534 unseen
state identities and previous joint-cost gains match the prior analysis, as do
all 410 expanded training states. The
the primary <= endpoint-only <= cost-only opportunity relation holds statewise.

Initial unseen preflight failed on 14 path-length comparisons (maximum
4.126e-6 m) because the audit converted native float32 path arithmetic to
float64. nDTW/SDTW already matched exactly. Corrected only the reconstruction
arithmetic to native float32; **no tolerance, route gate or evaluator changed**.
The failed directory, logs and v1 source snapshot remain intact. The old pilot
analysis is superseded by the same v2 source used for unseen. Thirteen focused
tests pass, including an exhaustive DTW path comparison, route shortcut/loop,
reversed movement, required detour and exact native float32 reconstruction.

## Limits and decision

DTW follows geometric reference samples, not the full meaning of instructions.
It cannot independently certify wall/topology consistency. Prefix errors and
cursor commitment can hide recovery; a fixed selected endpoint can penalize
further route progress; sample count affects cumulative cost. Cost caps can
exclude useful expensive actions. The gate is deliberately conservative and
is not claimed to be a lower bound on counterfactual episode improvements.

The result narrows an alternative explanation: the entire unseen goal-progress
opportunity does not disappear under this route proxy, and broader training
also contains a smaller, concentrated signal. It does not establish trainability,
novelty or deployment gains. Original AAA: NO-GO. Focused graph-decision diagnosis:
CONDITIONAL GO; no ranker is trained or integrated.

The planned single oracle action intervention has now completed on the frozen
seven training routes while retaining all 64 routes/order. All controls pass;
SR stays 4/7, SPL and mean nDTW improve, but two failed routes deteriorate and
total primitives increase 674 -> 711. The ordered-reference/local-cost gate does
not certify a useful or cheap continuation. See `SINGLE_INTERVENTION_REPORT.md`
for every case and `GRAPH_DECISION_NEXT_GATE.md` for the revised learning gate.
The immutable schedule keeps its original planning-status field; execution
manifests and accepted audits record completion. No model is trained.
