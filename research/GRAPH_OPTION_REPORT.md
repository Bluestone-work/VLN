# Controller-consistent graph-option diagnostic

Date: 2026-10-07. No learned scorer or adaptive abstraction policy is evaluated.

## Measurement contract

The frozen ETPNav baseline captures every valid graph action after navigation
logits and before selected ghost deletion. A row corresponds to one graph ID,
including historical ghosts, not one raw proposal. The complete action retains
merged target, nearest front, back path, original NumPy dtypes and effective
tryout=false. The reconstructed selected dictionary must match the native action
exactly. Frozen 768-dimensional graph embeddings are stored separately from
privileged goal-distance labels.

In an independent simulator, every admissible action is executed from reset
task state, restored pose and controller RNG. The actual selected option is a
sentinel in every state. Two states per episode are chosen by metadata hash
before labels; their actions are repeated in reverse order. Endpoint, rotation,
progress, primitive/collision sequences, RNG and done must satisfy the previous
calibration gates. Raw branch traces and sensor hashes are retained.

The native high-level horizon is preserved. At the final decision, non-STOP
options are inadmissible and skipped. Actual learned/forced STOP states are
excluded from rank-only progress comparisons; STOP itself follows the baseline
historical node and back path. No success rule or episode limit is changed.

## Training pilot: GRAPH-OPTION-PILOT-001

Configuration: `configs/GRAPH_OPTION_PILOT_001.json`. Eight training episodes,
four scenes, two distinct routes per scene, one instruction per route, selected
by seeded metadata hashing before outcomes. Native frozen evaluation mode is
used on training data; no parameter updates or waypoint augmentation occur.

| Validation check | Result |
| --- | --- |
| Decisions / variable action-set size | 63 / 4–31 structural options |
| Admissible forward branches | 760 |
| Selected native action dictionary equality | 63 / 63 |
| Selected execution sentinels | 63 / 63 pass |
| Reverse-order checks | 218 / 218 pass, 16 states |
| Sentinel / reverse maximum endpoint error | 0 m / 0 m |
| Sensor hashes | 63 / 63 selected, 218 / 218 reversed |
| Final-budget non-STOP branches excluded | 30 |
| Traced versus untraced episode metrics | 144 / 144 exact |

Native baseline SR/SPL/nDTW is 100.00/95.62/89.21% on these eight training
episodes. This high-performing sample has no SR headroom and is not a
generalization or benchmark result. The sampling manifest is retained, including
all episode and route IDs; episodes are not reselected after seeing success.

Among 55 non-STOP decisions, comparing selected execution to the best actual
non-STOP alternative produces:

| Comparison | Mean available extra goal progress | States >=0.25 m |
| --- | ---: | ---: |
| All valid options | 0.038215 m | 2 / 55 |
| No more primitive actions than selected | 0.001024 m | 0 / 55 |
| No longer movement path than selected | 0 m | 0 / 55 |
| Both cost caps | 0 m | 0 / 55 |
| Current proposals plus selected action | 0.038215 m | 2 / 55 |
| Added opportunity from historical ghosts | 0 m | 0 / 55 |

Selected mean progress is 1.293489 m, best alternative 1.331704 m. Only four
states have positive raw gain; the episode-cluster 95% interval for mean raw gain
is [0, 0.077089] m. Eight episodes/four scenes make these intervals exploratory.
They are intervals on an optimistic local max, not evidence of learned gains.

For example, episode 10124, step 9: the alternative gains another 1.4973 m of
goal progress but uses 11 versus 3 primitives and walks 2.0150 versus 0.7531 m.
This is not an equal-cost selection improvement. No ranker training is justified
by this pilot. The preselected unseen66 measurement tests whether this ceiling
on a small training sample also occurs on unseen scenes, without changing
features, thresholds, checkpoint or analysis controls.

Artifacts: `results/graph_option_pilot/`, including sampling, capture/control,
noninterference, smoke/full probe, raw branch traces and `analysis_001/`.

## Unseen validation measurement

Configuration is fixed in `configs/GRAPH_OPTION_UNSEEN_001.json`, reusing the
previously frozen 66-route/11-scene subset. The same two reverse states per
episode and cost controls apply. No validation fitting is performed.

All 600 captured decisions join exactly to native execution traces. The probe
executes **7,113 admissible branches**. Selected sentinels pass 600/600, and
reverse-order repetition passes 1,516/1,516 actions across 132 states. Endpoint
errors are zero; every checked observation hash matches. At the final-decision
budget, 207 structural non-STOP options are excluded without execution.

Every one of 1,188 episode metrics and 18 aggregates matches the fresh untraced
control. The capture also matches the pre-hook archived baseline byte-for-byte.
Native subset SR/SPL/nDTW stays 66.67/54.55/60.56%; it is not a new-agent result.

The 66 actual STOP states are excluded, leaving 534 eligible non-STOP states:

| Comparison | Mean extra goal progress | Positive states | States >=0.25 m |
| --- | ---: | ---: | ---: |
| All admissible non-STOP options | 1.284608 m | 250 / 534 | 231 / 534 |
| No more primitives than selected | 0.200184 m | 93 / 534 | 72 / 534 |
| No longer movement path than selected | 0.342688 m | 129 / 534 | 114 / 534 |
| Both cost caps | **0.148787 m** | **71 / 534** | **59 / 534** |
| Current proposals plus selected action | 0.606657 m | 208 / 534 | 188 / 534 |
| Added raw opportunity from historical ghosts | 0.677951 m | 129 / 534 | 113 / 534 |

Mean selected goal progress is 0.615680 m, unconstrained best 1.900287 m.
Applying both cost caps reduces the apparent mean opportunity by **88.42%**.
For example, episode 1496, step 8, has a raw 17.5055 m gain, but the best branch
uses 95 instead of 9 primitives and walks 17.9247 instead of 2.0000 m. Its joint
cost-capped gain is zero. Raw local progress strongly rewards long recovery
options and must not be presented as equal-cost ranking improvement.

The remaining joint-capped mean has episode-cluster 95% interval
**[0.097752, 0.207658] m** and scene-cluster interval
**[0.093913, 0.204321] m** (5,000 resamples, seed 20261007). All 11 scene means
are positive, ranging from 0.037203 to 0.328147 m, while the median state gain is
zero. These describe an optimistic privileged local maximum on a fixed sample;
they do not measure a learned model's statistical improvement.

### Descriptive failure follow-up

The 59 joint-capped states with gain >=0.25 m occur in 28 episodes. Ten have a
collision during the original selected option; 49 do not. Thus a simple
collision-only explanation cannot describe all these local opportunities.
Among baseline-failed episodes, 36/205 non-STOP states have such a gain, compared
with 23/329 states in successful episodes. Outcome stratification is post hoc
and confounded by episode difficulty, not a causal failure partition.

Representative joint-capped examples (zero-based step; chosen by largest gain):

- Episode 221, step 10: selected historical ghost uses 57 primitives / 6.0000 m
  and loses 3.7070 m of goal progress; an alternative uses 39 / 4.5000 m and
  gains 1.7057 m. Neither branch collides. The original episode fails.
- Episode 1238, step 9: selected current ghost uses 16 primitives / 2.2611 m,
  progress -1.2807 m; another current ghost uses 9 / 2.2601 m, progress +2.2318 m.
  Neither branch collides. This is a local alternative outcome, not proof that
  changing the action would complete the instruction or episode successfully.

Artifacts: `results/graph_option_unseen/analysis_001/summary.json`,
`interpretation.json`, `probe_full002/`, `noninterference_002/`, and provenance
snapshots. An early check started before the control finished and failed
preflight; the dependent first probe therefore executed no branches. Preserve
`noninterference_001/`, `probe_full001/` and their logs as rejected attempts;
`attempts.json` identifies accepted runs. No settings or gates were relaxed.

## Verification and provenance

Across train and unseen, 663 decision sentinels and 1,734 reverse checks pass.
There are 7,873 forward branches; 237 final-budget options are skipped.
An independent raw-trace audit additionally checks **all 9,607 executions**
(including reversed repeats), verifying start pose, starting goal distance,
starting RNG, exact action dictionaries, episode identity, admissibility and
complete coverage. Start-position and goal-distance errors are zero. Accepted
summaries are `graph_option_pilot/integrity_001/summary.json` and
`graph_option_unseen/integrity_001/summary.json`.
All 36 focused tests pass (30 diagnostic/protocol tests plus six existing action
generator tests), along with compilation and `git diff --check`. Source hashes,
checkpoint hash, dataset manifests, git state, hardware, seed and stdout/stderr
are retained. Historical results are not overwritten.

## Broader training and ordered-route follow-up

`GRAPH-OPTION-TRAIN64-001` freezes 64 new training routes across 16 scenes,
excluding pilot routes and both validation splits' routes. All 474 captured
states, 5,090 forward branches, 474 selected sentinels and 1,359 reverse checks
complete. Thirty-eight inadmissible final-budget options are skipped. Independent
integrity checks pass on all 6,449 executions including repeats. Endpoint errors
are zero; all checked observation hashes match. Tracing preserves all 1,152
per-episode metrics exactly. Frozen training-subset SR/SPL/nDTW is
85.94/81.10/83.15%, not a new-agent score.

On 410 eligible training states, raw mean local goal opportunity is 0.228170 m,
primitive-capped 0.065716 m, path-capped 0.091953 m and joint-capped 0.053452 m.
Nineteen states in ten routes/seven scenes have joint-capped gain >=0.25 m.

The separately frozen ordered-reference audit further reduces training opportunity
to **0.042087 m**, with 14 material states in seven routes/five scenes. Seven
material states come from a single episode, 6962. This concentration prevents
interpreting all rows as independent training evidence. Unseen66 retains
**0.126962 m** and 49 material states after the identical gate. The pilot stays
zero. No scorer is fit. See `ROUTE_COMPATIBILITY_REPORT.md` for the metric,
clustered intervals, counterexamples, attempt history and future continuation gate.

All 414 native path-length/nDTW/SDTW comparisons reconstruct exactly across
138 episodes; previous cost-only state results remain unchanged. Outputs:
`results/graph_option_train64/` and `results/route_compatibility_001/`.

## Limits and decision

Goal geodesic progress is privileged and local. It can penalize necessary detours
and can favor longer options. The cost caps control that confound but may exclude
useful behavior with a necessary upfront cost. Neither raw nor capped regret is
a causal six-way failure partition or a guaranteed episode-performance gain.
These are fixed-snapshot branches, not a closed-loop oracle navigation rollout.

The old raw-proposal proxy gaps are not directly comparable numerically: this
study changes label identity, includes historical graph options and uses a
different sample. Goal-directed progress also does not establish instruction-
consistent reference-route progress. A branch may cut a corner or skip a landmark.

**NO-GO for the current NMS-density adaptive selector remains unchanged.**
Method-level prior overlap also narrows the original premise; see
`LITERATURE_METHOD_AUDIT_20261007.md`.

**CONDITIONAL GO for a narrow graph-decision efficiency study.** The frozen
seven-route intervention now completes with exact controls. Native continuation
retains some SPL/nDTW benefit, but success stays 4/7, two failed routes deteriorate
and total primitives rise by 37. The one-step cost and route gates do not certify
recovery or episode-level compute benefit. No scorer is trained. See
`SINGLE_INTERVENTION_REPORT.md` for all cases, and require prospective validation
of continuation-aware labels on new training routes before any fitting. The
H=2 reversal in the two bad cases is post-hoc evidence, not a validated new rule.
No RL, VLM, new loss or navigation model is added.
