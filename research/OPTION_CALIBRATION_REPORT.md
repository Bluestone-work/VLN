# Full graph-option replay calibration

Date: 2026-10-07. Research gate: validate measurement before training a new ranker.

Follow-up: the subsequent full-option pilot and unseen measurement are complete
in `GRAPH_OPTION_REPORT.md`. This report retains the narrower scope of the
initial selected-option calibration.

## Implementation and controls

`vlnce_baselines/adaptive_action/option_calibration.py` registers an opt-in
`VLNCEOptionTraceEnv` subclass. The research runner imports it explicitly; the
normal baseline entry point is unchanged. It delegates to the original
`step`, `wrap_act`, `turn` and single/multi-step controller. No new policy,
candidate generator, reward or loss is introduced.

The trace retains typed full action dictionaries, merged targets and back paths,
pre/post pose and Python/NumPy RNG, primitive actions/collisions/poses, numeric
observation hashes, and privileged goal distances marked for analysis only.
STOP is an explicit event with collision=null; cached collision from the last
movement is not attributed to STOP.

Separate sequential replay follows complete episodes. Isolated replay resets
episode/task and measures, restores the captured pose/RNG and executes one full
option. It does not rewind a live policy rollout. Full-episode task history is
not reconstructed for isolated branches; endpoint, primitive/collision sequence,
goal progress, RNG and terminal flag are the preregistered checks.

Protocol: released `ckpt.iter12000.pth`, seed 100, one RTX 4090 / one worker,
`IL.back_algo=control`, original sensors, 15 high-level decision limit and 3 m
success rule. Sliding remains enabled, making effective tryout **false**.
This experiment does not validate stochastic tryout, noise or video mode.

## OPTION-CALIBRATION-001: val_seen64

Config: `configs/OPTION_CALIBRATION_001.json`.

| Check | Result |
| --- | --- |
| Episodes / scenes | 64 / 4 |
| Selected options | 511: 447 ghost, 64 STOP |
| Sequential replay | 511 / 511 pass |
| Isolated replay | 511 / 511 pass |
| Maximum endpoint / progress error | 0 m / 0 m |
| Maximum rotation error | 4.21e-8 rad |
| Primitive sequence / collision / RNG / done | exact in both modes |
| Numeric observation hash agreement | 511 / 511 in each mode |
| Primitive events / collision events | 4,293 / 376 |
| Options with nonempty back path | 29 |
| Effective tryout options | 0 |
| Untraced control, per-episode metric comparisons | 1,152 / 1,152 exact |
| Aggregate metric comparisons | 18 / 18 exact |

Capture and control result files have identical SHA256 values. Baseline
SR/SPL/nDTW remain **79.6875 / 70.1608 / 73.6074%**. This verifies metric
noninterference, not latency neutrality; tracing, rendering and serialization
add diagnostic work and are not used for a deployable inference-cost claim.

Artifacts under `results/option_calibration/`: `capture_001/`, `control_001/`,
`replay_smoke001/`, `replay_full001/`, `noninterference_001/`, `execution_001/`.
Capture/control manifests record checkpoint/source hashes, git state, protocol,
hardware, command and environment. Replay records capture/source SHA256.
Each run's stdout/stderr is stored beside its directory; no old run is overwritten.
`cycle_20261007/` adds the consolidated machine-readable result, pip environment,
working-tree diff and archived source snapshots whose hashes match each capture.
The environment remains Habitat-Lab/Sim 0.1.7, PyTorch 1.9.1+cu111 and NumPy 1.19.5.

## Selected execution symptoms

The seen64 trace directly observes 91/447 ghost options with a collision, 58/447
ending more than 0.5 m horizontally from their selected target, and 30/447 more
than 1 m away. Median horizontal residual is 0.1570 m; mean 0.2882 m.
Target residual is measured against the requested merged ghost, not the goal.
These thresholds are diagnostics, not changes to benchmark success definitions.

Among 13 failed episodes, seven have a collision somewhere, three end with a
forced STOP and four have native oracle-success=1. These categories overlap.
None loses the 3 m goal neighborhood during its final STOP option on this sample.
This does not establish that STOP timing or earlier recovery is correct.

The old 0.052 m execution proxy cannot justify dismissing controller effects.
Conversely, a collision or residual does not prove the selected target was
appropriate, reachable, or responsible for episode failure. Causal selection
versus execution counts remain unmeasured.

## Balanced unseen extension

OPTION-CALIBRATION-002 uses six distinct routes per unseen scene and one
instruction per route: 66 episodes over all 11 unseen scenes. Selection uses
seeded SHA256 ordering of metadata before outcomes, with no train-route overlap.
The original dataset is untouched; the copied subset and its source hashes are
in `results/option_calibration/unseen_balanced66_sampling/`. This is a balanced
diagnostic sample, not a replacement for the 1,839-episode benchmark.

| Check | OPTION-CALIBRATION-002 result |
| --- | --- |
| Episodes / scenes / distinct routes | 66 / 11 / 66 |
| Selected options | 600: 534 ghost, 66 STOP |
| Sequential replay | 600 / 600 pass |
| Isolated replay | 600 / 600 pass |
| Maximum endpoint / progress error | 0 m / 0 m |
| Maximum rotation error | 4.21e-8 rad |
| Primitive sequence / collision / RNG / done | exact in both modes |
| Numeric observation hash agreement | 600 / 600 in each mode |
| Primitive events / collision events | 5,730 / 341 |
| Options with nonempty back path / effective tryout | 41 / 0 |
| Untraced control, per-episode metric comparisons | 1,188 / 1,188 exact |
| Aggregate metric comparisons | 18 / 18 exact |

Baseline subset SR/SPL/nDTW is **66.6667 / 54.5468 / 60.5601%**, identical
with and without tracing. It must not be compared as an improvement over the
full-split 57.0962% SR: the evaluation samples differ. Replay and control
artifacts use the corresponding `capture_002/`, `control_002/`,
`replay_full002/`, `noninterference_002/`, `execution_002/` directories.

The unseen trace has 88/534 ghost options with a collision, 48/534 more than
0.5 m horizontally from the selected target, and 26/534 more than 1 m away.
Among 22 failed episodes, ten have a collision, five have forced STOP and two
have native oracle-success=1. These are overlapping symptoms with the same
limitations as seen64. Examples are in `FAILURE_ANALYSIS.md`.

## Verification

The two samples cover 130 episodes and 1,111 selected options. Each replay mode
passes all 1,111 options (2,222 option executions in total); no required gate
was relaxed after inspection. Source traces contain 10,023 primitive events,
including 130 STOP markers, 717 collision events and 70 nonempty back paths.

All 24 focused tests pass: eight serialization/RNG/replay-gate tests, three
sampling/noninterference tests, seven existing ranker-validity tests and six
existing action-generator/mask tests. Python compilation and `git diff --check`
pass. Outputs are `unit_tests_20261007.*` and `action_tests_20261007.*` in the
calibration results directory.

## Conclusion and next gate

Both seen64 and balanced unseen66 selected-option replay and baseline
noninterference gates pass.
This establishes a tested execution path for selected options under the stated
settings. Unexecuted alternatives, learned selection and navigation improvement
are **not** validated by this result. `OPTION_LABEL_PROTOCOL.md` defines the
next controlled all-option pilot and its stop criteria.

The NMS-density Adaptive Action Abstraction selector remains **NO-GO**.
Graph-decision measurement remains **CONDITIONAL GO**; no ranker, RL or VLM
component is integrated into navigation.
