# Prospective frozen graph-preference replication 001

**Decision: NO-GO for scaling the frozen first-disagreement recipe.**
All six registered arms and fidelity audits completed. The SR point gain remains,
but mean primitive cost increases and the added-feature gate fails. Protocol:
`GRAPH_VALUE_PROSPECTIVE_REPLICATION_PROTOCOL.md`.

## Question and chronology

Does the previous exploratory first-disagreement result replicate, and does
the full scorer add value over graph scores plus execution distance?

The preceding 96-route experiment rescued three routes with no success loss,
SR/SPL/nDTW changes of +3.125/+2.302/+0.956 percentage points and 1.677 fewer
primitives per route. Its schedule was selected after inspecting the target
census. This experiment freezes the protocol, sample, models and executable
sources before any new target capture.

Registration UTC: `2026-10-07T15:32:08.344326+00:00`; foundation started at
`15:32:09.564338`; target capture at `15:32:12.145484`. Registration and execution
commit: `6fb04822b81c5a08003b11271d6d9b9ecbaf763c`, pushed before capture.
The registration's `source_git_commit` records its parent `d1ee8e8`, because the
registration itself was included in the next commit. Twenty-two file hashes
pin the source/config/model/checkpoint/data contents, including dirty core files;
the commit alone is insufficient to reconstruct the working tree.

## Population and controls

Metadata-only sampling (seed 20261024) selects 96 R2R **training** routes,
12 in each of eight scenes. The exclusion inventory contains 1,584 unique route
keys, including both previous 96-route cohorts. The declared-route overlap
audit passed. No route was replaced after observing native success or failure.
Scene overlap is allowed. This is a holdout from the declared research
diagnostics, **not from baseline-checkpoint training** and not a benchmark.

Scenes: ZMojNkEp431, D7N2EKCX4Sj, 1pXnuDYAj8r, pRbA3pwrgk9, sKLMLpTHeUy,
8WUmhLawc2A, 17DRP5sb8fy and Vvot9Ly1tCj.

All arms retain checkpoint `release_r2r/ckpt.iter12000.pth`, simulator seed 100,
one worker/GPU, normal RGB/depth, sliding enabled (tryout therefore inactive),
the control backtracking algorithm, graph masks, native STOP and the 15-decision
horizon. Hardware inventory: two NVIDIA RTX 4090 cards, driver 595.84; each run
uses one GPU. Python/Habitat provenance is in the capture/run manifests.

## Frozen models and comparisons

FULL is byte-identical to the prior frozen linear scorer. Both models use only
the old fresh training census: 47 states, 996 actions and 5,668 strict dominance
pairs. COST refits the same ridge-10 objective/normalization using only graph
logit, normalized logit rank, ghost distance and back-path length. Other weights
are zero. COST also uses full-return labels; this experiment tests feature
sufficiency, not the necessity of long-horizon labels.

FULL and COST-own each replace at most one non-STOP action at their first
disagreement with native, then let the original navigator continue. COST-matched
scores at FULL's first-disagreement state and can retain the native action.
Three random arms (seeds 20261031/32/33) uniformly sample the same admissible
non-STOP set at FULL's state, including native abstention. All 96 routes remain
in every denominator. COST-matched depends on FULL's timing, so it is not an
independent reduced deployment policy.

The linear pairwise scorer cannot exploit state-only features that cancel
between actions: entropy, STOP probability and candidate count have zero
weights. FULL's embedding inputs are summary statistics, not a new semantic
reasoning model. No target continuation labels were used for fitting or timing.

## Registered decision rules

FULL must improve mean SR, not reduce mean SPL/nDTW, not increase mean primitive
count, rescue SR without nDTW loss in at least two scenes, and beat mean random
SR. Every condition is required. Failure means NO-GO for scaling this frozen
first-disagreement recipe, not rejection of all graph-value methods.

Additional-feature evidence requires FULL to beat COST-matched in SR with a
paired scene-bootstrap interval excluding zero, without mean SPL/nDTW/primitive
harm. Bootstrap: 5,000 samples, seed 20261025, eight scene clusters. These small,
overlapping training-scene intervals cannot establish benchmark generalization.

## Results (completed 2026-10-08 Asia/Shanghai)

All values use all 96 routes. SR/SPL/nDTW are percentages; primitive counts are per route.

| Arm | Changed routes | SR | SPL | nDTW | Primitives | Rescued / lost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Native | 0 | 91.6667 | 84.4428 | 84.8572 | 56.3125 | — |
| FULL | 29 | 94.7917 | 86.2138 | 85.7952 | 57.5729 | 4 / 1 |
| COST-own | 74 | 94.7917 | 86.6087 | 86.1980 | 59.7396 | 4 / 1 |
| COST-matched | 23 | 93.7500 | 85.4720 | 85.9446 | 57.3333 | 3 / 1 |
| Random 20261031 | 28 | 91.6667 | 78.3281 | 78.9223 | 75.6042 | 3 / 3 |
| Random 20261032 | 26 | 90.6250 | 76.2502 | 79.0493 | 73.7500 | 1 / 2 |
| Random 20261033 | 25 | 90.6250 | 77.5448 | 78.7259 | 76.2917 | 2 / 3 |

FULL changes 29/96 routes (30.21%); COST-own changes 74/96 (77.08%). At the
29 FULL states, COST-matched changes 23 and abstains on six. FULL and
COST-matched choose exactly the same action on 21 of the 29 eligible routes.
This leaves only eight routes distinguishing their candidate choices.

| FULL minus native | Paired mean change | 95% scene-bootstrap interval |
| --- | ---: | --- |
| SR (pp) | +3.1250 | [-2.0833, +8.3333] |
| SPL (pp) | +1.7710 | [-2.2699, +6.2761] |
| nDTW (pp) | +0.9379 | [-0.9243, +3.9784] |
| SDTW (pp) | +1.1041 | [-2.4781, +5.4638] |
| Final goal error (m) | -0.3124 | [-0.8612, +0.0466] |
| Path length (m) | -0.0561 | [-0.3446, +0.2138] |
| Primitives / route | +1.2604 | [-1.2188, +4.3023] |
| High-level decisions / route | +0.2188 | [+0.0833, +0.3542] |
| Collision events / route | +0.1771 | [-0.0729, +0.4583] |

All four primary FULL-native intervals include zero. This replication therefore
does not establish a reliable SR gain, even though its point change is positive.
The mean random SR is 90.9722%; FULL beats this by 3.8194 pp (scene interval
[+1.0417, +6.9444] pp). Beating random does not satisfy the separate native
cost gate. The three random seeds are action randomizations, not training seeds.

## Reduced-feature contrast and route harms

FULL minus COST-matched: SR +1.0417 pp (CI [0, +3.1250]), SPL +0.7418 pp,
nDTW −0.1494 pp and primitives +0.2396/route. The added-feature gate fails:
the SR interval touches zero, and mean nDTW/cost worsen. COST-own matches
FULL SR with higher SPL/nDTW point means, but uses 2.1667 more primitives than
FULL and intervenes much more often. Neither contrast establishes a superior
deployable reduced model or a necessary semantic/long-horizon feature mechanism.

| Arm | Routes losing nDTW | Routes losing SPL | Routes using more primitives |
| --- | ---: | ---: | ---: |
| FULL | 15 | 12 | 18 |
| COST-own | 37 | 28 | 48 |
| COST-matched | 11 | 10 | 15 |
| Random 20261031 | 23 | 18 | 27 |
| Random 20261032 | 22 | 21 | 25 |
| Random 20261033 | 22 | 17 | 22 |

FULL rescues 1943, 4790, 4010 and 6098, but loses previously successful 4841.
Only the first three rescues also preserve nDTW, across two scenes.

| Route | SR change | nDTW change (pp) | Primitive change |
| --- | ---: | ---: | ---: |
| 1943 | +1 | +44.4153 | -50 |
| 4790 | +1 | +83.0929 | -19 |
| 4010 | +1 | +2.4652 | +0 |
| 6098 | +1 | -15.6878 | +44 |
| 4841 | -1 | -4.3217 | -16 |

## Post hoc cost accounting

With exact pre-intervention prefixes, the episode cost difference can be split
into the replaced option and the remaining continuation. This is accounting,
not proof of a causal mediation mechanism or a new fitted gate.

| Arm | Replaced options, total primitive delta | Remaining continuations | Full episodes |
| --- | ---: | ---: | ---: |
| FULL | -133 | +254 | +121 |
| COST-own | -238 | +567 | +329 |
| COST-matched | -79 | +177 | +98 |

For FULL, 14 routes use a cheaper replacement option but a more expensive
complete episode. The four rescued routes jointly save 25 primitives; the one
lost route saves 16; routes with unchanged binary success add 162. Therefore
the net +121 primitives cannot be explained as simply paying to rescue failures.
Immediate execution distance is an inadequate substitute for continuation cost.

## Fidelity and decision

All six disabled runs reproduce 724 native high-level records each. Baseline
trace/control equality covers 1,728 episode metrics and 18 aggregate metrics.
All 205 actual substitutions pass exact action/prefix checks. Independent final
metric reconstruction passes 576 comparisons per arm (3,456 total), with zero
absolute error. Cross-arm equivalence passes 838 route comparisons.
The replay foundation checks 7,994 forward and 2,032 reverse one-option branches;
these 10,026 probes are **not** full-episode counterfactual continuations.

| Registered FULL-native condition | Outcome |
| --- | --- |
| positive_sr | PASS |
| nondecreasing_spl | PASS |
| nondecreasing_ndtw | PASS |
| no_primitive_increase | FAIL |
| quality_rescue_in_two_scenes | PASS |
| sr_beats_mean_random | PASS |

**NO-GO for scaling the frozen first-disagreement recipe.** The mean primitive
gate fails. The additional-feature gate also fails. Preserve the positive SR
point estimate and negative evidence together; do not retune thresholds, features
or weights on this cohort to reverse the decision. This does not prove proposal
coverage failure, execution failure dominance, or that every graph-value method
is ineffective. It does not provide new support for an interrupt policy.

The useful next question is whether harmful early substitutions can be identified
from information available before acting, with full continuation cost rather than
local distance as the target. Before any new learning, register a diagnostic on
development data that includes native-successful routes and retains abstention.
This is a future diagnostic question, not an approved claim of a new method.
Independent scenes/route cohorts remain necessary for subsequent confirmation.
Adaptive waypoint density, interrupt-policy training, RL and VLM remain closed.

The exploratory preceding cohort remains in its original report; its positive
cost result does not override this prospectively registered failure. Do not pool
the two cohorts post hoc and call the combined mean a passed replication.

## Operational exception, preserved before retry

After FULL, COST-own and COST-matched had passed all audits, the first random
arm's disabled run failed while starting the Habitat forkserver worker with
`BrokenPipeError`. Its hook log is empty and no episode trace exists. The original
manifest, stdout/stderr and failed cycle status are retained. The operational
amendment in `continuation_001/amendment.json` was written before retrying with
run ID `random_20261031_disabled_retry001`; the original run was not overwritten.
All registered source hashes and the execution commit were checked again.
No sample, seed, action, outcome gate or scientific setting changed. The remaining
arms use the original frozen runner components. Cycle status retains the failed
stage as well as the successful continuation, rather than presenting a clean
first attempt.

## Artifact locations

All paths below are relative to
`research/results/graph_value_prospective_replication_001/`:

- `registration_001.json`, `sampling_001/route_audit_001.json`, `models/`:
  prospective registration, declared-route exclusions and frozen models.
- `capture_001/manifest.json`, `execution_sources_001/`: runtime provenance,
  source snapshots and the dirty execution-source patch.
- `noninterference_001/`, `integrity_001/`: baseline invariance and option replay.
- `schedules_001/`: every frozen intervention/abstention decision.
- `replication_cycle_001/`, `continuation_001/`: all arms, the failed startup,
  disabled/enabled audits, independent metric reconstructions and durable logs.
- `paired_analysis_001/summary.json`: the unmodified registered analysis.
- `equivalence_audit_001/`: independent equality of identical cross-arm actions.
- `route_diagnostics_001/`: explicitly post hoc per-route accounting, not a new gate.
- `figures_001/`: standalone PNG/PDF plots of paired effects and intervals.

Raw sensor/primitive/option traces remain local and are identified by archive
hashes. Reviewable metrics, per-route tables, provenance and logs are retained in
Git. The legacy intervention hook conservatively labels its metadata as oracle;
the schedule and audit distinguish these learned choices from privileged ones.
This remains an offline one-action intervention harness, not a shipped online
navigation policy. No inference-latency improvement is claimed.
