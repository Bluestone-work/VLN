# Frozen preference continuation mechanism audit

**Decision: the frozen recipe remains NO-GO.** The new evidence describes a
local execution-cost preference with inconsistent continuation costs. It does
not establish a deployable abstention rule or a reason to train an interrupt
policy. This is a retrospective audit of existing runs, not new navigation
performance or an independent replication.

## Scope and fidelity

All 27 earlier FULL interventions and all 29 replication FULL interventions are
primary observations. COST-own (74) and COST-matched (23) are secondary controls
with overlapping replication episodes. All 153 scheduled events are retained,
including native successes. Each arm retains its 96-route population denominator.
Do not pool primary cohorts into a new benchmark result or count overlapping
arms as independent samples.

The specification was frozen after earlier outcomes were known and before this
derived census. It reuses collision, 0.5 m horizontal residual, and >=3-primitives/
<=0.1 m-motion stall definitions from the critical census. Target reselection
requires the displaced ghost ID; stable targets have <=0.25 m 3D drift.
These fixed physical thresholds were not optimized on outcomes. A stable target
does not imply an identical complete action or prove a detour was unnecessary.

All 153 actual action/prefix checks and primitive/collision/path accounting
checks pass against archived full-episode metrics. Four tests pass, including
future-label invariance of pre-action features, moving-ghost identity, rotation
stall accounting, and STOP return movement. No checkpoint, sensor, controller,
STOP or 15-decision setting changed; no new simulator episodes or models ran.
Original seed 100, checkpoints, hardware and execution commits are retained
per batch; analysis source/config/input hashes and stdout/stderr are archived.

## Where the costs occur

Values below are total primitive changes over each 96-route population.

| Arm | Changed routes | Changed option | Later navigation | Later STOP | Whole episode |
| --- | ---: | ---: | ---: | ---: | ---: |
| Exploratory FULL | 27 | -106 | +36 | -91 | -161 |
| Replication FULL | 29 | -133 | +200 | +54 | +121 |
| Replication COST-own | 74 | -238 | +518 | +49 | +329 |
| Replication COST-matched | 23 | -79 | +123 | +54 | +98 |

The earlier net −161 is sensitive to route 10199, which alone contributes
−155; removing it for a descriptive sensitivity check leaves −6, not a new
evaluation score. Its costs are −22 immediate and −133 later navigation.
A different rescued route, 3161, changes immediate cost by +10 and later
navigation by −1, but avoids 91 primitives of budget-STOP return execution.
Therefore immediate savings alone do not explain that earlier positive result.

In the replication, STOP cost increases come solely from 6098 (+45) and 6020
(+9), both with native and intervention budget STOP. Route 6098 replaces at
step 13, so there is no subsequent non-STOP navigation: immediate −1 plus
STOP +45 produces net +44. It gains success while losing 15.688 nDTW points.
This is an interaction with native return-to-stop behavior, not an altered
success definition or proof that the STOP policy itself should be changed.

## Cost reversals and execution symptoms

A reversal means the replacement option uses fewer primitives, but its complete
episode uses more. All other interventions remain available as comparisons.

| Arm | Reversals | Native-successful reversals | Alternative execution symptom | Stable target selected next | Reversal cost total |
| --- | ---: | ---: | ---: | ---: | ---: |
| Exploratory FULL | 10 | 10/10 | 2/10 | 4/10 | +131 |
| Replication FULL | 14 | 12/14 | 2/14 | 7/14 | +189 |
| Replication COST-own | 40 | 37/40 | 4/40 | 12/40 | +465 |
| Replication COST-matched | 12 | 10/12 | 2/12 | 5/12 | +156 |

Only 2/10 and 2/14 primary reversals exhibit immediate replacement execution
symptoms; these symptoms do not cover most observed cost reversals. This
selected, mostly short-option population cannot reject execution failures or
interrupt benefits across the whole native agent. No interrupt was tested here.

Stable immediate reselection occurs in 4/10 and 7/14 reversals, but also in
4/17 and 2/15 other FULL interventions. Thus it is neither universal nor
exclusive to expensive continuations, and it is visible only after replacement.
It must not be used as a pre-action feature. Route 418 illustrates the pattern:
no replacement execution symptom, immediate −6 primitives, then the same ghost
selected at the next decision with only 4.7 cm target drift. Its first two
options cost 19 more primitives than the original native option; total cost
rises by 24 and nDTW falls 1.720 pp. These paths have different observations
and decision counts, so this accounting does not establish a causal remedy.

Route 10430 is a different case: immediate −8 but later navigation +55, with
21 additional later collision events. Native and alternative behavior both
require examination; an execution symptom does not certify target correctness.
Route 4841 loses success despite using 16 fewer primitives and having no
immediate replacement execution symptom. Cost-only reasoning misses this harm.

## What the frozen score is selecting

All 56 primary alternatives are current-observation proposals with empty
back-paths. Their final ghost segments are shorter than native in 26/27 earlier
and 27/29 replication events (53/56 total). The candidate-current and empty-
back-path flags consequently have no variation within these treated samples.

Native historical versus current candidates gives only one historical event
in the earlier cohort (10199, a large rescue) and three in the replication
(no SR+nDTW rescue, one cost reversal). Same-front membership has the same
counts here. This is insufficient support for a stable structural abstention
rule; it does not rule out useful information in other continuous/history
features, which this audit does not fit.

An explicitly post hoc algebraic attribution retains the frozen weights:
`contribution_j = weight_j * (alternative_j - native_j) / scale_j`.
Every group sum reproduces its pair-score margin. It does not remove features,
rerank all candidates, or constitute a causal ablation.

| Cohort / subset | Events | Largest positive contribution: execution geometry |
| --- | ---: | ---: |
| Exploratory FULL / all_events | 27 | 24 |
| Exploratory FULL / cost_reversals | 10 | 10 |
| Replication FULL / all_events | 29 | 25 |
| Replication FULL / cost_reversals | 14 | 13 |

Geometry is the largest positive contribution in 49/56 FULL choices and
23/24 cost reversals. It is also largest in 14/17 and 12/15 non-reversal choices.
This explains much of the scorer's behavior but is not a reliable failure
detector. A full-return training label does not, by itself, demonstrate that
the fitted function has learned long-horizon state-dependent value.

## Interpretation and next gate

The prospective NO-GO is unchanged. The supported diagnosis is narrower than
a new method: this frozen linear scorer frequently substitutes shorter local
targets, and any immediate savings may be offset by later navigation or STOP
return execution. It occasionally rescues large errors, but the evidence does
not support systematic deployment or a collision-triggered solution to these
particular harms. No majority causal failure category is assigned.

Before any new training, the next bounded diagnostic is a **training-support
audit on the old development census only**: compare strict-pair accuracy with
the actual native-versus-chosen full-return outcomes, retaining mixed/tied
pairs and native abstention. Check how much training support involves STOP or
comparisons between two non-native actions that the deployed trigger never
directly evaluates. Use the existing 47-state/996-action archive; no new labels
or replication-route fitting. This is a postmortem of the failed recipe, not
permission to expand it. Its frozen specification is
`PREFERENCE_TRAINING_SUPPORT_PROTOCOL.md`.

## Artifacts

Protocol/config: `PREFERENCE_CONTINUATION_MECHANISM_PROTOCOL.md` and
`configs/PREFERENCE_CONTINUATION_MECHANISM_001.json`.
All outputs are under `results/preference_continuation_mechanism_001/`:

- `registration.json`: explicitly retrospective specification and input hashes.
- `analysis_001/summary.json`, `events.json`, `events.csv`: all 153 events,
  primary cohorts and overlapping controls separated.
- `attribution_001/`: post hoc specification and all 56 conserved score splits.
- `figures_001/cost_components.png` and `.pdf`: standalone component plot.
- stdout/stderr and test records; original primitive/sensor traces stay in
  their existing archived locations and are referenced by SHA256.
