# Frozen preference training-support audit

**Decision: NO-GO remains for the frozen first-disagreement recipe.** Retire
all-pair accuracy as a sufficient development gate: the frozen model achieves
94.34% strict-pair accuracy on its own fitting corpus, but identifies none of
the 14 strict native-improving move comparisons. Its two native-relative
overrides include one strict degradation and one mixed success/quality/cost
tradeoff. No new policy is trained or evaluated.

## Scope and controls

This executes `PREFERENCE_TRAINING_SUPPORT_PROTOCOL.md`, using only the old
`critical_graph_value_fresh` development census and its already frozen FULL
model. The question/config were committed in `6100821` before this analysis;
executable/input hashes were recorded before running the analyzer.
The frozen protocol/config retain their pre-run status; completion is recorded
in this report and the result `status.json`.

All 47 states and 996 full-return action cases are retained: ten failed native
routes in six training scenes. The frozen feature names, weights, means and
scales match the original fitting corpus. No model is refit. The prospective
96-route replication outcome files are not read. These are **in-sample**,
failure-selected, correlated critical states, not fresh validation or the
first disagreements of a complete deployed navigation policy.

Original full returns retain seed 100, native checkpoint/controller/sensors,
STOP behavior and the 15-decision limit. This audit runs no simulator episodes.
Its registration pins the fitting data, graph logs, model, analysis code and
existing independent full-return audit. Original simulation hardware/config
provenance is retained separately from the current analysis execution commit.

## Pair support and accuracy

There are **12,159** unordered action pairs: **5,668** strict dominance pairs,
**6,491** mixed/incomparable pairs (53.38%), and **0** metric ties. Strict
dominance requires no degradation in SR, SPL, nDTW, final goal distance, path
length and primitive count, plus at least one strict improvement. Mixed
comparisons were excluded from the old pairwise fit, not labeled failures.

The following three subgroups partition all strict pairs. Native means the
actual effective action index, not a separately reconstructed argmax.

| Strict-pair population | Pairs | Share of fitting pairs | Frozen model correct | Native logit correct | Change in correct pairs |
| --- | ---: | ---: | ---: | ---: | ---: |
| All strict pairs | 5668 | 100.00% | 5347/5668 (94.34%) | 4829/5668 (85.20%) | +518 |
| Contains STOP | 424 | 7.48% | 411/424 (96.93%) | 296/424 (69.81%) | +115 |
| Native vs another move | 418 | 7.37% | 403/418 (96.41%) | 404/418 (96.65%) | -1 |
| Two non-native moves | 4826 | 85.14% | 4533/4826 (93.93%) | 4129/4826 (85.56%) | +404 |

The apparent all-pair improvement is 518 correct comparisons: +404 between
two non-native moves, +115 involving STOP, and **−1** in native-versus-move
comparisons. The latter constitute only **7.37%** of strict fitting pairs.
Two non-native moves can still matter in a general ranking policy; the point
is that their accuracy does not certify safe overrides of this native policy.
STOP-involving accuracy also does not certify this override rule, which
explicitly protects the original STOP decision.

`all_native_involving` is an additional overlapping diagnostic, including
native STOP cases: 525 strict pairs, model 509/525 and logit 499/525. Do not
add it to the disjoint groups above or confuse it with movable native overrides.

The 94.34% here is frozen-model **in-sample** accuracy; it is not the earlier
93.86% leave-one-scene-out feasibility result. Neither number measures actual
episode-policy performance.

## Which native-relative direction is supported?

A post hoc directional split of the already audited 418 strict native-move
pairs exposes their class balance. It adds no model, threshold or selection.

| Preferred action in full-return labels | Pairs | States / routes / scenes | Frozen model correct | Native logit correct |
| --- | ---: | --- | ---: | ---: |
| Keep native | 404 | 33 / 10 / 6 | 403 | 404 |
| Prefer alternative | 14 | 12 / 6 / 4 | 0 | 0 |

Thus the model preserves 403/404 correct native preferences but recognizes
**0/14** strict improvements over native, despite seeing those labels during
fitting. These 14 comparisons occur at 12 states in six routes and four scenes.
A strict improvement need not rescue success; it means the recorded six-metric
return dominates native. Counts across the two direction rows can overlap in
states/routes/scenes and must not be added as independent support.

## Actual masked choices at the frozen critical states

The audited selector uses the same weights, admissible action masks and original
index tie-break as the prior intervention rule. Native STOP/budget decisions
are protected. It independently scores each archived critical state; it does
not replay the first-disagreement trigger across whole episodes.

- 10 states retain protected STOP/budget behavior.
- Of 37 movable states, 35 retain native and two choose an alternative.
- All 47 native full returns reproduce their archived failed baseline metrics.
- A strictly dominating non-STOP option exists in 12 movable states. Eleven
  retain native; the remaining state selects a strictly worse action.
- No selected override strictly improves all six metrics; both harm nDTW,
  path length and primitive cost. One rescues binary success.

| Route / step | Native → selected index | Full-return relation | ΔSR | ΔSPL (pp) | ΔnDTW (pp) | Δgoal error (m) | Δpath (m) | Δprimitives |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 10506 / 10 | 38 → 39 | strict_degradation | +0 | +0.0000 | -0.2248 | +0.4955 | +4.3948 | +59 |
| 3865 / 9 | 24 → 20 | mixed | +1 | +41.6745 | -9.1749 | -2.0632 | +3.2134 | +13 |

The 10506 override selects a known dominated action and adds 59 primitives.
The 3865 override rescues success but loses 9.175 nDTW points and adds 13
primitives; this comparison is mixed and was not a strict training pair.
These are conditional returns from already executed census branches, not a
newly executed learned episode-policy SR/SPL/nDTW result.

The fitting states contain **zero native-successful full continuations**.
They therefore cannot estimate how often an override harms an already
successful native route. This limitation complements, but does not replace,
the separately reported prospective intervention failure.

## Verification

All expected state/action/strict-pair counts match. All admissible native
actions have a returned case; action identities, case validity and frozen
normalization match. All 47 native action returns reproduce the baseline.
Six tests cover protected STOP/budgets, masks and tie-breaking, future-label
invariance, mixed success/quality tradeoffs, numerical ties and pair partitioning.

A separate verifier recomputes pair labels with vector inequalities and choices
with masked sorting: **12,159 pair checks, 47 choices and 564 returned metric
components** all pass. It shares frozen feature extraction and does not claim
an independent simulator reproduction. No replication outcomes enter either
analyzer or verifier.

## Research decision

The earlier inference from strong all-pair accuracy to a promising override
policy was too weak. The relevant positive direction is sparse and entirely
missed even in-sample; aggregate accuracy gains occur elsewhere. Together with
the prior actual intervention failure and local-geometry score attribution,
this closes the current strict-pair linear scorer plus first-disagreement
recipe as a supported method candidate.

This does not establish that graph ranking is unlearnable, that proposal
coverage dominates failures, or that interruption is the remedy. The native
action or another graph action already provides a better return in the observed
counterexamples. No new algorithm, RL/VLM component or predictor change follows
from this audit. Further development needs an independently justified mechanism
and native-relative full-return decision evidence, including successful-route
controls and abstention; all-pair accuracy alone will no longer pass a gate.

## Artifacts

Root: `research/results/preference_training_support_001/`.

- `registration.json`: frozen specification, executable and input hashes.
- `analysis_001/summary.json`: pair partitions and per-route/per-scene counts.
- `analysis_001/states.json`: every native/selected return and component delta.
- `analysis_001/pairs.csv`: all 12,159 pairs, including mixed comparisons.
- `directional_support_001/`: explicit post hoc class-balance split and all
  14 missed native-improving comparisons.
- `verification_001/summary.json`, tests and stdout/stderr: audit evidence.
- `source_provenance.json`, `status.json`, `archive_manifest.json`: original
  simulation settings and current completion/provenance metadata.
