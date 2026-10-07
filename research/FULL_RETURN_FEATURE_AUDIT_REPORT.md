# Full-return feature audit

Date: 2026-10-07  
Experiment: `FULL-RETURN-FEATURE-AUDIT-001`  
Status: descriptive post-hoc analysis; no model and no rollout

## Purpose

The complete-return pilot produced too few dominating alternatives to justify
training. This audit asks a narrower question: do fields already available to a
deployed graph navigator at the decision state explain the observed final-return
tradeoffs? It does not fit a predictor or select a new threshold.

## Inputs and controls

The analysis reads only the frozen graph records before execution: native logits,
valid/visited masks, normalized logit entropy, graph action geometry, back-path
length, and whether an alternative was a current proposal. It joins these fields
with the already accepted full-return outcomes only for descriptive analysis.
The 16 states, 32 alternatives, candidate identities, source hashes and strict
six-metric dominance rule are unchanged. No goal, reference route, branch result
or final outcome enters a feature.

## Paired candidate comparison

At each state, compare the top-logit alternative with the seeded graph-ID
alternative. Across the 16 paired states:

| Metric, top minus seeded | Mean | Scene bootstrap 95% |
| --- | ---: | ---: |
| Success | +0.0625 | [-0.1250, +0.2500] |
| SPL | +0.1353 | [-0.1005, +0.3531] |
| nDTW | +0.1219 | [-0.0524, +0.3106] |
| Goal distance | -0.1217 m | [-2.8095, +2.6671] |
| Path length | -6.6864 m | [-11.5277, -2.3351] |
| Primitive actions | -47.5 | [-76.69, -21.25] |

The top-logit alternative is usually cheaper than the seeded alternative, but
the success, SPL and nDTW intervals cross zero. The paired outcomes contain 3
dominates, 2 dominated and 11 mixed cases. This is a cost/quality tradeoff, not a
stable target for a binary selector.

## Exploratory associations

Pooled Spearman correlations between pre-action feature differences and final
outcome differences are:

| Feature difference | SPL | nDTW | Path length | Primitive actions |
| --- | ---: | ---: | ---: | ---: |
| Native logit gap | -0.305 | -0.422 | +0.680 | +0.718 |
| Commanded polyline delta | -0.615 | -0.775 | +0.877 | +0.851 |
| Back-path node delta | -0.485 | -0.688 | +0.838 | +0.866 |

These values are descriptive and heavily confounded by the two candidate
construction rules. A longer commanded alternative naturally costs more; this
does not show that shorter actions are better for navigation. The logit relation
changes between the top-logit and seeded subsets, and the sample is only eight
scenes. No significance test, threshold search or fitted ranker is justified.

## Decision

**NO-GO for feature-based selector fitting on this data.** The audit supports
recording explicit action-cost and back-path features in a future prospective
study. It does not support a new adaptive abstraction model, an RL reward, or a
VLM feature. Any future model validation would require independent scenes, a fixed feature
contract and prospective full-return evaluation. The current failed gate is not
bypassed by resampling for positive labels.

Machine-readable output: `research/results/full_return_feature_audit/analysis_001/summary.json`.

## Cost decomposition follow-up

The feature/cost association was decomposed before interpretation. In the
highest-logit schedule, the 16 alternatives add 248 primitive actions: 109 in
the intervened decision and 139 in later navigation. In the seeded schedule,
1,008 additional primitives split into 330 immediate, 643 later navigation and
35 later STOP actions. Primitive and collision totals tie exactly to accepted
native episode metrics.

Commanded polyline difference correlates with later-navigation primitive change
at 0.629 for top-logit and 0.700 for seeded alternatives (0.726 with all later
parts in the seeded group). This does not identify a causal mechanism: candidate
geometry, graph rank, later choices and episode difficulty are entangled. It
only shows that a cost penalty would need to account for continuation, while a
single-step action-cost label would be incomplete.

Output: `research/results/full_return_feature_audit/cost_001/summary.json`.


## Checks and limitations

Five new feature-contract tests pass: privileged-field injection cannot affect
features; rigid coordinate transforms preserve costs; historical-node return
length is counted instead of direct target distance; invalid masks/nonfinite
scores/privileged flags are rejected; entropy is invariant to a logit offset.
Five existing full-return and thirteen intervention/audit tests also pass (23).
All 32 primitive and collision decompositions tie exactly to accepted metrics.
Source compilation and git diff whitespace checks pass. Tracked baseline core
differences are byte-identical to the start-of-cycle snapshot.

The top alternative uses fewer primitives than the seeded alternative on 10/16
routes, not uniformly; it is fully dominated by the seeded alternative on two
routes. The sole dominating alternative versus baseline (2508) occurs at low
normalized entropy (0.0727), with a large native logit gap (3.3236). This single
case cannot define a low-entropy intervention rule, but it warns against assuming
that uncertainty alone would detect the previously observed opportunity.

All correlations use already inspected outcomes; candidates are conditioned on
native logits, two observations share each state, only eight training scenes
are present and only one simulator seed was used. The tests validate analysis
contracts, not model benefit. Raw logs for the cost analysis and tests are saved.
The initial feature command's printed summary was recovered from its JSON and
labelled as recovered, rather than represented as a captured raw stdout file.
