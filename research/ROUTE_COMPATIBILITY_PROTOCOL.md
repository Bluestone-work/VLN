# Frozen protocol: ordered reference-route diagnostic (2026-10-07)

Question: does the previously observed cost-controlled local goal opportunity
survive a check against the ordered instruction reference route?

This protocol is specified before computing route-compatibility outcomes. Prior
goal-distance outcomes are known; this is a follow-up diagnostic, not a blinded
confirmatory study. No deployable model is trained or tuned on validation data.

## Training coverage

`GRAPH_OPTION_TRAIN64_001.json` fixes 16 metadata-hashed training scenes and four
distinct routes per scene, one metadata-hashed instruction per route. Exclude
all eight pilot routes before selection. Seed 20261007 selects metadata only;
simulator seed remains 100. Require zero route overlap with either validation
split, and check scene assets before starting. Do not oversample failed routes.
Use the existing capture, untraced control, complete-option probe, selected
sentinel, reversed-order and independent integrity gates without modification.

## Paths and integrity

Reference R is the evaluator's `{split}_gt.json.gz[episode_id]['locations']`,
not the sparse `reference_path` episode field. Construct the actual baseline
prefix P from episode start and every captured primitive pose, removing only
exact consecutive position duplicates, as `Position` does. Include the native
STOP/backtracking path. Recompute full baseline path length and original
fastdtw nDTW/SDTW from traces and compare to native saved metrics (1e-6).
This equality is a data-integrity gate, not a redefinition of evaluation.

## Ordered prefix diagnostic

Use exact cumulative Euclidean DTW for the diagnostic only:

    D(i,j) = ||P_i - R_j|| + min(D(i-1,j), D(i,j-1), D(i-1,j-1))

Start at R_0; endpoints are open. At each actual decision, the reference cursor
c is the smallest endpoint j >= the previous cursor minimizing D(last,j).
The cursor starts at zero and only follows the actual selected trajectory;
no counterfactual branch can update a later baseline prefix or cursor.

For each alternative b, extend a COPY of the same prefix DP row using its full
executed motion path. Its open endpoint j_b is the smallest minimizer over j>=c.
Let s be the actually selected action and j_s its open endpoint. Define a
conservative, explicitly named **matched-prefix route gate**:

1. j_b >= j_s (no smaller matched ordered reference prefix);
2. D_b(last,j_s) <= D_s(last,j_s) + 1e-5 (no worse cumulative alignment to the
   SAME selected reference endpoint).

The selected action always passes. Keep goal progress and path/primitive cost
caps exactly as in the preceding study. Report best goal-progress gain among
joint-cost-capped options passing the gate, and how many previously material
states (>=0.25 m) remain. Deterministic ties prefer the selected action then
lower graph index. Save every candidate's endpoint, reference arc length,
matched-endpoint cost, gate results and physical costs.

Secondary, predeclared sensitivity: also report the route-endpoint-only gate
without the matched-endpoint cost condition, plus endpoint and cost conditions
separately for the original joint-capped goal-best action. Neither sensitivity
replaces the primary result. No tolerance search or validation-driven tuning.

## Interpretation and limitations

This is a privileged geometric route-compatibility proxy, not semantic
instruction verification and not future SR/SPL/nDTW. The reference is sampled;
Euclidean DTW cannot establish wall/topology consistency by itself. Accumulated
prefix misalignment and monotonic cursor commitment can hide recovery actions.
Comparing to the selected endpoint can conservatively penalize an alternative
that travels farther along the route. Sample count affects cumulative DTW.
The joint path/primitive caps also reject useful but more expensive detours.
Report these limitations even if the result is favorable. No counterfactual
continuation exists, so do not score branch prefixes as full episode nDTW.

Report scene/episode clustered descriptive intervals and representative states;
no bootstrap sample is an independent training seed. If training data still
lacks material, route-consistent alternatives, do not fit a scorer. Even with
signal, a future training-only holdout study is required before integration.
