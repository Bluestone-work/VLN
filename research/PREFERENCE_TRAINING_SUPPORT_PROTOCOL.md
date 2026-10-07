# Frozen preference training-support audit: next bounded diagnostic

Status: question and inputs specified on 2026-10-08; analysis not yet run.
This is a postmortem of the NO-GO recipe, not authorization to scale it.

## Question

Does high accuracy on strict dominance pairs describe the comparisons the
frozen deployed replacement rule actually makes? The continuation mechanism
audit found geometry-dominated scores and frequent cost reversals without
immediate execution symptoms. Audit the old fitting support before proposing
another objective, feature set or trainable trigger.

## Inputs and boundary

Use only `critical_graph_value_fresh/plan_001.json` and
`critical_graph_value_fresh/full_002/results.jsonl`, their captured graph rows,
and the byte-identical already frozen FULL model. This is the model's own
47-state/996-action development census. It is an **in-sample diagnostic**.
The prospective 96-route replication's outcome files must not be read.
No new rollout, model fit, cross-validation sweep, threshold search or labels.

## Fixed analysis

1. Check complete case coverage and retain all 47 states. Reconstruct all 5,668
   strict pairs using the corrected feature/metric schema. Failure to reproduce
   these counts blocks interpretation and requires an explicit archived fix.
2. Classify every unordered action pair as strict dominance, exact metric tie,
   or mixed/incomparable, using the existing metric tolerance. Independently
   split pairs by inclusion of STOP and inclusion of the actual native action
   index. These splits overlap; do not add their counts as disjoint categories.
3. Score using the existing FULL weights and normalization, never refit. Report
   pair accuracy separately for native-involving move pairs, pairs between two
   non-native moves, and STOP-involving pairs. Keep zero-support groups explicit.
4. At each native decision independently, preserve native STOP and budget stop.
   Otherwise score all currently admissible non-STOP actions with the existing
   tie-break. Use archived full returns for native and selected actions; missing
   outcomes are unsupported, not inferred or replaced by one-step progress.
5. Report selected-vs-native strict improvement, strict degradation, mixed and
   tied outcomes; every component SR/SPL/nDTW/path/primitive/final-error delta;
   whether a better non-STOP option exists; and native abstention counts.
6. Report per route/scene and per critical state. Multiple states from one route
   are correlated. Do not average counterfactual states into a claimed episode
   SR, compose a new navigation policy, or pretend these are first-disagreement
   states sampled prospectively from an unperturbed full navigation run.

## Decision rule

This diagnostic cannot grant GO for new model training. If native-relative
decisions remain mixed/harmful despite high all-pair accuracy, retire all-pair
accuracy as a development success gate for this recipe. If these in-sample
choices are strong, report that distribution/trigger transfer remains unresolved;
the result still cannot overturn the failed prospective intervention gate.
Do not proceed to RL/VLM, waypoint changes, or an interrupt policy from either
outcome. A later causal experiment must separately identify a remediable mechanism.

Config: `configs/PREFERENCE_TRAINING_SUPPORT_001.json`. Save executable-source
hashes before running its eventual analyzer; this document does not claim the
code or the outcome has already been registered or evaluated.
