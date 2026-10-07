# Next gate: outcome-blind full-return labels

Date: 2026-10-07. Status: design only; no new sample or experiment executed.
This follows `CONTINUATION_LABEL_REPORT.md`. It is not a new adaptive abstraction
model and does not reopen the NMS-density AAA claim.

## Why change the question

H=1 gains can reverse; H=2 rejects both harmful and useful actions; equal
primitive budgets change decisions without resolving those errors. New material
labels occur on only five successful routes / three scenes. Picking states
because local goal progress is already positive creates a restricted sample.
Some large savings emerge only through later navigation and native STOP returns.

The next question is whether complete native-continuation outcomes provide
consistent action preferences on states selected **without oracle outcomes**.
This should be resolved before fitting a scorer or inventing a longer horizon.

## Minimal prospective design to freeze before execution

1. Select a small fixed training-only scene/route sample by metadata hash,
   excluding previously inspected routes and documenting checkpoint training
   exposure. Choose one baseline non-STOP state per route by a frozen hash,
   regardless of local progress, baseline success, uncertainty or final return.
   Retain routes with no eligible state explicitly; do not replace them.
2. Retain the native action plus at most two valid alternatives selected using
   information already available to the agent: the highest nonselected graph
   logit and a seeded graph-ID sample of another valid option. Freeze tie and
   duplicate handling. This is limited candidate coverage, not an optimal oracle.
3. Reuse the calibrated full-action representation, native graph update/ghost
   consumption, sensors, controller, STOP and 15-decision horizon. Disabled
   controls, exact event prefixes and isolated full-option execution checks
   remain prerequisites. Run each alternative in its own actual continuation;
   never splice in a baseline suffix after an altered action.
4. Record final SR/SPL/nDTW/SDTW, goal error, primitive/path cost, collisions,
   STOP return cost and the entire observed outcome vector. Do not collapse
   conflicting metrics into an arbitrary newly tuned reward. Describe dominated,
   improved and mixed-outcome alternatives separately with explicit tolerances.
5. Report the local/H=2 labels on these same states only as diagnostic comparisons.
   Preserve bad alternatives, ties and failures. Cluster by route/scene; use
   at least three training seeds only if a later learned model is justified.

## Go/no-go

If useful full-return alternatives are sparse, concentrated or unstable under
controls, stop this label-distillation direction. If benefits exist only under
privileged future selection, that alone is not a deployable result. If there is
adequate distributed support, the following gate is a small held-out
frozen-feature prediction test with no privileged inputs and no RL/VLM.

No sample size, candidate thresholds, reward weights or learned model are
declared validated by this design. Freeze the actual config before collecting
its outcomes; do not promote the present twelve post-selected interventions as
its prospective test set. The current user authorizes continued evidence-gated
research without asking for permission at each round.
