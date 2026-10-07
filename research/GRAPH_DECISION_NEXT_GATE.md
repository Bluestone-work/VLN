# Next graph-decision evidence gate (2026-10-07)

This is a prospective study design, not permission to expand the architecture
or a claim that graph ranking is novel. The NMS-density AAA hypothesis remains
unsupported. Training64, the ordered-route audit, the seven-route intervention
and the new prospective H=2 study are complete. **Current short-window labels
do not justify a learned scorer.** The next design is outcome-blind state/action
sampling with complete returns; see `FULL_RETURN_LABEL_NEXT_GATE.md`.

## Why a positive branch diagnostic is insufficient

The comparison keeps a baseline state fixed and measures one executed action.
Selecting another action changes the later observations, graph, STOP behavior
and remaining decision budget. Less-negative goal progress can delete useful
backtracking, and a geometric reference proxy cannot verify instruction meaning.
Cost-capped oracle selection uses actual future branch costs, which a deployed
policy does not know. These effects require separate held-out validation.

## Minimal supervised feasibility study, only if labels support it

1. Count material cost-and-route-compatible alternatives by distinct episode
   and scene in **training only**. Repeated decisions from one failed episode
   are correlated, not independent training examples. Do not oversample failure
   episodes after seeing validation results or select favorable scene splits.
2. Freeze scene-based cross-validation using a metadata hash before fitting.
   All decisions/instructions for a route stay together. If positive training
   labels cover too few scenes for meaningful held-out evaluation, expand
   training coverage by the same metadata protocol or stop fitting; report the
   sample limitation instead of relaxing labels until performance looks good.
3. Start with the existing navigator logit as the unchanged baseline. A possible
   first test is a small regularized residual scorer on frozen, already captured
   deployable graph embeddings. Train three fixed seeds when feasible; no new
   vision/language encoder, RL, VLM or multi-loss architecture.
4. Goal distances, reference trajectories, actual branch primitive/path costs,
   counterfactual endpoints and oracle labels are **targets/evaluation data**.
   They may not enter the scorer, candidate mask or selector at held-out
   inference. In particular, do not apply an oracle cost cap at deployment.
5. Report original navigator and learned candidate selection on identical
   state/action sets. The offline test evaluates complete selected branches,
   including incurred cost, route deviations and adverse label changes. Keep
   ties/no-op residual behavior explicit. Weight or cluster uncertainty by
   episode and scene; do not report rows as independent trials.

No model configuration or hyperparameter sweep is selected from unseen66.
The already inspected unseen66 set is an exploratory diagnostic set, not a
pristine final confirmation set. A genuinely prospective test must predeclare
previously uninspected scenes/episodes and preserve their held-out status.

## Completed continuation check and remaining gate

The frozen single intervention per route ran on seven training routes/five
scenes, selected from earlier local outcomes. All 64 routes retained original
order; 57 were unchanged controls. Disabled traces/metrics match exactly;
enabled execution changes exactly seven actions and reproduces their isolated
probes. Native STOP, graph updates and the 15-decision horizon remain intact.
See `SINGLE_INTERVENTION_REPORT.md` for all cases and machine-readable sources.

Treated-route SPL gains 5.5570 pp and nDTW 4.2386 pp, but success stays 4/7,
two failed routes deteriorate, final goal error rises 0.6155 m and total primitive
count rises 674 -> 711. Immediate action costs save 17 primitives; later native
continuation more than consumes that saving. These are privileged post-selected
training outcomes, not a deployable model or held-out improvement.

The subsequently executed test was **prospective comparison of one-step versus short
native-continuation labels on new training routes**. Freeze sampling, horizons,
cost accounting and selection rules before new outcomes, keep all adverse/tied
cases and cluster observations by route/scene. The H=2 reversals on two routes
are hypothesis-generating only; they do not validate a filter. Include native
continuation through STOP and final episode costs when judging label usefulness.
Do not teleport, extend the horizon or replay baseline suffixes after an altered
action. That prospective test now completes in `CONTINUATION_LABEL_REPORT.md`:
five eligible routes / three scenes in 16 new training scenes; H=2 keeps two
efficiency gains but rejects the largest delayed gain. The mixed schedule was
executed and verified; no learned model was trained. Equal-primitive analysis
also leaves harmful acceptance and useful rejection. Do not tune a new horizon
on those same cases. The full-return sampling design remains unexecuted.

## Stop criteria

- Training positives remain sparse/concentrated and do not support a held-out
  supervised comparison: do not integrate a ranker.
- Oracle continuation destroys the apparent local benefit: revisit the label
  objective or stop this pivot before increasing model capacity.
- A small scorer has no consistent benefit across held-out training scenes and
  seeds, or uses more execution cost to obtain it: reject that scorer.
- Any apparent benefit requires privileged test-time filtering or changed
  sensors/controller/STOP/budget: invalidate the comparison.

Only after a controlled learned closed-loop improvement should full held-out
R2R evaluation, broader novelty audit and eventual RxR transfer be considered.
