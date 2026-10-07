# Route-disjoint graph preference execution diagnostic

This addendum replaces the infeasible scene-independence requirement for this
user-authorized route-disjoint, scene-overlapping development study. The old
capture configuration references GRAPH_VALUE_CONFIRMATION_PROTOCOL.md; that
frozen configuration is preserved, and it does not establish scene independence.

Sampling 001 was invalid: 12/96 routes overlapped historical ranker diagnostics.
Sampling 002 excludes all 96 exposed pilot routes, five prior cohorts, four
historical ranker logs and both R2R validation splits. It contains 96 routes in
8 training scenes. These are held out from the listed research diagnostics,
not from baseline checkpoint imitation training. No benchmark claim is allowed.

The native baseline and failure census have already been inspected. Thus this
execution rule is an exploratory follow-up, not a prospectively untouched
confirmation. No target full-return label enters model fitting or action choice.
The model is the fixed ridge-10 linear pairwise preference fitted on the previous
fresh cohort: 47 states, 996 actions, 5,668 strict dominance pairs, schema v2.
Normalization uses training actions only. State-constant features cancel in
linear pairwise differences; this experiment cannot demonstrate entropy or
instruction conditioning. The target cohort must never be added to this fit.

## One-intervention rule

For each of all 96 baseline routes, visit native states in time order and choose
the first state where the frozen ranker's top admissible non-STOP action differs
from the native action. If the native action is STOP, forced STOP, or there is no
alternative, leave it untouched. At most one substitution occurs; all subsequent
navigation uses the unchanged navigator through native STOP or 15 decisions.
This rule needs only current graph features and the ranker; recorded state
identity is an exact replay audit aid. It is not an event-triggered policy.

Matched random controls use exactly these same eligible states, with independent
seeds 20261021, 20261022, 20261023. Choose uniformly among the same admissible
non-STOP action set including the original choice. Random choices equal to the
native action are valid abstentions, never replaced with forced alternatives.
Every arm retains all 96 routes in its denominator. No online model retraining,
new sensor, option interruption, reference route or goal information is used.

Before each enabled rollout, run and audit its disabled hook. After rollout,
check every untouched trace, exact pre-intervention prefix and substituted option
against calibrated branch replay; reconstruct final native metrics independently.
Save both stdout and stderr, provenance and immutable schedules.

## Reporting and decision

Report paired changes in SR, SPL, nDTW, primitive count, path length, and high-level
decisions over all 96 routes and treated routes; count success losses and rescues.
Report scene-cluster bootstrap intervals (8 clusters; exploratory) and each random
seed separately. Random seeds are not training seeds or independent test sets.

A GO toward a new prospective study requires learned SR+nDTW rescue in at least
two scenes, positive all-route SR gain, nonnegative mean SPL/nDTW, no mean primitive
increase, and higher SR than the mean of matched random controls. Uncertainty must
be reported; confidence intervals crossing zero preclude a firm performance claim.
If these conditions fail, this linear one-intervention recipe is NO-GO for scaling.
Do not tune it on the present target outcomes or add RL/VLM. Distinguish failure of
this recipe from absence of graph-choice opportunities or proof of proposal failure.
