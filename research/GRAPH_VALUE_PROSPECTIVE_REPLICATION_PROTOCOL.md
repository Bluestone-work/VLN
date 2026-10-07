# Prospective route-disjoint replication 001

Freeze before any target capture or counterfactual outcome. This follows the
exploratory 96-route result; that cohort is excluded, not reused for tuning.
No new model family, RL, VLM, waypoint modification or option interruption.

## Population and fixed settings

96 routes, eight scenes, twelve routes per scene; metadata hash seed 20261024.
Exclude all sources in the prior sampling_002 manifest and sampling_002 itself
(the invalid sampling_001 was already excluded). Source R2R train, baseline
released ckpt.iter12000, simulator seed 100, one worker/GPU, normal RGB/depth,
sliding true, control backtracking, STOP semantics and 15-decision horizon.
Scene overlap allowed; this is a research-route holdout, not checkpoint-training
holdout or a validation benchmark. Route audit must pass before capture.
Do not resample because of native success/failure counts.

## Frozen models (only old fresh training labels)

FULL: byte-identical frozen_ranker_fresh.json from execution study 002. Do not
refit or rescale it. Its 47 states / 996 actions / 5,668 strict dominance pairs
came solely from critical_graph_value_fresh/full_002.
COST: same training action features, normalization, strict pairs and ridge=10;
fit only graph_logit, logit_rank_percentile, ghost_distance_m,
back_path_length_m. All other coefficients are exactly zero. This reduced model
still uses old full-return labels; it tests feature sufficiency, not whether
long-horizon labels are necessary. No target labels used for fitting.

## Six intervention arms plus native

1. FULL: substitute at most one non-STOP action at its first disagreement with
   native. Preserve STOP, budget stops and masks; then native continuation.
2. COST-own: identical rule using COST, possibly at a different first state.
3. COST-matched: evaluate COST at FULL's first-disagreement state; include native
   in its action set and abstain if native has highest score. This isolates
   candidate ranking at equal timing. No FULL disagreement means no intervention.
4–6. RANDOM: uniformly sample the same admissible non-STOP set, including native,
   at FULL's first-disagreement state. Seeds 20261031, 20261032, 20261033.
Tie-breaking for deterministic scores: smallest original action index.
All 96 routes remain in every denominator. A route with no changed action is
an exact native control. Zero-change arms are exact no-ops and must be recorded
as such, never forced into a different action. Schedules are generated only
from model scores, current graph masks/geometry/features and episode metadata.
The schedule generator must not read probe outcomes, goal or reference labels.

## Fidelity gates and reporting

Before enabling each nonempty arm, run its disabled hook and audit exact native
metrics/trajectory. Verify actual substitutions against calibrated one-option
replays, untouched routes and every pre-event prefix including sensors/RNG.
Independently reconstruct six final native metrics for every arm. Retain durable
stdout/stderr, configs, sampled IDs, hashes, dirty source diffs and all failed
attempts. Full raw sensor/primitive traces may remain local with hashes.

Primary comparison FULL vs native over all 96 routes: SR, SPL, nDTW and primitive
count. Also report rescue/loss counts, treatment rate, route harms, high-level
count, path and collision events. Secondary FULL vs COST-matched, FULL vs
COST-own and FULL vs mean RANDOM. Scene-cluster paired bootstrap: seed 20261025,
5,000 replicates over eight equal-sized scene clusters. Report each random seed;
these are not independent learned-training replications. No significance claim
from a bootstrap interval alone in eight overlapping training scenes.

Replication point gate: FULL has positive mean SR, nonnegative mean SPL/nDTW,
nonpositive mean primitive delta, SR+nDTW rescues in at least two scenes, and
higher SR than mean RANDOM. Missing support or failed fidelity is not a pass.
If any point gate fails: NO-GO for scaling this recipe; keep all evidence and
stop fitting/tuning it on these targets. A passed point gate with uncertain
quality/cost intervals is CONDITIONAL GO for broader independent evidence.
Even if FULL passes, claim added value over the reduced features only if FULL
beats COST-matched in SR and the paired SR interval excludes zero, with no mean
SPL/nDTW/cost harm. Otherwise feature-mechanism evidence is unresolved or favors
the simpler model; do not equate four features with semantic value reasoning.

No failure-critical census or target label fitting is performed before these
arms complete. Any subsequent failure analysis is explicitly post hoc.
