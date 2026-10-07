# Failure-critical full-return census and interrupt oracle

Registered 2026-10-07 after explicit user direction. Prior coarse/default/fine
work remains a negative diagnostic; no selector, PPO or VLM is trained here.

## Population and intervention

Use all nine baseline failures in the existing train64 cohort (seed 100).
Validation66 is descriptive only in this cycle. Do not select normal states or
select candidates by favorable branch outcomes. Critical states are the union
of all non-STOP decisions with collision >=1, horizontal target deviation >=0.5m,
or >=3 primitives with <=0.1m motion; the last non-STOP decision; and terminal
STOP. Deduplicate states. These are retrospective failure/symptom definitions,
not a deployable trigger. Every admissible native graph action is tested at each
critical state, including STOP and the selected action. Forced-horizon masks
remain unchanged. Full return means a single action intervention followed by the
original navigator to native termination within the original 15 decisions.

Replay each episode from its original reset to reconstruct language, observations,
topology and STOP history. Require exact graph IDs, masks, logits, selected
full-action dictionaries and controller-prefix reproduction before intervention.
Reuse the checkpoint in memory but recreate the graph each rollout. Controls must
reproduce all recorded native episode metrics. Independently reconstruct final
SR, SPL, nDTW, SDTW, distance and length from actual primitive paths.

## Small interrupt oracle

From the failed training routes with an execution symptom, take the earliest
qualifying ghost-segment event per route, at most four routes in episode-ID order.
Only test cuts within the final ghost-control segment, after any backtracking,
and before its last primitive; backtracking interrupts are out of scope. Event
is the first collision, or third consecutive forward displacement <=0.01m.
For deviation-only options, a midpoint with residual greater than nominal
remaining forward distance +0.5m is eligible. Eligibility uses recorded executed
outcomes; this is ORACLE analysis, not a learned/deployable trigger.

At a frozen primitive cut, compare (a) native continuation, (b) acquire the same
sensor panorama then continue the pending action, (c) acquire that panorama,
cancel the remaining option, return the actual intermediate observation and
invoke the unchanged navigator at the next decision. Native consumption versus
restoring the pending ghost are separately labeled controls. Restoring a pending
ghost does not rewind physics, measures, collisions or RNG. Preserve the actual
prefix; reuse the valid front predecessor because cuts after backtracking only
are eligible. Every navigator invocation counts toward 15. Record sensor renders,
primitive counts, policy calls, graph restoration and full-episode outcomes.
Extra sensing is disclosed and matched by the sensing-only arm.

## Interpretation and decision gate

Report route-level overlapping rescue sets, state-level action coverage and
all returns. Replacing a non-STOP with a successful native alternative supports
ranking opportunity under this continuation policy. Replacing native STOP with
a successful continuation supports a termination opportunity. Interruption rescue
relative to both native and sensing-only controls supports interrupt/replanning
opportunity, not a pure low-level controller diagnosis.

No rescue under all native options does NOT prove proposal failure: continuation
selection, termination, instruction errors and untested states remain possible.
Proposal coverage needs a separate validated reachable-action oracle; execution
needs matched-target controls. Keep unresolved/mixed categories explicitly.
No causal partition is forced. A direction becomes CONDITIONAL GO only with at
least two rescued routes in distinct scenes and valid controls. A majority claim
requires >50% of the nine failed routes with attributable evidence, including
unresolved cases in the denominator; overlapping evidence yields MIXED. A small
interrupt pilot can justify a larger registered confirmation but not a majority
or deployable-performance claim. All gates failing means NO-GO for learning;
report insufficient coverage separately from a negative result.
