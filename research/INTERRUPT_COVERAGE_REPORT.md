# Interrupt timing coverage and successful-route exposure

2026-10-08. **CONDITIONAL GO for one bounded timing oracle; NO-GO for learning an
interrupt policy.** The prior first-event pilots leave substantial timing coverage
unmeasured, and collision events also expose many already successful routes.

This retrospective analysis uses the two old training cohorts only. It reads no
alternative-action returns, interruption returns, prospective replication outcomes
or validation traces. There is no model fit or new simulator rollout in this audit.
Rules/code were committed in `21403dd` and input hashes recorded before execution.

| Cohort / native outcome | Routes | Routes with symptoms | Eligible routes | Eligible options | Eligible cuts | Previously tested cuts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original train64 / failed | 9 | 4 | 4 | 28 | 69 | 4 |
| Fresh train64 / failed | 10 | 7 | 7 | 28 | 87 | 4 |
| Original train64 / successful | 55 | 17 | >=15 | >=15 | >=40 | 0 |
| Fresh train64 / successful | 54 | 12 | >=12 | >=28 | >=88 | 0 |

The failure strata have complete phase labels. The historical experiments covered
8/156 cuts (5.13%), 8/56 options (14.29%), and 8/11 eligible failed routes (72.73%).
These denominators describe correlated timing opportunities, not independent
experimental samples. Most extra cuts lie on repeatedly stalled routes. The three
eligible fresh-cohort routes omitted by the four-route cap are 5247, 5648 and 7362
(two scenes); their cut counts are 3, 29 and 7. No improvement is imputed to them.

All enumerated cuts include a forward collision. Stall labels overlap 38 failed
and 7 successful cut records; there are no additional deviation-only midpoint
cuts. Thus this corpus does not support claims about distinct semantic, stall-only
or deviation-only trigger mechanisms.

Successful-route traces lack phase labels. We therefore count eligible cuts only
on empty-back-path actions, where executor construction proves that all primitives
are in the ghost segment. Three successful move options with a nonempty back-path
remain phase-unknown. They are not counted as negative. Empty-back-path eligibility
on failed routes is 26/79 options in train64 and 27/109 in fresh64; successful-route
counterparts are 15/324 and 28/356. These are descriptive conditional strata.

At least 27/109 successful routes encounter a qualifying event. This is exposure
to a possible intervention, **not an observed harmful-interruption rate**, and not
evidence that the symptom is useless. Only paired interventions can measure harm.
The failed ranker and these symptom counts do not establish an interrupt benefit.

## Verification and retained failure

All 217 failed-route control options reproduce native action, primitive, pose,
sensor, RNG and distance records. All 188 empty-back-path failed options have
only ghost phases. Both historical interrupt plans reproduce exactly.

Seven boundary/noninterference tests pass. A separate verifier uses vector geometry
and three-primitive windows rather than the analyzer's running streak, checks 878
known-phase move options and all 284 cut identities/reasons, and directly executes
the AST-isolated historical event selector for both plan checks. No simulator is
imported by either analysis.

Attempt001 incorrectly required a phase entry for Habitat STOP, which bypasses
`wrap_act`. It failed before any result was produced. Attempt002 limits that check
to move actions; eligibility thresholds and population are unchanged. Failed logs,
both registrations, original source commit and a registered-source snapshot are
retained. Do not overwrite attempt001 or call it an experimental negative result.

## Next controlled question

Does varying the collision cut, while keeping navigator/controller/checkpoint
fixed, recover failed routes without damaging successful ones? A bounded follow-up
can use **all 175 known eligible cuts in the fresh training cohort**: 87 cuts on
7 failed routes and 88 on 12 successful routes. At each cut compare matched sensing
and continuation against interruption with consumed versus retained pending ghost.
Use all 19 native route controls, preserve physical prefixes and the 15-decision
budget, and report single-event full returns and first-event harm separately.

This is a retrospective ORACLE timing sweep, not a deployable selector or a
benchmark. Taking a best cut uses privileged outcomes and must remain labeled as
such. Require route-quality rescue in at least two scenes before any new
independent confirmation; even passing that gate does not authorize model training.
If the upper opportunity remains negligible, close this collision-interrupt recipe.

Artifacts: `results/interrupt_coverage_002/analysis_001/{summary,options,cuts,routes}.json`,
`verification_001/summary.json`, registrations, source snapshot and logs.
