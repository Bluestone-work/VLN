# Interrupt-event coverage audit

Specified before running the new analysis, 2026-10-08. This is a retrospective
audit of two old training cohorts, not another learned-method experiment.

Question: how much of the eligible interruption timing space did the old
first-event/four-route pilots cover, and do the same symptoms occur on successful
native routes? A failed graph-preference recipe is not positive interrupt evidence.

Use only the original train64 and fresh train64 baseline captures, their native
metrics, failure-control phase traces, frozen critical-state plans and interrupt
plans. Do not read prospective replication outcomes, validation traces, alternative
action returns or interruption returns. No model fitting or simulator execution.

Retain every route and non-STOP option. Report collision, horizontal endpoint
deviation >=0.5 m, and >=3 primitives with total motion <=0.1 m separately and
as a union. These are retrospective symptoms, not attributable failures.

For phase-labeled failed-route controls enumerate every non-final ghost forward
collision and third-or-later consecutive forward displacement <=0.01 m, but only
within symptom-qualified options, matching the old outer eligibility rule. Keep
overlapping collision/stall tags. When an option has no such event, retain the
old deviation-midpoint fallback exactly. Count cuts, options, routes and scenes;
do not treat repeated cuts as independent samples. Reconstruct the original
lexicographic first-event/max-four selection and require exact plan equality.

For successful routes, the original captures lack phase labels. Empty back-path
options have only ghost primitives by executor construction; report their exact
eligibility. Mark other successful options phase-unknown, never infer phases or
treat unknown as negative. Also report the same empty-back-path subset for failed
routes. Verify native capture/control equality and the empty-path phase rule on
all available failed controls. Symptom incidence on success is exposure, not a
measured harmful-intervention rate. Do not estimate trigger precision from these
partial strata or pool scenes as independent.

Decision: this audit cannot grant GO to an interrupt policy. If old pilots cover
only a small part of eligible options/timings, a separately specified timing
oracle may be justified; its controls must include successful-route exposure,
matched sensing and native continuation. If little timing support remains, do
not run a timing sweep merely to search for a positive result. No threshold
search, new labels, RL/VLM or proposal changes follow automatically.

Save input/source hashes before execution, immutable output directories,
stdout/stderr, tests, route/option/cut records and the central research log.
