# Preference continuation mechanism audit 001

Date: 2026-10-08. This is a retrospective descriptive audit after the frozen
first-disagreement recipe failed its prospective primitive-cost gate. Its final
navigation outcomes have already been inspected. Freezing this analysis before
extraction does not turn the existing data into a prospective confirmation.

## Question

What actually follows a cheap replacement whose complete navigation costs more?
Separate immediate execution symptoms, subsequent graph navigation, STOP return
cost, and reselection of the displaced native target. These are overlapping
observations, not mutually exclusive causes or counterfactual repairs.

## Population

Use every scheduled FULL/learned intervention in both existing 96-route cohorts:
27 in the earlier exploratory cohort and 29 in the prospective replication.
Include native-successful and failed routes. Report cohorts separately; never
pool their averages into a passed replication. COST-own and COST-matched from
the replication are secondary descriptions with overlapping episodes, not
independent replications. Keep all 96 routes in population-level denominators.
No additional simulator rollout, fitted model, tuned schedule, RL or VLM.

## Frozen extraction

- Reuse the existing exact-prefix audits. Verify actual scheduled actions and
  every primitive prefix again, and tie cost/collision accounting to archived
  final paired metrics. Preserve an empty group rather than omit it.
- Split primitives/movement/collisions into the changed option, subsequent
  non-STOP navigation, subsequent STOP execution, and other actions.
- A cost reversal is a strictly cheaper changed option followed by a strictly
  more expensive full episode. Preserve all other outcome combinations.
- On both native and replacement options, use the previous critical-census
  thresholds: collision count > 0; horizontal target residual >= 0.5 m;
  stall = >= 3 primitives with total traveled distance <= 0.1 m. These are
  post-execution symptoms, not proof that an interruption would rescue a route.
- Find the first subsequent reselection of the displaced native ghost ID. IDs
  are monotonic within a graph, but merged target positions can move. Record
  the full target displacement and call it geometrically stable only when
  3D drift <= 0.25 m (one native forward-step length). A stable ghost ID/target
  is not an identical complete action: starting pose and back-path can differ.
- Record whether this is the immediate next high-level decision, its cost,
  and the first two options' combined cost versus the original native option.
  This unequal-observation/decision-count comparison is descriptive only.
- Record final policy versus forced-budget STOP from native graph logs and
  intervention hook logs. Do not equate STOP return cost with stopping failure.

## Information boundary

Extract pre-action features only from the captured graph row and the two
admissible actions: native/alternative current-proposal flags, shared front
node, empty alternative back-path, native and alternative probabilities,
entropy, step, candidate count and execution-geometry features already used
by the frozen model. Report the four categorical structural strata without
threshold optimization, selecting a best filter, or composing a new policy.

Future collision/residual/stall, target reselection and final metrics live under
separate post-action fields. Goal/reference information does not enter pre-action
features. Do not fit a classifier or estimate held-out prediction accuracy from
these retrospective outcomes. Protected replication routes remain excluded from
future policy fitting/tuning.

## Interpretation gate

This audit cannot grant GO for model training or overturn the failed prospective
gate. A frequent stable immediate reselection motivates a completion/timing
counterfactual; frequent execution symptoms motivate a matched interrupt oracle;
neither observation establishes its remedy. If neither concentrates the adverse
outcomes across both cohorts, report unresolved mechanism rather than inventing
a selector. Any causal follow-up needs its own fixed question and controls.

Record config, source/input hashes, execution Git commit, original simulator
seed/checkpoint provenance, stdout/stderr, per-event rows and central log entry.
