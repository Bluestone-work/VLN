# Full-return branch alignment diagnostic

At the 16 outcome-blind full-return states, every one of the 32 selected
alternatives had lower one-step goal progress than the native selected action.
This is a property of the frozen candidate schedules, not all graph options.

| Alternative source | Mean one-step progress delta | Mean one-step primitive delta | Mean progress rank | Mean efficiency rank |
| --- | ---: | ---: | ---: | ---: |
| highest nonselected logit | -1.724 m | +6.81 | 3.19 | 3.25 |
| seeded graph-ID | -3.998 m | +20.63 | 5.69 | 6.25 |

There are no cases where a selected alternative is locally better but finally
dominated. There is one case where it is locally worse but finally dominates:
route `2508`. Its advantage over the baseline appears only after later continuation. The
baseline itself reaches the neighborhood first and then departs.
This establishes a sampling limitation, not impossibility of learning: the
shortlist contains delayed rescue evidence but no positive one-step advantage.
The previous NO-GO for fitting remains based on sparse full-return support,
not on this descriptive observation alone.

The diagnostic is not an oracle upper bound. The branch probes are privileged
one-step outcomes, and all candidate alternatives were selected from native
graph logits and masks. No new trajectory or model was run.

Output: `research/results/full_return_branch_alignment/analysis_001/summary.json`.

All 32 one-step primitive deltas match the corresponding actual intervention exactly; goal-progress deltas match within 1e-5 m. Progress-per-primitive rank is exploratory, not a tuned objective.
