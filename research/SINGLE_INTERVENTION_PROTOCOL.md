# Single-action continuation experiment (2026-10-07)

The seven-route schedule was frozen in the preceding cycle, before any
continuation outcomes: `configs/GRAPH_SINGLE_INTERVENTION_TRAIN7_001.json`.
It selects one metadata-hashed material cost-and-route-compatible alternative
per affected training route. This is a targeted, post-selection oracle
diagnostic, not an unbiased benchmark sample or a deployable model.

## Execution

Keep the original 64 training routes in their original simulator order, original
seed 100, checkpoint, sensors, controller, STOP and 15-decision horizon. First
run the new interceptor disabled. Require all original per-episode/aggregate
metrics and all 474 action traces to match the archived baseline. Changing only
the trainer registration and a guarded callback must not change native actions.

After that gate passes, enable exactly the seven scheduled non-STOP replacements.
Check graph IDs, masks, logits, typed positions and original full-action identity
against the archive at every state before a route's intervention. At the event,
check the alternate typed full action before modifying only the graph index.
Let the existing builder update the front node and delete the selected ghost.
All later decisions, observations, topology and STOP choices come from the
unchanged navigator's actual continuation. Do not teleport, restore simulator
state, substitute a baseline suffix, add an action or extend the decision budget.

The research-only hook reads privileged precomputed event choices; it is not
a learned abstraction policy. The default runner never imports this hook.

## Fidelity and analysis gates

- Disabled run: all 64 episodes, metrics, actions, poses, RNG states, primitive
  collision sequences and observation hashes match the archive.
- Enabled run: every pre-intervention prefix matches; exactly one intervention
  per selected route; no intervention in the other 57; all those 57 complete
  trajectories and metrics must remain identical.
- Each intervention execution must reproduce its previously measured isolated
  alternative's primitives, endpoint, goal progress and observation hashes.
- Verify native STOP gating and ghost deletion throughout, no missing episodes,
  and no trajectory exceeds 15 decisions. Preserve all failed attempts.

Report paired SR, SPL, nDTW, SDTW, goal distance, path length, high-level decisions,
primitive steps and collision counts: individually for all seven routes, their
mean, and the complete 64-route population. Report successes rescued and lost,
positive/negative/tied paired outcomes and execution cost. A local oracle gain
is not automatically an episode gain. No favorable route may be removed.

All results are descriptive from one frozen simulator seed and five selected
training scenes. Do not use a bootstrap or the unchanged 57 routes to inflate
the effective intervention sample size. No trainable model, extra losses, RL,
VLM, benchmark expansion or hyperparameter selection is part of this experiment.
If local benefit disappears under native continuation, reject these labels as
sufficient justification for scorer training and report the negative result.
