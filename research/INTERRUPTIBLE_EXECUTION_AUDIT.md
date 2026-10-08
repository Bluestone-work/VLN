# Interruptible execution and observability audit

## 2026-10-08 — separate interrupt timing result

The frozen graph-preference recipe remains NO-GO. A separate175-cut collision
oracle now completes544 full rollouts with successful-route controls: consume
has4/7 quality rescues in3 scenes (4/10 of all fresh64 failures), but first-event
consume lowers nDTW on8/12 native-successful routes. No execution-majority or safe
trigger is established. CONDITIONAL GO applies only to independent causal
confirmation, not learning. See `INTERRUPT_TIMING_ORACLE_REPORT.md` and
`INTERRUPT_NEXT_GATE.md`. Earlier entries below describe historical decisions.

Date: 2026-10-07  
Scope: released ETPNav execution loop, research tracing hooks and literature overlap. Read-only audit; no controller or graph code changed.

## What the current executor does

For a selected ghost (`act=4`), `VLNCEDaggerEnv.step` executes the entire action as one environment transition:

1. follow the graph back-path with `multi_step_control`;
2. render an observation with `get_observation_at` after the front-node segment;
3. turn toward the ghost and execute all computed forward primitives;
4. render another observation after the ghost segment, overwriting the front-node observation;
5. return that final observation to the high-level trainer. Execution does not guarantee target arrival.

`single_step_control` calls `wrap_act` for individual turns and forward moves, but ordinary non-video `wrap_act` uses `sim.step_without_obs` and returns no observation. The trainer therefore has no policy decision boundary or visual observation between those primitive actions. Research traces retain primitive poses and collisions after execution; they do not make those observations available to the deployed policy.

For STOP (`act=0`), the executor first follows the selected historical back-path, then sends Habitat STOP. It may return to another node before stopping; when the selected historical node is current, no such return is needed. A near-goal current pose can therefore still produce a failed native STOP return.

## Graph-state transaction problem

In the high-level trainer, `GraphMap.delete_ghost(ghost_vp)` is called before the environment executes the selected ghost, and `prev_vp` is updated to the front node before `envs.step`. If the controller were interrupted in the middle of the option, graph state would already represent a consumed action while the agent would be at an intermediate pose. A valid interruptible executor would need to define, and test, at least:

- whether the consumed ghost is restored, retained or split;
- which real node becomes the current predecessor after partial backtracking;
- how a new observation is encoded into GraphMap;
- how partial primitive cost and collision history are recorded;
- whether the unfinished target remains eligible;
- how already-executed physics, collision accounting and RNG are retained while pending graph/control intent is committed or cancelled. Physical execution must not be silently rewound.

No such transaction exists in the current implementation. The baseline already consumes the ghost without guaranteeing controller success; this is an existing assumption, not a newly discovered interruption bug. Adding a mid-option STOP head without defining partial-action semantics would change graph states and invalidate the current action-identity protocol.

## Observability and novelty

The single primitive-arrival example (unseen route `1593`) is an execution timing symptom, not evidence that the current agent can observe a useful interrupt signal. The deployed high-level policy sees observations only after the macro action completes. A future prototype would need an explicit sensing and compute budget, interruption frequency limit, and graph transaction tests.

The literature audit already found that Beyond Waypoints (arXiv `2606.07244v1`, Sections 4.1–4.2) predicts a state-dependent trajectory length and truncates trajectory candidates before endpoint selection. Trajectory-level planning and progressive refinement also have precedents in DifNav. Therefore a generic “stop a macro action adaptively” claim is not a sufficient novelty position. This is overlap with adaptive duration, not proof of the same online mechanism. The cited Beyond Waypoints section truncates candidate trajectories before execution; it does not establish fresh-observation interruption during execution. A distinct contribution remains unverified; graph bookkeeping alone is an engineering requirement, not a demonstrated scientific contribution.

## Decision

**Do not implement interruptible execution in the current cycle.** The current evidence supports documenting the observability and graph-state bottleneck, but not adding a mid-option controller, STOP head, RL loss or VLM. If the direction is revisited, first specify the boundary between optional pause/resume and a newly issued high-level action, preserve executed state, and require matched sensing/inference budgets. Review the literature before implementing even a prototype. A simulator smoke test would be justified only by a separately supported mechanism, not by this one validation route.

## Verification limits and a possible future control

The checker verifies 13 source-string anchors only. Manual source inspection supports the ordering above; the checker does not prove control flow, dynamic correctness or an interruption implementation. The failed first checker attempt required inherited parent-method text inside the tracing subclass; its source and logs are retained under `preflight_failed_001`. Correcting that checker did not change navigation code. `analysis_002` adds explicit scope fields and an output-directory argument; `analysis_001` is preserved. Source/config/report snapshots, cached-literature checksum verification and baseline-diff comparison are recorded in `results/interruptible_execution_audit/archive_001/cycle_summary.json`.

Any future experiment must distinguish added sensing from added decisions. At a prespecified interval, a sensing-only control would acquire the same sensor observations as an interrupting policy, with the same rendering and encoder budget, while continuing the pending action. A replanning control would use those same observations. Each new navigator call must count against the unchanged 15-decision limit; native STOP, sensors, checkpoint, collision settings and realized physical prefix must remain identical. Count renders, encoder calls and policy calls separately. This is a comparison specification, not an approved scientific hypothesis or an implemented experiment. A deployable trigger and independent evidence of sufficiently frequent recoverable failures are still missing; the single validation case must not become a training label or sampling rule.

Relevant source locations:

- `vlnce_baselines/common/environments.py` (`wrap_act`, `single_step_control`, `multi_step_control`, `VLNCEDaggerEnv.step`);
- `vlnce_baselines/ss_trainer_ETP.py` (ghost consumption and `prev_vp` update);
- `vlnce_baselines/models/graph_utils.py` (`delete_ghost`);
- `research/LITERATURE_METHOD_AUDIT_20261007.md`.
