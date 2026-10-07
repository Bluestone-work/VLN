# Full-return training16 execution protocol

2026-10-07. FULL-RETURN-LABEL-TRAIN16-001. Freeze this protocol and both
schedules before any altered continuation. The metadata sample contains 16
routes over eight new-to-diagnostics training scenes, excluding 35 prior scenes.
The released checkpoint has seen training scenes; this is not held-out testing.

Select one state per route by SHA256(seed|scene|episode|step|state) among native
non-STOP states with at least one valid distinct non-STOP alternative. No local
outcome, goal distance, reference route or final success enters selection. Select
the highest-logit alternative (ties: lower graph index), then a distinct second
alternative by SHA256(20261008|scene|episode|step|graph-ID). If none is available,
retain that route as control for that schedule; do not choose another state.
The schedules contain typed actions and graph identities. Neither continuation
may change STOP, controller, checkpoint, sensors or the original 15 decisions.

Report both actual schedules on all 16 routes and on treated routes separately.
At each sampled state compare the complete native-return vectors against the
baseline, including SR/SPL/nDTW/SDTW, goal error, path and primitive counts,
collisions and native STOP cost. Primary descriptive dominance uses success,
SPL, nDTW (higher better), goal error, path length and primitive count (lower
better), numerical tolerance 1e-6 per continuous metric and exact integer costs.
An alternative dominates only if none worsens and at least one improves. Report
dominated, tied and mixed alternatives without an arbitrary reward scalar.
Do not claim the two-candidate shortlist is an exhaustive oracle upper bound.

Local/H=2 comparisons are diagnostic only. Keep sign disagreements and all
negative outcomes. No learning is authorized by a positive oracle alone. Fewer
than six distinct scenes with dominating alternatives is insufficient support
for a held-out prediction study; do not resample to fill that quota. Even broader
support needs a separate prospective frozen-feature feasibility gate.

Calibration: traced/untraced control, selected sentinels, reversed execution,
independent branch integrity, disabled interceptor, exact event prefix/action,
and native final-metric reconstruction. Save every attempt and source version.
The initial sample asset check used is_dir on .glb paths and failed before output;
only the check was corrected to is_file. Schedule code review caught a missing
step hash argument and candidate duplication before any schedule/continuation.

This cycle's baseline launch left EGL_PLATFORM and CUDA_VISIBLE_DEVICES unset;
the saved simulator/Torch GPU overrides still explicitly choose GPU 0. Preserve
the same launch environment for paired continuations; no cross-cycle latency
comparison is made. Simulator/checkpoint/sensors/evaluator stay unchanged.
