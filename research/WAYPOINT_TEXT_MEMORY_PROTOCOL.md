# Waypoint Text Memory Protocol

The proposed memory unit is a read-only, action-conditioned side record attached
to a native graph waypoint. For each frozen decision state, capture the RGB-D
view at each available camera bearing (`rgb`, `rgb_30`, ..., `rgb_330`) and retain
the exact native graph action identity, endpoint geometry, instruction, and source
hashes. A description generator, if later added, must be frozen and run offline;
its output cannot use reference paths, goal distance, future collisions, or final
outcomes.

The first feasibility control is not a learned policy. Compare three representations
on scene-disjoint states: native ETP graph features; a geometry-only text control
(bearing/distance words); and frozen instruction-grounded sector descriptions.
All downstream use must be late fusion or a residual action score so the native
Graph Transformer is unchanged.

Success-rate testing requires paired native navigation with the same checkpoint,
seed, route list, controller, STOP mechanism, and decision budget. The semantic
side memory may rerank only native actions and must be gated. Offline endpoint
semantic evidence must first show independent BENEFIT/HARM signal before any online
intervention is permitted.

An “accurate description” claim requires viewpoint-level validation on held-out
images. CLIP similarity alone is only a probe and is not a textual ground-truth
description.
