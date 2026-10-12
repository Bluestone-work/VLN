# Semantic Memory Audit

Audited at commit `8a11e50` on `research/adaptive-action-audit-20261007`.

`GraphMap` in `vlnce_baselines/models/graph_utils.py` stores real node position (`node_pos`), the latest/aggregated panorama embedding (`node_embeds`), visit step (`node_stepId`), graph connectivity and weighted shortest paths (`graph_nx`, `shortest_path`, `shortest_dist`), and per-node stop scores (`node_stop_scores`). Ghosts store proposal positions and running means (`ghost_pos`, `ghost_mean_pos`), a running sum/count visual feature (`ghost_embeds`), front-node identities (`ghost_fronts`), optional real positions, and native ghost provenance. Runtime batching adds relative bearing/elevation/distance, graph distance and graph step features.

The graph encoder then combines these visual and geometry features with the current instruction representation and history. This is contextual latent conditioning, not a persistent semantic record. The node object does not explicitly store object or landmark identity, place category, instruction phrase matches, completed sub-instruction, transition semantics such as “pass the sofa” or “enter the kitchen”, or an instruction-progress state. A stop score is stored, but it is a policy output, not semantic evidence.

The frozen A0 traces contain graph positions, action identities, logits, embeddings, instruction text, primitive poses, and RGB/depth observation hashes. They do not contain RGB panorama pixels, a stable simulator viewpoint id for each visited node, or a join from the numeric runtime node id to the CLIP HDF5 key. The repository has CLIP view features, but their viewpoint alignment is not available in these traces. Consequently an instruction-grounded visual semantic label cannot be recovered without rerendering/replaying the simulator or adding a validated panorama capture step.

## Explicit field audit

| Information | Current node/ghost storage | Assessment |
|---|---|---|
| position/orientation geometry | position plus runtime relative geometry; orientation is not persisted on node | partial |
| visual embedding | `node_embeds`, `ghost_embeds` | yes, latent only |
| step id | `node_stepId` for real nodes; ghosts have no step id | partial |
| stop score | `node_stop_scores` | yes, policy output |
| topology connection/distance | NetworkX graph and shortest path/distance | yes |
| ghost aggregation | mean position, running feature sum/count, front list | yes |
| object/landmark identity | none | missing |
| place category | none | missing |
| instruction progress | no explicit field | missing |
| instruction-grounded relation/transition | none | missing |
| contextual language representation | recomputed in model; not stored per node | latent/recomputed |

This audit does not alter the baseline or any dense/A2/single-override path.
