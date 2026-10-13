# Directional semantic oracle annotation protocol

Protocol frozen: 2026-10-13T13:44:09.617944+00:00

Scene split is frozen in `scene_split.json`; development is used only to test UI and join alignment. Formal test scenes are never used to change ontology.

Sampling uses only frozen pre-action graph options. For each selected state, every unique nearest 30° sector containing at least one admissible non-STOP native action is sampled once. STOP has no image sector. Selection is deterministic by SHA-256 state key, taking the first 12 states per selected scene.

Annotators see one current pre-action RGB sector, the instruction and candidate phrase. They must not see endpoint images, goal/reference route, evaluator metrics, BENEFIT/HARM labels, or native/alternative outcomes.

Labels: landmark multi-select (doorway, door, sofa/couch, painting, table, chair, bed, stairs, hallway/corridor, archway, podium, pew, kitchen, bedroom, other, none, uncertain); navigability (navigable opening, not navigable, uncertain, not applicable); relation (ahead, left, right, behind, doorway/opening, corridor continues, room entrance, landmark cue only, none, uncertain); instruction support (yes/no/uncertain) for visible evidence in the current frame only; confidence (high/medium/low). Temporal completion is never annotated.

Join key: scene_id, episode_id, high_level_step, node_id, bearing_deg, image_sha256.
