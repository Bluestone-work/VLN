# Primary-source literature screening — 2026-10-07

Status: title/abstract and metadata screening via the arXiv API. This is **not a
completed full-paper/code novelty audit**. No novelty claim is authorized.
The exact response and retrieval timestamp are archived in
`research/results/ranker_validity/literature_screen_20261007.json`.

Follow-up: `LITERATURE_METHOD_AUDIT_20261007.md` checks primary method sections
and identifies important prior overlap. The table below preserves the earlier
abstract-only screening; full code/supplement coverage remains incomplete.

| Primary source | Representation described in the abstract | Implication for this project |
| --- | --- | --- |
| [ETPNav, 2304.03047v3](https://arxiv.org/abs/2304.03047v3), first posted 2023-04-06 | Predicted waypoints organized into online topology; cross-modal planning and obstacle-aware control. | The local baseline already has a dynamic graph size; dynamic candidate count alone is not a contribution. |
| [Revisiting Topological Graphs for Macro Action based Closed-loop RL, 2609.03906v1](https://arxiv.org/abs/2609.03906v1), 2026-09-03 | Frontier macro actions with a controller, action-aware value head and graph PPO. | A generic pivot to graph-action RL has substantial existing overlap. Do not claim graph PPO or macro actions as new. |
| [DifNav, 2508.09444v1](https://arxiv.org/abs/2508.09444v1), 2025-08-13 | Conditional diffusion unifies generation/planning over future continuous actions, trained with DAgger. | Contrasts with fixed waypoint proposals, but abstract-level screening does not establish a learned switch among action representations. |
| [Beyond Waypoints: A Trajectory-Centric Waypointing Paradigm, 2606.07244v1](https://arxiv.org/abs/2606.07244v1), 2026-06-05 | TSDF-guided diffusion grounds proposals in executable trajectories; trajectory-enhanced navigation. | Proposal/controller consistency already has direct precedent. Label calibration here is an experimental control, not a novelty claim. |
| [AgenticNav, 2606.10577v3](https://arxiv.org/abs/2606.10577v3), first posted 2026-06-09, updated 2026-10-02 | VLM tool interface selects RGB pixels with on-demand depth and memory tools. | Uses a different action interface; cannot assume all recent methods use a fixed predicted-waypoint set. |
| [SparseNav, 2609.26408v1](https://arxiv.org/abs/2609.26408v1), 2026-09-22 | Instruction-conditioned semantic acquisition and hybrid frontier/local directional candidates. | Hybrid action sets and instruction-conditioned computation require careful comparison before claiming an adaptive-space distinction. |

No retrieved abstract explicitly establishes the identical proposed
coarse/default/fine gating experiment. **Absence from these abstracts is not
evidence of novelty or exhaustive coverage.** Full methods, supplementary
material and available code still need checking for representation switching,
adaptive horizon, candidate budgets and recovery actions. Hierarchical VLN,
instruction-conditioned waypoint generation and general dynamic action-space
work remain incomplete parts of that audit.

Search also returned a different paper titled *Beyond Waypoints: Dual-Heatmap
Grounding for Cross-Embodiment Semantic Navigation* (2605.19420). It is not the
trajectory-centric VLN-CE paper above; title ambiguity must not merge their claims.

Decision implication: maintain the current NMS-selector NO-GO. A future
graph-ranking study would need a specific, supported failure mechanism and
fair end-to-end gains; a generic ranking head, graph PPO, or executable
trajectory module is insufficient on this screening alone.
