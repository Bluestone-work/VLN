# Semantic Node Memory Validation Report

HEAD: `0abb78b4e368ad3807081d3f52bd8680ffb334f2`

The original 44-episode pre-action capture was rerun with the already validated
multi-view sensor path. The rerun has 373/373 identical graph rows and traces:
action dictionaries, graph IDs, visited masks, logits, options, primitive
traces, poses, distances, terminal flags and all 746 original sensor hashes
match exactly. It adds 12 RGB + 12 depth views at every state. The alignment
status is `PASS_WITH_EXPLICIT_PRE_ACTION_TIMING`.

Semantic representation:

- Only visited nodes are stored. Each row has `node_id`, position, place,
  landmark name, bearing/relative direction, relation, ordered instruction
  evidence, a hashed structural embedding, and global confidence.
- Every one of the 373 nodes has 12 RGB bearings (0, 30, ..., 330 degrees) and
  12 depth views. Ghosts do not receive semantic memory.
- The frozen Qwen2-VL-2B extractor is used only offline. Its confidence is not
  calibrated; this run abstained with zero phrase confidence for all nodes.

What is stored:

- `what`: visually proposed place and landmark names;
- `where`: bearing and front/left/right/behind direction when supported;
- navigation relations and one evidence record per ordered instruction phrase;
- `semantic_stop_score` from final-phrase evidence, plus an explicit hashed
  structured embedding.

What is explicitly NOT stored:

- future views, ghost endpoint semantics, reference paths, goal position or
  distance, action/outcome labels, native logits, or post-action observations;
- an assertion that an object was passed, a room was entered, or a turn was
  completed when a still view cannot establish it.

Dataset / cohort:

Episodes: 44
Scenes: 11
Decision nodes: 373
Native-success cohort: 44/44 by construction
Evaluation labels (goal position and outcome) are used only below, never as
semantic features.

## 1. STOP

Native: goal-node Top-1 recall 0.750; MRR 0.843; Spearman with negative
goal-distance 0.683.

Semantic: goal-node Top-1 recall 0.000; MRR 0.153; Spearman 0.000, scene
bootstrap 95% CI [0.000, 0.000].

Fusion: fixed 0.5 percentile fusion, with no outcome tuning, has Top-1 recall
0.750 and MRR 0.843. It does not improve native ranking.

Complementary cases: 0/44 (`native wrong / semantic correct`).

Harm cases: 33/44 (`native correct / semantic wrong`). The semantic score is
an abstaining tie in this run, so this is evidence against using it for STOP,
not evidence of a safe intervention.

CI: semantic scene-bootstrap CI is [0.000, 0.000].

## 2. Instruction progress

Ordered consistency: 0.000 adjacent transition accuracy; 0/44 episodes have
monotonic observed evidence.

Phrase recovery: 0.000 (0 matched phrase steps; 0/44 episodes recover a
phrase). Phrase ordering is generated from instruction text only; no route is
used for decomposition.

Failure examples: episodes `1778`, `413`, and `651` contain ordered text
subgoals, but the extractor returns zero-confidence evidence at every visited
node. There is therefore no defensible claim that the agent saw, passed, or
completed a landmark. The previous generic CLIP probe's 0.0126 STOP correlation
and 47.7% descriptive monotonic rate are not promoted to validated semantic
progress.

## 3. Backtracking

Native retrieval: historical-node MRR 0.759 across 373 prefixes.

Semantic retrieval: MRR 0.395; it improves over the native retrieved node in
28/373 states (7.5%), while its median distance gain is -1.724 m.

Opportunity: the positive fraction is a tie-prone retrieval diagnostic and does
not represent an executed recovery. It does not establish a useful
retrospective nDTW opportunity; no return action was changed.

CI: scene-bootstrap improvement-rate 95% CI [0.051, 0.099], with negative
median gain and no complementary STOP/progress signal. This isolated statistic
is insufficient for deployment.

## 4. Complementarity

| State/episode class | Count |
| --- | ---: |
| Native correct / semantic correct | 0 |
| Native correct / semantic wrong | 33 |
| Native wrong / semantic correct | 0 |
| Native wrong / semantic wrong | 11 |

Semantic evidence is constant zero in this frozen extractor run, so a native
feature probe cannot demonstrate independent prediction. The semantic-vs-native
STOP correlation is 0.000; no valid conditional extra signal is observed in
native-error states.

## 5. Shadow deployment

Does semantic branch alter native trajectory? **NO.**

The multiview capture itself is read-only and the original/native trajectory is
bitwise-equivalent at the recorded action, pose, graph and trace fields. A
semantic shadow rollout was not started because Stage 2/3 had no positive
signal; therefore no action, graph representation, or controller was changed.

## 6. Safety

Native-success retention: 44/44 is the frozen native-success cohort property;
there was no intervention.

Beneficial intervention: 0/44 episode-level STOP ranking cases.

Harmful intervention: 33/44 episode-level STOP ranking cases (0.750).

Risk-coverage: not applicable to an unexecuted action change. The fixed
confidence gate has zero semantic coverage because all extracted phrase
confidence values are zero.

Conclusion:

Explicit 360-degree visited-node storage is now technically aligned and
structurally richer than the native latent node memory, but this frozen semantic
extractor supplies no validated instruction-grounded evidence. The semantic
branch is worse than native STOP/history ranking and has no native-error rescue
cases.

Does explicit 360-degree semantic node memory add information beyond native ETP visual memory?

**NO for this validated extractor and cohort.** The result is a representation
and extractor NO-GO, not a claim that every future semantic model is impossible.

GO / NO-GO:

**NO-GO.** Do not add semantic nodes to the Graph Transformer, train a fusion
head, run PPO/GRPO/RL waypoint selection, change ghost selection, or launch
shadow/intervention deployment from this result. The required complementarity
condition is not met, and the safety condition is not met.

If GO:

Not applicable.

If NO-GO:

Stop this semantic node memory deployment direction until a frozen semantic
extractor demonstrates calibrated, held-out object/bearing/relation evidence.
The current Qwen2-VL run abstains on all phrase evidence; generic CLIP scores
are only probes and cannot fill that gap.

Next single highest-value experiment:

Create a small scene-disjoint held-out viewpoint audit with human labels for
object identity, bearing, doorway/corridor relation and instruction phrase
support. Calibrate and freeze one extractor against that audit, then rerun the
same 373-node offline evaluation. Do not perform any policy intervention unless
that audit and the conditional native-error analysis both pass.

Reproduction artifacts:

- Structured memory: `research/results/semantic_node_memory_features_004/semantic_node_memory.jsonl`
- Multiview capture and alignment: `research/results/semantic_node_memory_multiview_001/`
- Validation: `research/results/semantic_node_memory_validation_006/validation.json`
- Extractor: `research/tools/extract_panorama_node_memory.py`
- Evaluator: `research/tools/validate_semantic_node_memory.py`
- Protocol: `research/SEMANTIC_NODE_MEMORY_PROTOCOL.md`
