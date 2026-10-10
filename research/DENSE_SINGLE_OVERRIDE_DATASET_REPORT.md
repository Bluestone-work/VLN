# Dense Single-Override Dataset Report

The frozen A0/A1 val-unseen captures were matched by episode, high-level step, pre-pose, and observation hashes. Newly introduced admissible dense actions were identified by exact action signature difference from A0; the highest A1 logit was selected per matched state. This produces an inventory suitable for a replay runner.

The inventory contains 30 eligible matched states from 17 episodes and 9 scenes. The selected dense-minus-native logit margin has mean -0.7901 (range -3.9215 to 0.6276); this is descriptive only and is not a causal label.

The required causal continuation was not claimed from the available artifacts. Existing dense full-return cases continue with the dense policy after the first action, while the capture traces do not serialize exact simulator and GraphMap snapshots. Treating either as a single override would invalidate the causal label. The generated dataset therefore contains zero valid causal pairs and records every eligible state as a replay failure with reason `no exact simulator/GraphMap snapshot runner executed`.

This is a replay-infrastructure NO-GO, not evidence that BENEFIT or HARM is absent. No future-semantics feature, ranker, safety gate, or RL experiment should be started until a fidelity runner yields control and one-step intervention pairs on the frozen state inventory.
