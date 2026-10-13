# Waypoint Text Memory Feasibility Report

The requested idea is testable, but the existing ETPNav node does not contain the
required textual evidence. It stores visual and geometric latent features and
graph relations; it does not store per-bearing object descriptions or relations.
The repository also has no validated local caption/VLM checkpoint suitable for
claiming accurate descriptions. The available local CLIP model can rank prompts,
but it cannot produce trustworthy captions or relational statements.

This commit adds an opt-in multi-view capture configuration. The capture saves
all RGB/Depth sensor keys, including bearing-specific views, alongside the exact
native graph state. It does not alter baseline behavior. The next experiment is
to run a frozen description adapter on these views, validate description quality,
and only then test a late-fusion native-action gate.

The success-rate question must therefore be answered in two gates:

1. Does the description side memory distinguish native-action BENEFIT/HARM on
   exact single-action full-return pairs?
2. After an offline gate, does a critical-state gated late-fusion intervention
   improve nDTW/SR without destroying native successes?

Until gate 1 passes, adding text to GraphMap or training a new policy is NO-GO.
