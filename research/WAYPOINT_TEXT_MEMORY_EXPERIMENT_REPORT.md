# Waypoint Text Memory Experiment Report

The proposed idea is: for each graph waypoint, retain a bearing-indexed textual
description such as “at 30 degrees, a doorway is visible” and use it to improve
native action selection. The current repository can now capture the required
multi-view input without changing ETPNav. On the two-episode smoke, there are 19
decision rows and 24 sensors per row (12 RGB plus 12 depth), for 456 files. The
multi-view audit checks every file hash and graph-state key.

This validates the data path, not the navigation hypothesis. An accurate text
description requires a frozen caption/vision-language model and held-out
viewpoint validation. CLIP prompt similarity is insufficient because it does not
reliably express relations, bearing, occlusion, or instruction completion.

The success-rate experiment is preregistered as four paired arms:

* A0: released ETPNav native graph policy;
* A1: geometry-text control containing only bearing and distance words;
* A2: frozen per-bearing visual descriptions;
* A3: A2 plus instruction-grounded relevance.

All arms keep the native graph and controller fixed. Text is a read-only side
memory and enters only through a late residual action score. Training scenes and
validation scenes are disjoint. Before online navigation, endpoint observations
for the same state and every native ghost must be joined to exact single-action
full-return outcomes and labeled BENEFIT/HARM/UNCERTAIN. The current causal pair
file is empty, so a success-rate claim cannot yet be made.

The current decision is **capture PASS; efficacy untested; model change NO-GO**.
The next experiment is to generate the missing exact native-action full-return
pairs, then test whether A2/A3 separates beneficial from harmful actions before
running any navigation intervention.
