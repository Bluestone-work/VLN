# Gate A Dense-Candidate Full-Return Protocol

This is a privileged diagnostic only. A0 is the frozen native ETPNav graph
candidate set. A1 is the union of A0 and the additional candidates selected by
one frozen NMS pass over the same waypoint heatmap (`max_predictions=12`,
`sigma=(4,3)`). Candidate construction reads no reference route, goal,
outcome, or future execution state. The A1 list is frozen before any branch is
executed.

Each branch starts from the captured simulator pose, graph-prefix state and RNG.
The candidate runs through the unchanged native low-level controller and then
the unchanged navigator until native STOP or termination, with the original
15-decision budget and sensors. A0 outcomes come from the previously audited
native full-return branches; only A1 candidates absent from A0 are executed in
this extension. All results are analysis-only and cannot be interpreted as a
deployable policy.
