# Semantic Action-Conditioned Capture Protocol

This opt-in protocol replays native ETPNav actions from frozen pre-action states
in isolated simulator branches. The branch restores pose, episode measurements,
and RNG, executes the recorded native controller action, and saves the endpoint
RGB-D observation. It does not alter the live navigator or feed endpoint pixels
back into policy inference.

The selected-action replay must pass exact primitive and collision sequence,
endpoint pose, progress, RNG, terminal flag, and sensor-hash checks before any
semantic comparison is interpreted. A passing replay establishes state/action
fidelity only; it is not evidence that semantic features improve navigation.

The smoke cohort uses two frozen val-unseen episodes and 19 native decisions.
The next expansion may enumerate all admissible native ghosts from the graph
option capture, but must preserve the same branch sentinel and record action
identity, endpoint files, instruction, and raw controller trace for every option.
No future outcome, reference path, or goal distance may be used as an input.
