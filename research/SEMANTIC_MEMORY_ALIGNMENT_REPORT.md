# Semantic Memory Alignment Report

This report audits the opt-in RGB-D capture from `semantic_memory_alignment_smoke_001`.
The capture used the released ETPNav checkpoint, the frozen val-unseen replication dataset,
seed 100, and four episodes. No policy, graph encoder, waypoint predictor, controller, or
evaluator behavior was changed.

The graph capture contains 34 decision rows and 34 matching trace keys. Every decision row
contains one RGB and one depth `.npy` file. The audit verifies the file SHA-256 values and
the episode, high-level step, current viewpoint, and graph-id metadata. The capture also
reports `privileged_labels_in_features=false` for every row.

The semantic observation is intentionally collected before the current action is executed.
The existing trace schema records observation hashes after that action. Consequently, those
two hashes are not expected to match and must not be used as a false same-timestep check.
This is a temporal distinction, not a file-integrity failure. A future semantic extractor
must use the RGB-D file attached to the graph decision row as its state observation and must
not consume post-action trace observations or outcome fields.

The result is **alignment PASS for analysis-only pre-action capture**, with the explicit
limitation that this smoke does not yet provide validated object or relation semantics.
The next experiment is therefore offline visual-language extraction from these captured
panoramas, with model availability and image/viewpoint alignment recorded before any
STOP, progress, or native-ghost claim is made.
