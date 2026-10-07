"""Analysis-only oracle for the fixed-resolution go/no-go experiment.

The oracle consumes outcomes from all abstraction levels at the same recorded
state. Those outcomes require privileged simulator access and are therefore
never used by a deployable policy.
"""


def select_oracle_level(level_outcomes, objective="progress"):
    """Select the level with the best privileged one-step outcome.

    ``level_outcomes`` maps a level name to a sequence of candidate dictionaries
    containing ``progress`` and optionally ``reference_progress``. A missing or
    empty sequence is treated as unavailable. Ties are stable in insertion
    order. The caller must generate all levels from the same simulator state.
    """
    best = None
    best_value = None
    for level, candidates in level_outcomes.items():
        if not candidates:
            continue
        key = "reference_progress" if objective == "reference_progress" else "progress"
        value = max(float(candidate.get(key, float("-inf"))) for candidate in candidates)
        if best_value is None or value > best_value:
            best = level
            best_value = value
    return best, best_value
