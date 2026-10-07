"""Privileged diagnostic interventions; never used by a deployed policy."""


def controlled_graph_action(mode, policy_action, best_nonstop_action,
                            best_candidate_action, distance_to_goal):
    """Factor ranking and STOP interventions without changing action masks.

    ``rank_only`` preserves the learned STOP gate, including premature stops.
    ``stop_only`` replaces that gate at 1.5 m, then uses learned non-STOP
    ranking. ``rank_and_stop`` changes both. Missing local proposals fall back
    to the best learned non-STOP action; a globally empty action set falls
    back to STOP in the caller. Legacy behavior is kept explicitly for old runs.
    """
    if mode == "legacy_joint":
        return 0 if distance_to_goal <= 1.5 or best_candidate_action is None else best_candidate_action
    if mode == "rank_only":
        if policy_action == 0:
            return 0
        return best_candidate_action if best_candidate_action is not None else policy_action
    if mode in ("stop_only", "rank_and_stop"):
        if distance_to_goal <= 1.5:
            return 0
        if mode == "rank_and_stop" and best_candidate_action is not None:
            return best_candidate_action
        return best_nonstop_action
    raise ValueError("Unknown graph oracle control: {}".format(mode))
