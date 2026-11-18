"""Small, testable pieces of Counterfactual Branch-Relative GRPO (CBR)."""

from dataclasses import dataclass
from typing import Any, Callable, Iterable, List, Sequence
import numpy as np

from .branch_state import capture_branch_state, restore_into


@dataclass
class BranchReturn:
    action: Any
    reward: float
    advantage: float = 0.0
    metadata: dict = None


def select_candidate_actions(actions: Sequence, scores: Sequence,
                             topk: int = 3) -> List:
    """Select deterministic top-k candidates, preserving input order on ties."""
    if len(actions) != len(scores):
        raise ValueError("actions and scores must have equal length")
    if topk <= 0:
        return []
    order = sorted(range(len(actions)), key=lambda i: (-float(scores[i]), i))
    return [actions[i] for i in order[:topk]]


def compute_branch_advantages(returns: Iterable[float], eps: float = 1e-8):
    """Normalize returns within a decision, as required by GRPO."""
    values = np.asarray(list(returns), dtype=np.float64)
    if values.size == 0:
        return np.asarray([], dtype=np.float32)
    mean = values.mean()
    std = values.std()
    return ((values - mean) / (std + eps)).astype(np.float32)


def compute_branch_returns(branch_results: Sequence[BranchReturn]):
    advantages = compute_branch_advantages([x.reward for x in branch_results])
    for item, advantage in zip(branch_results, advantages):
        item.advantage = float(advantage)
    return list(branch_results)


def run_counterfactual_branches(env, gmap, prev_vp, actions,
                                rollout_fn: Callable, max_branches=None):
    """Run candidate branches from one snapshot and verify every reset.

    ``rollout_fn(action, gmap, prev_vp)`` must execute one branch and return a
    ``BranchReturn`` (or a ``(reward, metadata)`` tuple).  The caller owns the
    policy forward pass; this utility only enforces state isolation.
    """
    snapshot = capture_branch_state(env, gmap, prev_vp)
    chosen = list(actions if max_branches is None else actions[:max_branches])
    results = []
    for action in chosen:
        branch_gmap, branch_prev_vp = restore_into(env, snapshot, gmap, prev_vp)
        result = rollout_fn(action, branch_gmap, branch_prev_vp)
        if isinstance(result, BranchReturn):
            results.append(result)
        else:
            reward, metadata = result
            results.append(BranchReturn(action, float(reward), metadata=metadata))
    restore_into(env, snapshot, gmap, prev_vp)
    return compute_branch_returns(results)

