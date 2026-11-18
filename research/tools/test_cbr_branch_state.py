#!/usr/bin/env python
"""Fast deterministic CBR branch-isolation smoke test."""

import random
import numpy as np
import torch

from vlnce_baselines.common.counterfactual_rollout import (
    BranchReturn,
    compute_branch_advantages,
    run_counterfactual_branches,
    select_candidate_actions,
)


class FakeEnv:
    def __init__(self):
        self.position = 0
        self.episode_id = "fake-1"

    def capture_agent_state(self):
        return {"position": self.position, "episode_id": self.episode_id}

    def restore_agent_state(self, state):
        if state["episode_id"] != self.episode_id:
            raise ValueError("episode mismatch")
        self.position = state["position"]


def main():
    env = FakeEnv()
    gmap = {"vp0": {"step": 0}}
    prev_vp = "vp0"
    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)
    expected_rng = (random.random(), float(np.random.rand()), float(torch.rand(1)))
    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)
    actions = select_candidate_actions(["left", "forward", "right"], [0.1, 0.9, 0.5], 2)
    assert actions == ["forward", "right"]

    def rollout(action, branch_gmap, branch_prev_vp):
        assert env.position == 0
        env.position += {"left": -1, "forward": 2, "right": 1}[action]
        branch_gmap["vp0"]["step"] += 1
        random.random()
        np.random.rand()
        torch.rand(1)
        return BranchReturn(action, float(env.position), metadata={"prev_vp": branch_prev_vp})

    before = (env.position, gmap["vp0"]["step"], prev_vp)
    results = run_counterfactual_branches(env, gmap, prev_vp, actions, rollout)
    assert [x.action for x in results] == actions
    assert np.allclose(compute_branch_advantages([2.0, 1.0]), [1.0, -1.0], atol=1e-5)
    assert (env.position, gmap["vp0"]["step"], prev_vp) == before
    assert results[0].advantage > results[1].advantage
    actual_rng = (random.random(), float(np.random.rand()), float(torch.rand(1)))
    assert np.allclose(actual_rng, expected_rng, atol=1e-7)
    print("CBR_BRANCH_STATE_SMOKE_PASS", {"actions": actions, "advantages": [x.advantage for x in results]})


if __name__ == "__main__":
    main()

