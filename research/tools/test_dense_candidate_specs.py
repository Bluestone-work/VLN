import numpy as np
import torch

from vlnce_baselines.adaptive_action.dense_candidate_specs import candidate_sets_from_heatmap


def test_dense_sets_are_nested_and_deterministic():
    rng = np.random.RandomState(7)
    heatmap = torch.as_tensor(rng.rand(120, 12), dtype=torch.float32)
    first = candidate_sets_from_heatmap(heatmap)
    second = candidate_sets_from_heatmap(heatmap)
    assert first == second
    assert first["nested_a0_in_a1"]
    a0 = {(x["angle_index"], x["distance_index"]) for x in first["a0"]}
    a1 = {(x["angle_index"], x["distance_index"]) for x in first["a1"]}
    assert a0.issubset(a1)
    assert first["new_a1_count"] == len(a1 - a0)
