"""Frozen candidate specifications for Gate A proposal-coverage diagnostics.

This module is deliberately separate from the coarse/default/fine diagnostic
selector.  A1 is constructed as a union containing the exact native A0
indices and additional proposals from one frozen denser NMS pass.  The union
is formed before any simulator outcome is inspected, so A0 is guaranteed to
be nested in A1.
"""

import math

import torch

from vlnce_baselines.waypoint_pred.utils import nms


A0_MAX_PREDICTIONS = 5
A0_SIGMA = (7.0, 5.0)
A1_MAX_PREDICTIONS = 12
A1_SIGMA = (4.0, 3.0)


def _validate_heatmap(heatmap):
    if not torch.is_tensor(heatmap):
        heatmap = torch.as_tensor(heatmap)
    if heatmap.ndim != 2 or tuple(heatmap.shape) != (120, 12):
        raise ValueError("expected [120, 12] heatmap, got {}".format(tuple(heatmap.shape)))
    return heatmap.detach().cpu()


def _nms_indices(heatmap, max_predictions, sigma):
    wrapped = torch.cat((heatmap[-1:, :], heatmap, heatmap[:1, :]), dim=0)
    output = nms(
        wrapped.unsqueeze(0).unsqueeze(0),
        max_predictions=max_predictions,
        sigma=sigma,
    ).squeeze(0).squeeze(0)[1:-1, :]
    return {
        (int(angle), int(distance)): float(output[angle, distance].item())
        for angle, distance in output.nonzero(as_tuple=False)
    }


def candidate_sets_from_heatmap(heatmap):
    """Return frozen nested A0/A1 index sets and provenance metadata."""
    heatmap = _validate_heatmap(heatmap)
    native = _nms_indices(heatmap, A0_MAX_PREDICTIONS, A0_SIGMA)
    dense = _nms_indices(heatmap, A1_MAX_PREDICTIONS, A1_SIGMA)
    # Keep native proposals exactly and add only new same-heatmap proposals.
    union = dict(dense)
    union.update(native)
    ordered = sorted(union, key=lambda item: (-union[item], item[0], item[1]))
    native_ordered = sorted(native, key=lambda item: (-native[item], item[0], item[1]))
    dense_ordered = sorted(dense, key=lambda item: (-dense[item], item[0], item[1]))

    def pack(items, scores):
        return [
            {"angle_index": a, "distance_index": d, "score": float(scores[(a, d)])}
            for a, d in items
        ]

    return {
        "spec_version": 1,
        "a0": pack(native_ordered, native),
        "a1": pack(ordered, union),
        "a1_dense_pass": pack(dense_ordered, dense),
        "nested_a0_in_a1": set(native).issubset(set(union)),
        "a0_count": len(native),
        "a1_count": len(union),
        "new_a1_count": len(set(union) - set(native)),
        "a0_nms": {"max_predictions": A0_MAX_PREDICTIONS, "sigma": list(A0_SIGMA)},
        "a1_nms": {"max_predictions": A1_MAX_PREDICTIONS, "sigma": list(A1_SIGMA)},
    }


def geometry_for_entries(entries):
    """Convert provenance entries to the ETPNav angle/forward convention."""
    return [
        {
            "angle": 2.0 * math.pi - e["angle_index"] / 120.0 * 2.0 * math.pi,
            "distance": (e["distance_index"] + 1) * 0.25,
            "score": e["score"],
            "angle_index": e["angle_index"],
            "distance_index": e["distance_index"],
        }
        for e in entries
    ]
