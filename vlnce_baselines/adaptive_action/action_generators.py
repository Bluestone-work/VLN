"""Controlled waypoint-resolution variants for the first diagnostic cycle."""

from collections import namedtuple

import torch

from vlnce_baselines.waypoint_pred.utils import nms

ActionAbstractionSpec = namedtuple(
    "ActionAbstractionSpec", ["name", "max_predictions", "sigma"]
)

# ``default`` exactly matches the released ETPNav NMS setting. The other two
# variants change only proposal density, keeping the predictor and controller
# fixed. They are diagnostic proxies, not the final adaptive method.
ACTION_ABSTRACTIONS = {
    "default": ActionAbstractionSpec("default", 5, (7.0, 5.0)),
    "coarse": ActionAbstractionSpec("coarse", 3, (12.0, 7.0)),
    "fine": ActionAbstractionSpec("fine", 8, (4.0, 3.0)),
    "dense_native_union": ActionAbstractionSpec("dense_native_union", 12, (4.0, 3.0)),
}


def get_action_abstraction_spec(name):
    key = str(name).lower()
    if key not in ACTION_ABSTRACTIONS:
        raise ValueError(
            "Unknown ACTION_ABSTRACTION={!r}; expected one of {}".format(
                name, sorted(ACTION_ABSTRACTIONS)
            )
        )
    return ACTION_ABSTRACTIONS[key]


def candidate_indices_from_heatmap(heatmap_probs, abstraction):
    """Return ``(angle_idx, distance_idx, score)`` for one state.

    ``heatmap_probs`` must be the unwrapped ``[120, 12]`` probability map
    produced by the waypoint predictor.  The wrap, NMS, and crop operations
    intentionally mirror the released ETPNav implementation, so all levels
    are probed from exactly the same observation and predictor output.
    """
    if not torch.is_tensor(heatmap_probs):
        heatmap_probs = torch.as_tensor(heatmap_probs)
    if heatmap_probs.ndim != 2 or tuple(heatmap_probs.shape) != (120, 12):
        raise ValueError(
            "expected one [120, 12] waypoint heatmap, got {}".format(
                tuple(heatmap_probs.shape)
            )
        )
    spec = get_action_abstraction_spec(abstraction)
    wrapped = torch.cat(
        (heatmap_probs[-1:, :], heatmap_probs, heatmap_probs[:1, :]), dim=0
    )
    output = nms(
        wrapped.unsqueeze(0).unsqueeze(0),
        max_predictions=spec.max_predictions,
        sigma=spec.sigma,
    ).squeeze(0).squeeze(0)[1:-1, :]
    if str(abstraction).lower() == "dense_native_union":
        native = nms(
            wrapped.unsqueeze(0).unsqueeze(0),
            max_predictions=ACTION_ABSTRACTIONS["default"].max_predictions,
            sigma=ACTION_ABSTRACTIONS["default"].sigma,
        ).squeeze(0).squeeze(0)[1:-1, :]
        output = torch.maximum(output, native)
    indices = output.nonzero(as_tuple=False)
    return [
        (int(angle), int(distance), float(output[angle, distance].item()))
        for angle, distance in indices
    ]


def candidate_geometry_from_heatmap(heatmap_probs, abstraction):
    """Convert one heatmap into ETPNav's angle/distance candidate format."""
    indices = candidate_indices_from_heatmap(heatmap_probs, abstraction)
    angles = [2.0 * 3.141592653589793 - angle / 120.0 * 2.0 * 3.141592653589793
              for angle, _, _ in indices]
    distances = [(distance + 1) * 0.25 for _, distance, _ in indices]
    scores = [score for _, _, score in indices]
    return {"angles": angles, "distances": distances, "scores": scores}
