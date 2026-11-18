"""Native-anchored admission for transactional candidate expansion.

This module is deliberately policy-agnostic and has no simulator or training
dependencies.  It is a research prototype for TETP (Transactional Evidence
Topological Planning): unexecuted proposals are scored as queries against the
committed native memory, then admitted only when they remain supported by the
native action manifold and stable under equivalent query passes.

The gate does not use goal distance, reference paths, outcomes, or labels.  It
returns the native action whenever the certificate is not satisfied.
"""
from __future__ import absolute_import

import math


DEFAULT_GATE = {
    # The native-memory invariance tolerance is intentionally tight: this is a
    # certificate threshold, not a learned confidence score.
    "max_native_drift": 1.0e-4,
    # A candidate must have local support from at least one native action.
    "min_native_support": 0.50,
    # Candidate score variation across equivalent query passes is treated as
    # epistemic uncertainty and blocks admission when too large.
    "max_candidate_std": 0.25,
    # A new query is not allowed to win merely by an arbitrarily large OOD
    # logit excursion above the native score envelope.
    "max_score_excess": 0.50,
}


def _argmax(values):
    if not values:
        raise ValueError("argmax requires a non-empty sequence")
    return max(range(len(values)), key=lambda i: (float(values[i]), -i))


def _mean(values):
    return sum(float(x) for x in values) / float(len(values))


def _std(values):
    if len(values) <= 1:
        return 0.0
    mean = _mean(values)
    return math.sqrt(sum((float(x) - mean) ** 2 for x in values) / float(len(values)))


def _cosine(a, b):
    if len(a) != len(b) or not a:
        raise ValueError("embedding dimensions must match and be non-empty")
    aa = sum(float(x) * float(x) for x in a)
    bb = sum(float(x) * float(x) for x in b)
    if aa <= 0.0 or bb <= 0.0:
        return 0.0
    return sum(float(x) * float(y) for x, y in zip(a, b)) / math.sqrt(aa * bb)


def nearest_native_support(candidate_embedding, native_embeddings):
    """Return cosine support in the committed native action set."""
    if not native_embeddings:
        raise ValueError("native_embeddings must be non-empty")
    return max(_cosine(candidate_embedding, x) for x in native_embeddings)


def admit_candidate(
    native_logits,
    native_embeddings,
    candidate_logits,
    candidate_embeddings,
    candidate_score_passes,
    native_drift,
    config=None,
):
    """Choose at most one candidate, or abstain to the native action.

    Args:
        native_logits: scores for the committed native action set.
        native_embeddings: embeddings for the same native actions.
        candidate_logits: scores for uncommitted proposal queries.
        candidate_embeddings: embeddings for those proposal queries.
        candidate_score_passes: list of score lists, one per equivalent query
            pass.  Every row must have ``len(candidate_logits)`` values.
        native_drift: max absolute native-logit difference between the native
            memory-only and candidate-query passes.
        config: optional thresholds overriding DEFAULT_GATE.

    Returns a JSON-serializable dictionary with the selected source and a
    reason for every rejected candidate.  ``selected_index`` is an index in
    the concatenated action list: native actions first, candidates second.
    """
    if not native_logits or not native_embeddings:
        raise ValueError("native action set must be non-empty")
    if len(native_logits) != len(native_embeddings):
        raise ValueError("native logits/embeddings length mismatch")
    if len(candidate_logits) != len(candidate_embeddings):
        raise ValueError("candidate logits/embeddings length mismatch")
    if len(candidate_score_passes) == 0:
        raise ValueError("at least one candidate score pass is required")
    for row in candidate_score_passes:
        if len(row) != len(candidate_logits):
            raise ValueError("candidate score pass length mismatch")

    cfg = dict(DEFAULT_GATE)
    if config:
        cfg.update(config)
    drift_mode = str(cfg.get("native_drift_mode", "hard"))
    if drift_mode not in ("hard", "audit"):
        raise ValueError("native_drift_mode must be 'hard' or 'audit'")

    native_index = _argmax(native_logits)
    native_top = float(native_logits[native_index])
    records = []
    native_ok = float(native_drift) <= float(cfg["max_native_drift"])

    for i, score in enumerate(candidate_logits):
        passes = [float(row[i]) for row in candidate_score_passes]
        support = nearest_native_support(candidate_embeddings[i], native_embeddings)
        std = _std(passes)
        excess = float(score) - native_top
        reasons = []
        if not native_ok and drift_mode == "hard":
            reasons.append("native_memory_drift")
        if support < float(cfg["min_native_support"]):
            reasons.append("insufficient_native_support")
        if std > float(cfg["max_candidate_std"]):
            reasons.append("unstable_candidate_score")
        if excess > float(cfg["max_score_excess"]):
            reasons.append("ood_score_excess")
        records.append({
            "candidate_index": i,
            "score": float(score),
            "score_passes": passes,
            "score_mean": _mean(passes),
            "score_std": std,
            "native_support": support,
            "score_excess": excess,
            "native_memory_pass": native_ok,
            "native_drift_advisory": (not native_ok and drift_mode == "audit"),
            "support_pass": support >= float(cfg["min_native_support"]),
            "candidate_stability_pass": std <= float(cfg["max_candidate_std"]),
            "score_envelope_pass": excess <= float(cfg["max_score_excess"]),
            "admissible": not reasons,
            "reasons": reasons,
        })

    admissible = [r for r in records if r["admissible"]]
    if admissible:
        # The native anchor remains the reference.  A candidate must actually
        # beat it after passing the certificate; otherwise abstain.
        winner = max(admissible, key=lambda r: (r["score"], -r["candidate_index"]))
        if winner["score"] > native_top:
            selected_index = len(native_logits) + int(winner["candidate_index"])
            source = "candidate"
            reason = "certified_candidate_beats_native"
        else:
            selected_index = native_index
            source = "native"
            reason = "certified_candidates_do_not_beat_native"
    else:
        selected_index = native_index
        source = "native"
        reason = "no_candidate_passed_certificate"

    return {
        "selected_index": int(selected_index),
        "selected_source": source,
        "reason": reason,
        "native_index": int(native_index),
        "native_top_score": native_top,
        "native_drift": float(native_drift),
        "native_memory_certificate": native_ok,
        "native_drift_mode": drift_mode,
        "native_drift_advisory": (not native_ok and drift_mode == "audit"),
        "candidates": records,
        "config": cfg,
    }
