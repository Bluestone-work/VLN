#!/usr/bin/env python3
"""Evaluate simple non-privileged candidate-ranking signals offline."""

import argparse
import json
import math
from pathlib import Path


def mean(values):
    return sum(values) / len(values) if values else None


def rank(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    for position, index in enumerate(order):
        ranks[index] = float(position)
    return ranks


def corr(x, y):
    if len(x) < 2:
        return None
    mx, my = mean(x), mean(y)
    numerator = sum((a - mx) * (b - my) for a, b in zip(x, y))
    denominator = math.sqrt(
        sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y)
    )
    return numerator / denominator if denominator else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("diagnostics", type=Path)
    parser.add_argument("oracle", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    diagnostics = [json.loads(line) for line in args.diagnostics.read_text().splitlines() if line.strip()]
    oracle = [json.loads(line) for line in args.oracle.read_text().splitlines() if line.strip()]
    diag_map = {(row["episode_id"], row["high_level_step"]): row for row in diagnostics}
    oracle_map = {(row["episode_id"], row["high_level_step"]): row for row in oracle}
    signals = {
        "graph_logit": [],
        "heatmap_score": [],
        "short_distance": [],
        "negative_distance": [],
    }
    regrets = {name: [] for name in signals}
    top1_hits = {name: [] for name in signals}
    decisions = 0
    candidate_count = 0
    for key in sorted(set(diag_map) & set(oracle_map)):
        diagnostic = diag_map[key]
        oracle_row = oracle_map[key]
        oracle_candidates = oracle_row.get("levels", {}).get("default", [])
        mapping = {int(item["candidate_index"]): item for item in diagnostic.get("candidate_graph_mapping", [])}
        visual_candidates = {index: item for index, item in enumerate(diagnostic.get("waypoint_candidates", []))}
        items = []
        for index, candidate in enumerate(oracle_candidates):
            item = mapping.get(index)
            if item is None or not item.get("graph_valid") or item.get("graph_visited"):
                continue
            visual = visual_candidates.get(index, {})
            if item.get("graph_logit") is None:
                continue
            items.append({
                "progress": float(candidate["progress"]),
                "graph_logit": float(item["graph_logit"]),
                "heatmap_score": float(visual.get("score", candidate.get("score", 0.0))),
                "distance": float(candidate["distance"]),
            })
        if not items:
            continue
        decisions += 1
        candidate_count += len(items)
        best = max(item["progress"] for item in items)
        signal_values = {
            "graph_logit": [item["graph_logit"] for item in items],
            "heatmap_score": [item["heatmap_score"] for item in items],
            "short_distance": [-item["distance"] for item in items],
            "negative_distance": [item["distance"] for item in items],
        }
        progress = [item["progress"] for item in items]
        for name, values in signal_values.items():
            chosen = max(range(len(values)), key=lambda i: values[i])
            regrets[name].append(best - progress[chosen])
            top1_hits[name].append(float(progress[chosen] >= best - 1e-8))
            signals[name].append((values, progress))
    signal_summary = {}
    for name, grouped in signals.items():
        flat_signal = [value for values, _ in grouped for value in values]
        flat_progress = [value for _, progress in grouped for value in progress]
        rank_signal = [value for values, _ in grouped for value in rank(values)]
        rank_progress = [value for _, progress in grouped for value in rank(progress)]
        signal_summary[name] = {
            "candidate_level_pearson": corr(flat_signal, flat_progress),
            "candidate_level_spearman": corr(rank_signal, rank_progress),
            "top1_best_progress_rate": mean(top1_hits[name]),
            "top1_regret_mean_m": mean(regrets[name]),
        }
    result = {
        "schema_version": 1,
        "num_matched_decisions_with_candidates": decisions,
        "mean_eligible_candidates": candidate_count / decisions if decisions else None,
        "signals": signal_summary,
        "interpretation": {
            "graph_logit": "current learned graph navigator score",
            "heatmap_score": "waypoint NMS heatmap mass, available to the baseline",
            "short_distance": "negative candidate forward distance heuristic",
            "negative_distance": "forward distance with max-selection, included as a sanity control",
            "all_progress_labels": "privileged goal-distance progress; used only for offline analysis",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
