#!/usr/bin/env python3
"""Train a tiny non-privileged candidate ranker from oracle labels.

The privileged progress label is used only to fit the offline diagnostic. The
feature set contains quantities available to the deployed default navigator.
This script does not modify the navigation policy.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np


FEATURE_NAMES = (
    "graph_logit", "heatmap_score", "distance", "sin_angle", "cos_angle",
    "angular_dispersion", "policy_entropy", "candidate_count",
)


def load_rows(diag_path, oracle_path):
    diagnostics = [json.loads(line) for line in Path(diag_path).read_text().splitlines() if line.strip()]
    oracle = [json.loads(line) for line in Path(oracle_path).read_text().splitlines() if line.strip()]
    diag_map = {(row["episode_id"], row["high_level_step"]): row for row in diagnostics}
    oracle_map = {(row["episode_id"], row["high_level_step"]): row for row in oracle}
    records = []
    for key in sorted(set(diag_map) & set(oracle_map)):
        diagnostic = diag_map[key]
        oracle_row = oracle_map[key]
        visual = {index: item for index, item in enumerate(diagnostic.get("waypoint_candidates", []))}
        mapping = {int(item["candidate_index"]): item for item in diagnostic.get("candidate_graph_mapping", [])}
        candidates = oracle_row.get("levels", {}).get("default", [])
        count = len(candidates)
        for index, candidate in enumerate(candidates):
            graph = mapping.get(index)
            if graph is None or not graph.get("graph_valid") or graph.get("graph_visited"):
                continue
            if graph.get("graph_logit") is None:
                continue
            visual_item = visual.get(index, {})
            angle = float(visual_item.get("angle", candidate["angle"]))
            features = [
                float(graph["graph_logit"]),
                float(visual_item.get("score", candidate.get("score", 0.0))),
                float(candidate["distance"]),
                math.sin(angle),
                math.cos(angle),
                float(diagnostic.get("candidate_angular_dispersion", 0.0)),
                float(diagnostic.get("policy_entropy", 0.0)),
                float(count),
            ]
            records.append({"key": key, "features": features, "progress": float(candidate["progress"])})
    return records


def evaluate(records, weights, mean, scale):
    by_state = {}
    for record in records:
        key = record["key"]
        by_state.setdefault(key, []).append(record)
    regrets = []
    hits = []
    for items in by_state.values():
        target = max(item["progress"] for item in items)
        scores = [float(np.dot((np.asarray(item["features"]) - mean) / scale, weights)) for item in items]
        chosen = max(range(len(items)), key=lambda index: scores[index])
        regrets.append(target - items[chosen]["progress"])
        hits.append(float(items[chosen]["progress"] >= target - 1e-8))
    return {
        "decisions": len(by_state),
        "candidates": len(records),
        "top1_rate": float(np.mean(hits)) if hits else None,
        "regret_mean_m": float(np.mean(regrets)) if regrets else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("train_diagnostics", type=Path)
    parser.add_argument("train_oracle", type=Path)
    parser.add_argument("val_diagnostics", type=Path)
    parser.add_argument("val_oracle", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    train = load_rows(args.train_diagnostics, args.train_oracle)
    val = load_rows(args.val_diagnostics, args.val_oracle)
    x_train = np.asarray([record["features"] for record in train], dtype=np.float64)
    y_train = np.asarray([record["progress"] for record in train], dtype=np.float64)
    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    scale[scale < 1e-8] = 1.0
    z_train = (x_train - mean) / scale
    ridge = 10.0
    weights = np.linalg.solve(
        z_train.T @ z_train + ridge * np.eye(z_train.shape[1]),
        z_train.T @ y_train,
    )
    learned = evaluate(val, weights, mean, scale)
    graph_weights = np.zeros(len(FEATURE_NAMES))
    graph_weights[0] = 1.0
    heatmap_weights = np.zeros(len(FEATURE_NAMES))
    heatmap_weights[1] = 1.0
    result = {
        "schema_version": 1,
        "feature_names": list(FEATURE_NAMES),
        "train_decisions": len({record["key"] for record in train}),
        "train_candidates": len(train),
        "val_decisions": len({record["key"] for record in val}),
        "val_candidates": len(val),
        "ridge_lambda": ridge,
        "weights_standardized": weights.tolist(),
        "validation": {
            "linear_ranker": learned,
            "graph_logit": evaluate(val, graph_weights, mean, scale),
            "heatmap_score": evaluate(val, heatmap_weights, mean, scale),
        },
        "privileged_label": "one-step goal-distance progress",
        "deployable_features": list(FEATURE_NAMES),
        "interpretation": "offline train-to-val diagnostic; no navigation trajectory used the learned weights",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
