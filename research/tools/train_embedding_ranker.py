#!/usr/bin/env python3
"""Offline rankers over frozen instruction-conditioned graph embeddings."""

import argparse
import json
import random
from pathlib import Path

import numpy as np


def load_records(diag_path, oracle_path):
    diagnostics = [json.loads(line) for line in Path(diag_path).read_text().splitlines() if line.strip()]
    oracle = [json.loads(line) for line in Path(oracle_path).read_text().splitlines() if line.strip()]
    diag_map = {(row["episode_id"], row["high_level_step"]): row for row in diagnostics}
    oracle_map = {(row["episode_id"], row["high_level_step"]): row for row in oracle}
    records = []
    for key in sorted(set(diag_map) & set(oracle_map)):
        diagnostic = diag_map[key]
        oracle_row = oracle_map[key]
        mapping = {int(item["candidate_index"]): item for item in diagnostic.get("candidate_graph_mapping", [])}
        candidates = oracle_row.get("levels", {}).get("default", [])
        for index, candidate in enumerate(candidates):
            mapping_item = mapping.get(index)
            if mapping_item is None or not mapping_item.get("graph_valid") or mapping_item.get("graph_visited"):
                continue
            embedding = mapping_item.get("graph_embedding")
            if embedding is None:
                continue
            records.append({"key": key, "features": embedding, "progress": float(candidate["progress"]), "graph_logit": float(mapping_item["graph_logit"])})
    return records


def mean(values):
    return float(np.mean(values)) if values else None


def evaluate(records, weights, mean_vector, scale_vector, signal="embedding"):
    grouped = {}
    for record in records:
        grouped.setdefault(record["key"], []).append(record)
    regrets, hits = [], []
    for items in grouped.values():
        target = max(item["progress"] for item in items)
        if signal == "graph_logit":
            scores = [item["graph_logit"] for item in items]
        else:
            scores = [float(np.dot((np.asarray(item["features"]) - mean_vector) / scale_vector, weights)) for item in items]
        chosen = max(range(len(items)), key=lambda index: scores[index])
        regrets.append(target - items[chosen]["progress"])
        hits.append(float(items[chosen]["progress"] >= target - 1e-8))
    return {
        "decisions": len(grouped),
        "candidates": len(records),
        "top1_rate": mean(hits),
        "regret_mean_m": mean(regrets),
    }


def paired_regrets(records, pointwise, pairwise, mean_vector, scale_vector):
    grouped = {}
    for record in records:
        grouped.setdefault(record["key"], []).append(record)
    differences = {"pointwise_minus_graph": [], "pairwise_minus_graph": []}
    for items in grouped.values():
        target = max(item["progress"] for item in items)
        graph_choice = max(items, key=lambda item: item["graph_logit"])
        point_scores = [float(np.dot((np.asarray(item["features"]) - mean_vector) / scale_vector, pointwise)) for item in items]
        pair_scores = [float(np.dot((np.asarray(item["features"]) - mean_vector) / scale_vector, pairwise)) for item in items]
        point_choice = items[max(range(len(items)), key=lambda index: point_scores[index])]
        pair_choice = items[max(range(len(items)), key=lambda index: pair_scores[index])]
        graph_regret = target - graph_choice["progress"]
        differences["pointwise_minus_graph"].append((target - point_choice["progress"]) - graph_regret)
        differences["pairwise_minus_graph"].append((target - pair_choice["progress"]) - graph_regret)
    return differences


def bootstrap_ci(values, seed=20261007, samples=5000):
    rng = random.Random(seed)
    means = []
    for _ in range(samples):
        draw = [values[rng.randrange(len(values))] for _ in values]
        means.append(float(np.mean(draw)))
    means.sort()
    return [means[int(0.025 * len(means))], means[int(0.975 * len(means))]]


def fit_pointwise(train, mean_vector, scale_vector, ridge):
    x = np.asarray([(np.asarray(r["features"]) - mean_vector) / scale_vector for r in train])
    y = np.asarray([r["progress"] for r in train])
    return np.linalg.solve(x.T @ x + ridge * np.eye(x.shape[1]), x.T @ y)


def fit_pairwise(train, mean_vector, scale_vector, ridge):
    grouped = {}
    for record in train:
        grouped.setdefault(record["key"], []).append(record)
    diffs, labels = [], []
    for items in grouped.values():
        for left in range(len(items)):
            for right in range(left + 1, len(items)):
                difference = items[left]["progress"] - items[right]["progress"]
                if abs(difference) < 1e-8:
                    continue
                left_x = (np.asarray(items[left]["features"]) - mean_vector) / scale_vector
                right_x = (np.asarray(items[right]["features"]) - mean_vector) / scale_vector
                diffs.append(left_x - right_x)
                labels.append(1.0 if difference > 0 else -1.0)
    x = np.asarray(diffs)
    y = np.asarray(labels)
    return np.linalg.solve(x.T @ x + ridge * np.eye(x.shape[1]), x.T @ y), len(labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("train_diagnostics", type=Path)
    parser.add_argument("train_oracle", type=Path)
    parser.add_argument("val_diagnostics", type=Path)
    parser.add_argument("val_oracle", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    train = load_records(args.train_diagnostics, args.train_oracle)
    val = load_records(args.val_diagnostics, args.val_oracle)
    train_matrix = np.asarray([record["features"] for record in train], dtype=np.float64)
    mean_vector = train_matrix.mean(axis=0)
    scale_vector = train_matrix.std(axis=0)
    scale_vector[scale_vector < 1e-8] = 1.0
    ridge = 100.0
    pointwise = fit_pointwise(train, mean_vector, scale_vector, ridge)
    pairwise, pair_count = fit_pairwise(train, mean_vector, scale_vector, ridge)
    paired = paired_regrets(val, pointwise, pairwise, mean_vector, scale_vector)
    result = {
        "schema_version": 1,
        "feature_dim": int(train_matrix.shape[1]),
        "train_decisions": len({r["key"] for r in train}),
        "train_candidates": len(train),
        "val_decisions": len({r["key"] for r in val}),
        "val_candidates": len(val),
        "pair_count": pair_count,
        "ridge_lambda": ridge,
        "validation": {
            "pointwise_embedding": evaluate(val, pointwise, mean_vector, scale_vector),
            "pairwise_embedding": evaluate(val, pairwise, mean_vector, scale_vector),
            "graph_logit": evaluate(val, None, mean_vector, scale_vector, signal="graph_logit"),
        },
        "paired_regret_difference_vs_graph_logit": {
            name: {
                "mean_m": float(np.mean(values)),
                "bootstrap_95ci_m": bootstrap_ci(values),
            }
            for name, values in paired.items()
        },
        "weights_norm": {
            "pointwise": float(np.linalg.norm(pointwise)),
            "pairwise": float(np.linalg.norm(pairwise)),
        },
        "privileged_label": "one-step goal-distance progress",
        "interpretation": "offline train-to-val diagnostic; no navigation trajectory used these weights",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
