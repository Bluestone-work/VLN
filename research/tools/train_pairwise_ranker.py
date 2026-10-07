#!/usr/bin/env python3
"""Offline pairwise linear ranker diagnostic."""

import argparse
import json
from pathlib import Path

import numpy as np

from train_linear_ranker import FEATURE_NAMES, evaluate, load_rows


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
    mean = np.asarray([record["features"] for record in train], dtype=np.float64).mean(axis=0)
    scale = np.asarray([record["features"] for record in train], dtype=np.float64).std(axis=0)
    scale[scale < 1e-8] = 1.0
    grouped = {}
    for record in train:
        grouped.setdefault(record["key"], []).append(record)
    diffs, labels = [], []
    for items in grouped.values():
        for left in range(len(items)):
            for right in range(left + 1, len(items)):
                delta = (
                    np.asarray(items[left]["features"]) - mean
                ) / scale - (
                    np.asarray(items[right]["features"]) - mean
                ) / scale
                difference = items[left]["progress"] - items[right]["progress"]
                if abs(difference) < 1e-8:
                    continue
                diffs.append(delta)
                labels.append(1.0 if difference > 0 else -1.0)
    x = np.asarray(diffs, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    ridge = 10.0
    weights = np.linalg.solve(x.T @ x + ridge * np.eye(x.shape[1]), x.T @ y)
    learned = evaluate(val, weights, mean, scale)
    result = {
        "schema_version": 1,
        "feature_names": list(FEATURE_NAMES),
        "train_decisions": len(grouped),
        "pair_count": len(labels),
        "val_decisions": len({record["key"] for record in val}),
        "ridge_lambda": ridge,
        "weights_standardized": weights.tolist(),
        "validation": {"pairwise_linear_ranker": learned},
        "privileged_label": "one-step goal-distance progress used only for offline pair construction",
        "interpretation": "offline train-to-val diagnostic; no navigation trajectory used these weights",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
