#!/usr/bin/env python3
"""Summarize same-state adaptive-abstraction oracle JSONL records."""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def _mean(values):
    return sum(values) / len(values) if values else None


def _stderr(values):
    if len(values) < 2:
        return None
    mean = _mean(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return (variance / len(values)) ** 0.5


def _bootstrap_mean(values, seed=20261006, samples=5000):
    if not values:
        return None
    if len(values) == 1:
        return [values[0], values[0]]
    rng = random.Random(seed)
    means = []
    for _ in range(samples):
        draw = [values[rng.randrange(len(values))] for _ in values]
        means.append(_mean(draw))
    means.sort()
    return [means[int(0.025 * len(means))], means[int(0.975 * len(means))]]


def summarize(rows):
    levels = ("coarse", "default", "fine")
    best = {level: [] for level in levels}
    winners = Counter()
    gains = {level: [] for level in levels}
    by_bucket = defaultdict(lambda: {level: [] for level in levels})
    by_distance = defaultdict(lambda: {level: [] for level in levels})
    winners_by_distance = defaultdict(Counter)
    counts_by_distance = Counter()
    oracle_gains = []
    best_fixed_level = None
    for row in rows:
        for level in levels:
            item = row.get("best_by_level", {}).get(level)
            if item is not None:
                best[level].append(float(item["progress"]))
        winner = row.get("oracle_level")
        if winner:
            winners[winner] += 1
        default_item = row.get("best_by_level", {}).get("default")
        if default_item is None:
            continue
        default_progress = float(default_item["progress"])
        for level in levels:
            item = row.get("best_by_level", {}).get(level)
            if item is not None:
                gains[level].append(float(item["progress"]) - default_progress)
        # Candidate availability is the directly observed state proxy.  The
        # fixed default candidate count is available from the full level list.
        count = len(row.get("levels", {}).get("default", []))
        bucket = "0-2" if count <= 2 else "3-5" if count <= 5 else "6+"
        distance = float(row.get("distance_to_goal_before", 0.0))
        distance_bucket = "0-3m" if distance <= 3.0 else "3-6m" if distance <= 6.0 else "6m+"
        counts_by_distance[distance_bucket] += 1
        if winner:
            winners_by_distance[distance_bucket][winner] += 1
        for level in levels:
            item = row.get("best_by_level", {}).get(level)
            if item is not None:
                by_bucket[bucket][level].append(float(item["progress"]))
                by_distance[distance_bucket][level].append(float(item["progress"]))
        oracle_item = row.get("best_by_level", {}).get(winner)
        if oracle_item is not None and default_item is not None:
            oracle_gains.append(float(oracle_item["progress"]) - default_progress)
    fixed_level_mean = {level: _mean(best[level]) for level in levels}
    best_fixed_level = max(
        (level for level in levels if fixed_level_mean[level] is not None),
        key=lambda level: fixed_level_mean[level],
    ) if any(fixed_level_mean[level] is not None for level in levels) else None
    oracle_minus_best_fixed = []
    strict_winners = Counter()
    tie_count = 0
    for row in rows:
        values = {level: row.get("best_by_level", {}).get(level, {}).get("progress")
                  for level in levels}
        values = {level: float(value) for level, value in values.items() if value is not None}
        if not values:
            continue
        maximum = max(values.values())
        tied = [level for level, value in values.items() if abs(value - maximum) <= 1e-5]
        if len(tied) > 1:
            tie_count += 1
        else:
            strict_winners[tied[0]] += 1
        if best_fixed_level in values:
            oracle_minus_best_fixed.append(maximum - values[best_fixed_level])
    n = len(rows)
    return {
        "schema_version": 1,
        "num_decisions": n,
        "num_episodes": len({row.get("episode_id") for row in rows}),
        "best_progress_mean": fixed_level_mean,
        "best_progress_median": {
            level: (sorted(best[level])[len(best[level]) // 2] if best[level] else None)
            for level in levels
        },
        "oracle_selection_count": dict(winners),
        "oracle_selection_rate": {
            level: winners[level] / n if n else None for level in levels
        },
        "gain_over_default_mean": {level: _mean(gains[level]) for level in levels},
        "gain_over_default_stderr": {level: _stderr(gains[level]) for level in levels},
        "oracle_gain_over_default_mean": _mean(oracle_gains),
        "oracle_gain_over_default_stderr": _stderr(oracle_gains),
        "best_fixed_level_by_mean_progress": best_fixed_level,
        "oracle_minus_best_fixed_mean": _mean(oracle_minus_best_fixed),
        "oracle_minus_best_fixed_stderr": _stderr(oracle_minus_best_fixed),
        "strict_oracle_selection_count": dict(strict_winners),
        "strict_oracle_selection_rate": {
            level: strict_winners[level] / (n - tie_count) if n != tie_count else None
            for level in levels
        },
        "tied_best_decision_count": tie_count,
        "gain_over_default_bootstrap_95ci": {
            level: _bootstrap_mean(gains[level]) for level in levels
        },
        "conditional_best_progress_by_default_candidate_count": {
            bucket: {level: _mean(values) for level, values in data.items()}
            for bucket, data in sorted(by_bucket.items())
        },
        "conditional_best_progress_by_distance_to_goal": {
            bucket: {level: _mean(values) for level, values in data.items()}
            for bucket, data in sorted(by_distance.items())
        },
        "oracle_selection_rate_by_distance_to_goal": {
            bucket: {
                level: winners_by_distance[bucket][level] / counts_by_distance[bucket]
                for level in levels
            }
            for bucket in sorted(counts_by_distance)
        },
        "interpretation": {
            "oracle_is_privileged": True,
            "progress_definition": "distance_to_goal_before - distance_to_goal_after",
            "state_proxy": "number of candidates produced by default NMS",
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    summary = summarize(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
