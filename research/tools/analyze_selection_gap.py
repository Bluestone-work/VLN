#!/usr/bin/env python3
"""Compare realized default decisions with same-state oracle candidate potential."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def mean(values):
    return sum(values) / len(values) if values else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("diagnostics", type=Path)
    parser.add_argument("oracle", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    diagnostics = [json.loads(line) for line in args.diagnostics.read_text().splitlines() if line.strip()]
    oracle = [json.loads(line) for line in args.oracle.read_text().splitlines() if line.strip()]
    diag_by_key = {(row["episode_id"], row["high_level_step"]): row for row in diagnostics}
    oracle_by_key = {(row["episode_id"], row["high_level_step"]): row for row in oracle}
    keys = sorted(set(diag_by_key) & set(oracle_by_key))
    levels = ("coarse", "default", "fine")
    actual = []
    best = {level: [] for level in levels}
    winner_actual = defaultdict(list)
    target_types = Counter()
    matched = []
    for key in keys:
        d = diag_by_key[key]
        o = oracle_by_key[key]
        actual_progress = float(d["progress"])
        actual.append(actual_progress)
        for level in levels:
            item = o.get("best_by_level", {}).get(level)
            if item is not None:
                best[level].append(float(item["progress"]))
        winner_actual[o.get("oracle_level")].append(actual_progress)
        target_types[d.get("selected_target", {}).get("type", "unknown")] += 1
        matched.append({
            "episode_id": key[0],
            "high_level_step": key[1],
            "actual_progress": actual_progress,
            "oracle_level": o.get("oracle_level"),
            "oracle_best_progress": float(o.get("oracle_best_progress", 0.0)),
            "default_best_progress": float(o["best_by_level"]["default"]["progress"]),
            "selection_gap": float(o["best_by_level"]["default"]["progress"]) - actual_progress,
            "policy_entropy": float(d.get("policy_entropy", 0.0)),
            "selected_target_type": d.get("selected_target", {}).get("type"),
        })
    gaps = [row["selection_gap"] for row in matched]
    oracle_gaps = [row["oracle_best_progress"] - row["actual_progress"] for row in matched]
    ghost_rows = [row for row in matched if row["selected_target_type"] == "ghost"]
    ghost_gaps = [row["selection_gap"] for row in ghost_rows]
    result = {
        "schema_version": 1,
        "num_diagnostics": len(diagnostics),
        "num_oracle_records": len(oracle),
        "num_matched_decisions": len(matched),
        "actual_default_progress_mean": mean(actual),
        "actual_default_positive_rate": sum(value > 0 for value in actual) / len(actual) if actual else None,
        "best_candidate_progress_mean": {level: mean(best[level]) for level in levels},
        "default_candidate_selection_gap_mean_m": mean(gaps),
        "oracle_candidate_selection_gap_mean_m": mean(oracle_gaps),
        "ghost_only_matched_decisions": len(ghost_rows),
        "ghost_only_default_selection_gap_mean_m": mean(ghost_gaps),
        "ghost_only_actual_progress_mean": mean([row["actual_progress"] for row in ghost_rows]),
        "actual_progress_by_oracle_winner": {
            level: {"count": len(values), "mean": mean(values)}
            for level, values in sorted(winner_actual.items())
        },
        "selected_target_type_count": dict(target_types),
        "interpretation": {
            "default_candidate_selection_gap": "best default candidate progress minus realized default graph action progress; includes navigator selection and low-level execution",
            "oracle_candidate_selection_gap": "privileged best fine/default/coarse candidate progress minus realized default graph action progress; not deployable",
        },
        "records": matched,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "records"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
