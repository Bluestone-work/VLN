#!/usr/bin/env python3
"""Summarize the privileged graph-selection intervention."""

import argparse
import json
from collections import Counter
from pathlib import Path


def mean(values):
    return sum(values) / len(values) if values else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("oracle", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.oracle.read_text().splitlines() if line.strip()]
    overrides = [row for row in rows if row["policy_action_index"] != row["oracle_action_index"]]
    progress = []
    candidates = []
    for row in rows:
        selected = row.get("oracle_selected_candidate")
        if selected is not None:
            progress.append(float(selected["progress"]))
        candidates.extend(row.get("candidate_outcomes", []))
    result = {
        "schema_version": 1,
        "num_decisions": len(rows),
        "num_overrides": len(overrides),
        "override_rate": len(overrides) / len(rows) if rows else None,
        "oracle_selected_candidate_progress_mean_m": mean(progress),
        "oracle_candidate_count_mean": len(candidates) / len(rows) if rows else None,
        "oracle_action_type_count": Counter(
            "stop" if row["oracle_action_index"] == 0 else "ghost"
            for row in rows
        ),
        "policy_action_index_changed_count": len(overrides),
        "interpretation": {
            "privileged": True,
            "goal_distance_used": True,
            "trajectory_metrics_are_from_oracle_intervention_rollout": True,
            "not_deployable": True,
        },
    }
    result["oracle_action_type_count"] = dict(result["oracle_action_type_count"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
