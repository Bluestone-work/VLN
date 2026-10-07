#!/usr/bin/env python3
"""Decompose same-state default candidate, graph-selection, and execution gaps."""

import argparse
import json
from collections import Counter
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
    diag_map = {(row["episode_id"], row["high_level_step"]): row for row in diagnostics}
    oracle_map = {(row["episode_id"], row["high_level_step"]): row for row in oracle}
    keys = sorted(set(diag_map) & set(oracle_map))
    realized = []
    available_best = []
    selected_cf = []
    selection_gaps = []
    execution_gaps = []
    total_candidates = 0
    mapped_candidates = 0
    available_candidates = 0
    selected_current = 0
    selected_target_types = Counter()
    duplicate_graph_targets = []
    rank_values = []
    by_target = {}
    rows = []
    for key in keys:
        diagnostic = diag_map[key]
        oracle_row = oracle_map[key]
        oracle_candidates = oracle_row.get("levels", {}).get("default", [])
        mapping = diagnostic.get("candidate_graph_mapping", [])
        total_candidates += len(mapping)
        by_index = {int(item.get("candidate_index")): item for item in mapping}
        graph_ids = [item.get("graph_id") for item in mapping if item.get("graph_id") is not None]
        duplicate_graph_targets.append(len(graph_ids) - len(set(graph_ids)))
        eligible = []
        for index, candidate in enumerate(oracle_candidates):
            item = by_index.get(index)
            if item is None:
                continue
            mapped_candidates += 1
            if item.get("graph_valid") and not item.get("graph_visited"):
                available_candidates += 1
                eligible.append((index, float(candidate["progress"]), item))
        actual = float(diagnostic.get("progress", 0.0))
        realized.append(actual)
        best = max(eligible, key=lambda value: value[1]) if eligible else None
        best_value = best[1] if best else None
        selected_id = diagnostic.get("selected_target", {}).get("id")
        selected_items = [value for value in eligible if value[2].get("graph_id") == selected_id]
        selected_value = max((value[1] for value in selected_items), default=None)
        selected_target_types[diagnostic.get("selected_target", {}).get("type", "unknown")] += 1
        target_type = diagnostic.get("selected_target", {}).get("type", "unknown")
        target_stats = by_target.setdefault(target_type, {"count": 0, "selected_current": 0, "selection_gaps": [], "execution_gaps": [], "actual": []})
        target_stats["count"] += 1
        target_stats["actual"].append(actual)
        if selected_items:
            selected_current += 1
            target_stats["selected_current"] += 1
            selected_cf.append(selected_value)
            if best_value is not None:
                selection_gaps.append(best_value - selected_value)
                target_stats["selection_gaps"].append(best_value - selected_value)
            execution_gaps.append(selected_value - actual)
            target_stats["execution_gaps"].append(selected_value - actual)
            logits = [value[2].get("graph_logit") for value in eligible if value[2].get("graph_logit") is not None]
            selected_logit = max((value[2].get("graph_logit") for value in selected_items), default=None)
            if selected_logit is not None and logits:
                rank_values.append(1 + sum(logit > selected_logit for logit in logits))
        if best_value is not None:
            available_best.append(best_value)
        rows.append({
            "episode_id": key[0],
            "high_level_step": key[1],
            "actual_progress": actual,
            "eligible_candidate_count": len(eligible),
            "best_available_progress": best_value,
            "selected_counterfactual_progress": selected_value,
            "selection_gap": (best_value - selected_value) if best_value is not None and selected_value is not None else None,
            "execution_gap": (selected_value - actual) if selected_value is not None else None,
            "selected_target_type": diagnostic.get("selected_target", {}).get("type"),
        })
    result = {
        "schema_version": 1,
        "num_matched_decisions": len(keys),
        "total_default_candidates": total_candidates,
        "mapped_default_candidates": mapped_candidates,
        "mapping_coverage": mapped_candidates / total_candidates if total_candidates else None,
        "available_unvisited_candidates": available_candidates,
        "realized_progress_mean_m": mean(realized),
        "best_available_graph_candidate_progress_mean_m": mean(available_best),
        "selected_current_candidate_rate": selected_current / len(keys) if keys else None,
        "selected_counterfactual_progress_mean_m": mean(selected_cf),
        "graph_selection_gap_mean_m": mean(selection_gaps),
        "execution_gap_after_selected_candidate_mean_m": mean(execution_gaps),
        "selected_graph_logit_rank_mean": mean(rank_values),
        "duplicate_graph_target_mean_per_decision": mean(duplicate_graph_targets),
        "selected_target_type_count": dict(selected_target_types),
        "by_selected_target_type": {
            target_type: {
                "count": stats["count"],
                "selected_current_rate": stats["selected_current"] / stats["count"] if stats["count"] else None,
                "actual_progress_mean_m": mean(stats["actual"]),
                "graph_selection_gap_mean_m": mean(stats["selection_gaps"]),
                "execution_gap_mean_m": mean(stats["execution_gaps"]),
            }
            for target_type, stats in sorted(by_target.items())
        },
        "interpretation": {
            "eligible": "candidate mapped to a valid, unvisited graph node",
            "selection_gap": "best eligible candidate counterfactual progress minus selected candidate counterfactual progress",
            "execution_gap": "selected candidate counterfactual progress minus realized progress; includes graph ghost merge and low-level execution",
        },
        "records": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "records"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
