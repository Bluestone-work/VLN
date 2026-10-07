#!/usr/bin/env python3
"""Quantify candidate aliasing introduced by GraphMap ghost/real-node merging.

The waypoint generator may emit several geometrically different proposals that
GraphMap maps to one graph node.  A graph ranker then assigns one logit to all
of them, so candidate-level progress labels can overstate the action choices
that are actually available.  This script reports that loss explicitly using
the privileged same-state oracle labels; it never changes navigation.
"""

import argparse
import collections
import json
import statistics
from pathlib import Path


def load(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("diagnostics", type=Path)
    parser.add_argument("oracle", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    diagnostics = {(r["episode_id"], r["high_level_step"]): r for r in load(args.diagnostics)}
    oracle = {(r["episode_id"], r["high_level_step"]): r for r in load(args.oracle)}
    duplicate_state = 0
    duplicate_groups = 0
    duplicate_candidates = 0
    mixed_visited_groups = 0
    spreads = []
    high_spread = []
    state_rows = []
    for key in sorted(set(diagnostics) & set(oracle)):
        diagnostic = diagnostics[key]
        levels = oracle[key].get("levels", {}).get("default", [])
        groups = collections.defaultdict(list)
        for item in diagnostic.get("candidate_graph_mapping", []):
            index = int(item["candidate_index"])
            graph_id = item.get("graph_id")
            if not graph_id or not item.get("graph_valid") or index >= len(levels):
                continue
            groups[str(graph_id)].append((index, float(levels[index]["progress"]),
                                          bool(item.get("graph_visited"))))
        local_spreads = []
        local_groups = 0
        local_candidates = 0
        local_mixed = 0
        for graph_id, members in groups.items():
            if len(members) < 2:
                continue
            local_groups += 1
            local_candidates += len(members)
            values = [member[1] for member in members]
            spread = max(values) - min(values)
            local_spreads.append(spread)
            spreads.append(spread)
            if spread >= 0.5:
                high_spread.append({
                    "episode_id": key[0],
                    "high_level_step": key[1],
                    "graph_id": graph_id,
                    "candidate_indices": [member[0] for member in members],
                    "progress_values": values,
                })
            if len({member[2] for member in members}) > 1:
                local_mixed += 1
        if local_groups:
            duplicate_state += 1
            duplicate_groups += local_groups
            duplicate_candidates += local_candidates
            mixed_visited_groups += local_mixed
            state_rows.append({
                "episode_id": key[0],
                "high_level_step": key[1],
                "duplicate_groups": local_groups,
                "duplicate_candidates": local_candidates,
                "max_progress_spread": max(local_spreads),
                "mean_progress_spread": statistics.mean(local_spreads),
                "mixed_visited_groups": local_mixed,
            })

    decisions = len(set(diagnostics) & set(oracle))
    result = {
        "schema_version": 1,
        "diagnostic_file": str(args.diagnostics),
        "oracle_file": str(args.oracle),
        "decisions": decisions,
        "states_with_candidate_aliasing": duplicate_state,
        "aliasing_state_rate": duplicate_state / decisions if decisions else None,
        "duplicate_graph_groups": duplicate_groups,
        "duplicate_candidate_members": duplicate_candidates,
        "mixed_visited_alias_groups": mixed_visited_groups,
        "progress_spread_m": {
            "groups": len(spreads),
            "mean": statistics.mean(spreads) if spreads else None,
            "median": statistics.median(spreads) if spreads else None,
            "p90": sorted(spreads)[int(0.9 * (len(spreads) - 1))] if spreads else None,
            "max": max(spreads) if spreads else None,
            "groups_ge_0_5m": len(high_spread),
        },
        "interpretation": (
            "Diagnostic-only estimate: candidate progress labels are privileged "
            "one-step goal-distance changes. A spread within one graph ID means "
            "GraphMap merging aliases proposals that a graph ranker cannot score separately."
        ),
        "largest_aliases": sorted(high_spread, key=lambda row: max(row["progress_values"])
                                  - min(row["progress_values"]), reverse=True)[:20],
        "state_rows": state_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("state_rows", "largest_aliases")},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
