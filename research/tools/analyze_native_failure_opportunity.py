#!/usr/bin/env python3
"""Audit full-return action opportunity on the existing failure-enriched cohort.

This is a development diagnostic. The cohort was selected after native failure
was observed in an earlier critical-state study, so it is not an unbiased
evaluation set and must not be used to claim benchmark performance.
"""
import argparse
import glob
import json
from collections import Counter, defaultdict
from pathlib import Path


def compare(metrics, baseline):
    # Fixed lexicographic quality order matching the registered full-return
    # label semantics. Values are already computed by the full-return runner.
    for key, sign, eps in [
        ("success", 1, 1e-6), ("ndtw", 1, 1e-6), ("sdtw", 1, 1e-6),
        ("path_length", -1, 1e-5), ("steps_taken", -1, 1e-5),
        ("collisions", -1, 1e-6)]:
        delta = float(metrics.get(key, 0.0)) - float(baseline.get(key, 0.0))
        if abs(delta) > eps:
            return 1 if sign * delta > 0 else -1
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    states = defaultdict(list)
    graph = {}
    for path in glob.glob(str(args.cases / "e*_s*_a*" / "result.json")):
        row = json.loads(Path(path).read_text())
        if row["case"].get("mode") != "action":
            continue
        c = row["case"]
        key = (c["scene_id"], str(c["episode_id"]), int(c["high_level_step"]))
        states[key].append(row)
        graph_path = Path(path).parent / "graph" / "graph_options.jsonl"
        if graph_path.exists():
            for line in graph_path.open():
                g = json.loads(line)
                graph[(g["scene_id"], str(g["episode_id"]), int(g["high_level_step"]))] = g
    if not states:
        raise RuntimeError("No action result cases found")
    state_rows = []
    branch_counts = Counter()
    for key, branches in sorted(states.items()):
        baseline = branches[0]["baseline_metrics"]
        directions = [compare(x["metrics"], baseline) for x in branches]
        benefits = [x for x, d in zip(branches, directions) if d > 0]
        harms = [x for x, d in zip(branches, directions) if d < 0]
        success_rescues = [x for x in benefits if x["metrics"]["success"] > baseline["success"] + 1e-6]
        graph_row = graph.get(key, {})
        row = {
            "scene_id": key[0], "episode_id": key[1], "high_level_step": key[2],
            "native_success": bool(baseline["success"] > 0.5),
            "native_stop": bool(graph_row.get("effective_index") == 0),
            "branches": len(branches), "benefit_branches": len(benefits),
            "harm_branches": len(harms), "tied_branches": len(branches) - len(benefits) - len(harms),
            "has_benefit_alternative": bool(benefits),
            "success_rescue_branches": len(success_rescues),
            "has_success_rescue": bool(success_rescues),
            "quality_only_opportunity": bool(benefits) and not bool(success_rescues),
            "baseline_metrics": {k: baseline[k] for k in ["success", "ndtw", "sdtw", "path_length", "steps_taken", "collisions"]},
        }
        state_rows.append(row)
        branch_counts["branches"] += len(branches)
        branch_counts["benefit_branches"] += len(benefits)
        branch_counts["harm_branches"] += len(harms)
        branch_counts["tied_branches"] += len(branches) - len(benefits) - len(harms)
    episodes = sorted({r["episode_id"] for r in state_rows})
    scenes = sorted({r["scene_id"] for r in state_rows})
    opportunity_episodes = sorted({r["episode_id"] for r in state_rows if r["has_benefit_alternative"]})
    rescue_episodes = sorted({r["episode_id"] for r in state_rows if r["has_success_rescue"]})
    scene_summary = defaultdict(lambda: Counter())
    for r in state_rows:
        s = r["scene_id"].split("/")[-2]
        scene_summary[s]["states"] += 1
        scene_summary[s]["opportunity_states"] += int(r["has_benefit_alternative"])
        scene_summary[s]["success_rescue_states"] += int(r["has_success_rescue"])
    summary = {
        "schema_version": 1,
        "source_cases": str(args.cases),
        "cohort_status": "outcome-enriched development diagnostic; not an unbiased benchmark",
        "states": len(state_rows), "branches": int(branch_counts["branches"]),
        "episodes": len(episodes), "scenes": len(scenes),
        "all_native_failure": all(not r["native_success"] for r in state_rows),
        "branch_counts": dict(branch_counts),
        "states_with_benefit_alternative": sum(r["has_benefit_alternative"] for r in state_rows),
        "states_without_benefit_alternative": sum(not r["has_benefit_alternative"] for r in state_rows),
        "states_with_success_rescue": sum(r["has_success_rescue"] for r in state_rows),
        "states_quality_only_opportunity": sum(r["quality_only_opportunity"] for r in state_rows),
        "opportunity_episodes": opportunity_episodes,
        "success_rescue_episodes": rescue_episodes,
        "scene_summary": {k: dict(v) for k, v in sorted(scene_summary.items())},
        "label_features_separation": "This is outcome analysis only; no model feature is fitted.",
    }
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output / "states.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in state_rows))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
