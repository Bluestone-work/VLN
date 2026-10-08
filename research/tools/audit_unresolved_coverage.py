#!/usr/bin/env python3
"""Summarize unresolved failure routes without fitting a policy.

This is a post-hoc diagnostic over already executed exhaustive native-action
returns.  It measures the available action branches and terminal/execution
symptoms, but it does not label proposal failure causally: an untested action
representation, earlier timing, or a controller failure can still explain a
route with available graph actions.
"""
import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path


def _load(path):
    with open(path) as f:
        return json.load(f)


def _route_summary(summary_path, run_dir):
    summary = _load(summary_path)
    unresolved = {
        str(route_id)
        for route_id, info in summary.get("routes", {}).items()
        if info.get("unresolved_after_tested_interventions")
    }
    rows = defaultdict(list)
    controls = {}
    for path in glob.glob(str(Path(run_dir) / "cases" / "*" / "result.json")):
        rec = _load(path)
        case = rec.get("case", {})
        route_id = str(case.get("episode_id", ""))
        mode = case.get("mode")
        if mode == "control":
            controls[route_id] = rec
        elif mode == "action" and route_id in unresolved:
            rows[route_id].append(rec)

    output = []
    for route_id in sorted(unresolved, key=lambda x: (len(x), x)):
        cases = rows.get(route_id, [])
        by_step = defaultdict(list)
        for rec in cases:
            by_step[int(rec["case"].get("high_level_step", -1))].append(rec)
        states = []
        all_nonstop = 0
        all_actions = 0
        for step, state_cases in sorted(by_step.items()):
            actions = [r["case"].get("action", {}) for r in state_cases]
            non_stop = [a for a in actions if a.get("act") != 0]
            ghosts = sorted({str(a.get("ghost_vp")) for a in non_stop if a.get("ghost_vp") is not None})
            fronts = sorted({str(a.get("front_vp")) for a in non_stop if a.get("front_vp") is not None})
            reasons = sorted({reason for r in state_cases for reason in r["case"].get("reasons", [])})
            quality = [r for r in state_cases if r.get("metrics", {}).get("success", 0) > 0]
            quality_ndtw = [r for r in quality if r.get("metrics", {}).get("ndtw", 0) >= r.get("baseline_metrics", {}).get("ndtw", 0)]
            states.append({
                "high_level_step": step,
                "branch_count": len(state_cases),
                "nonstop_branch_count": len(non_stop),
                "unique_ghost_targets": len(ghosts),
                "unique_front_nodes": len(fronts),
                "reasons": reasons,
                "collision_branch_count": sum(1 for r in state_cases if r.get("primitive_collision_events", 0) > 0),
                "sr_rescue_branch_count": len(quality),
                "quality_rescue_branch_count": len(quality_ndtw),
                "action_indices": sorted({int(r["case"].get("action_index", -1)) for r in state_cases}),
            })
            all_nonstop += len(non_stop)
            all_actions += len(state_cases)
        control = controls.get(route_id, {})
        cm = control.get("metrics", {})
        output.append({
            "episode_id": route_id,
            "scene_id": summary.get("routes", {}).get(route_id, {}).get("scene_id"),
            "critical_state_count": len(states),
            "full_return_branch_count": all_actions,
            "full_return_nonstop_branch_count": all_nonstop,
            "has_nonstop_branch": bool(all_nonstop),
            "max_state_nonstop_branch_count": max((s["nonstop_branch_count"] for s in states), default=0),
            "max_state_unique_ghost_targets": max((s["unique_ghost_targets"] for s in states), default=0),
            "native_collision_events": control.get("primitive_collision_events", 0),
            "native_collisions_metric": cm.get("collisions"),
            "native_forced_stop": bool(cm.get("forced_stop_count", 0)),
            "native_high_level_steps": cm.get("high_level_steps"),
            "native_distance_to_goal": cm.get("distance_to_goal"),
            "native_ndtw": cm.get("ndtw"),
            "states": states,
        })
    return {
        "status": "complete",
        "post_hoc": True,
        "no_model_trained": True,
        "causal_proposal_failure_not_identified": True,
        "unresolved_route_count": len(output),
        "routes": output,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", action="append", nargs=2, metavar=("SUMMARY", "RUN"), required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cohorts = []
    for summary, run in args.summary:
        cohorts.append({"summary": summary, "run": run, "result": _route_summary(summary, run)})
    result = {"experiment_id": "UNRESOLVED-COVERAGE-CENSUS-001", "cohorts": cohorts}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": "complete", "cohorts": len(cohorts), "output": str(out)}, indent=2))


if __name__ == "__main__":
    main()
