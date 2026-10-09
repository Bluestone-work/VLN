#!/usr/bin/env python3
"""Validate native graph-action identity and write a leakage-checked pilot dataset.

This intentionally consumes accepted capture artifacts only. It does not fit a
ranker and it does not turn immediate goal progress into a label.
"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

FORBIDDEN = {
    "success", "final_success", "oracle_success", "distance_to_goal", "goal_distance",
    "reference_path", "gt_shortest_path", "future_collision", "continuation_outcome",
    "final_goal_distance", "spl", "ndtw", "sdtw", "future_progress", "progress_m",
    "distance_after", "target_error_horizontal_m", "post_pose", "oracle_arrival",
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def load_jsonl(path):
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def forbidden_keys(value, path=""):
    found = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key.lower() in FORBIDDEN:
                found.append(path + key)
            found.extend(forbidden_keys(child, path + key + "."))
    elif isinstance(value, list):
        for i, child in enumerate(value):
            found.extend(forbidden_keys(child, path + str(i) + "."))
    return found


def action_identity(option):
    return (option.get("action_type"), option.get("graph_id"),
            bool(option.get("current_proposal")), option.get("index"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--capture-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--max-states", type=int, default=8)
    p.add_argument("--full-return-results", type=Path)
    args = p.parse_args()
    root = args.capture_root
    decisions = load_jsonl(root / "probe_full001" / "decisions.jsonl")
    branches = load_jsonl(root / "probe_full001" / "branch_traces.jsonl")
    if not decisions or not branches:
        raise ValueError("capture is missing decisions or branch traces")
    selected = decisions[: args.max_states]
    selected_keys = {(d["scene_id"], str(d["episode_id"]), d["high_level_step"]) for d in selected}
    branch_map = {}
    for row in branches:
        # Reverse-order calibration intentionally repeats an index; the pilot
        # uses the forward execution for each state/action exactly once.
        if row.get("order") != "forward":
            continue
        key = (tuple(row["decision_key"]), row["index"])
        if key in branch_map:
            raise ValueError("duplicate branch identity: {}".format(key))
        branch_map[key] = row
    rows = []
    action_count = 0
    stop_count = historical_count = current_count = 0
    for decision in selected:
        state_key = (decision["scene_id"], str(decision["episode_id"]), decision["high_level_step"])
        options = [o for o in decision["options"] if o.get("admissible") is not False]
        if not options:
            raise ValueError("empty admissible set: {}".format(state_key))
        identities = set()
        actions = []
        for option in options:
            ident = action_identity(option)
            if ident in identities:
                raise ValueError("duplicate native action identity: {}".format(ident))
            identities.add(ident)
            branch_key = ((decision["scene_id"], str(decision["episode_id"]), decision["high_level_step"]), option["index"])
            branch = branch_map.get(branch_key)
            if branch is None or branch.get("order") != "forward":
                raise ValueError("missing forward branch: {}".format(branch_key))
            trace = branch["trace"]
            if trace["episode_id"] != str(decision["episode_id"]):
                raise ValueError("episode/action episode mismatch")
            # Isolated probes execute one option in a fresh worker and encode
            # that local execution step as zero; the decision key is the
            # authoritative high-level state identity.
            action = {
                "action_identity": {
                    "index": option["index"], "graph_id": option.get("graph_id"),
                    "action_type": option.get("action_type"),
                    "current_proposal": bool(option.get("current_proposal")),
                },
                "deployable_features": {
                    "native_graph_logit": option.get("logit"),
                    "action_type": option.get("action_type"),
                    "current_proposal": bool(option.get("current_proposal")),
                    "back_path_nodes": option.get("back_path_nodes"),
                    "native_motion_path_m": option.get("motion_path_m"),
                    "native_primitive_events": option.get("primitive_events"),
                    "native_collision_events": option.get("collision_events"),
                },
                "raw_option": option,
                "raw_native_branch_trace": trace,
            }
            leaked = forbidden_keys(action["deployable_features"])
            if leaked:
                raise ValueError("privileged feature leakage: {}".format(leaked))
            actions.append(action)
            action_count += 1
            if option.get("action_type") == 0:
                stop_count += 1
            elif option.get("current_proposal"):
                current_count += 1
            else:
                historical_count += 1
        rows.append({
            "schema_version": 1,
            "state_key": {"scene_id": state_key[0], "episode_id": state_key[1], "high_level_step": state_key[2]},
            "native_selected_index": decision["effective_index"],
            "native_policy_index": decision["policy_index"],
            "state_deployable_features": {
                "forced_stop": bool(decision.get("forced_stop")),
                "budget_stop": bool(decision.get("budget_stop")),
                "admissible_action_count": len(options),
                "native_selected_action_is_stop": any(o["index"] == decision["effective_index"] and o.get("action_type") == 0 for o in options),
            },
            "actions": actions,
            "raw_decision": decision,
        })
    full_return_rows = 0
    if args.full_return_results:
        for result in load_jsonl(args.full_return_results):
            if "metrics" not in result or "baseline_metrics" not in result:
                raise ValueError("full-return result lacks raw metrics")
            full_return_rows += 1
    args.output_dir.mkdir(parents=False, exist_ok=False)
    dataset = args.output_dir / "pilot_dataset.jsonl"
    with dataset.open("x") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    manifest = {
        "experiment_id": "LONG-HORIZON-GRAPH-VALUE-PILOT-001",
        "source_capture": str(root), "source_sha256": {str(p): digest(p) for p in [root / "probe_full001" / "decisions.jsonl", root / "probe_full001" / "branch_traces.jsonl"]},
        "states": len(rows), "actions": action_count, "current_actions": current_count,
        "historical_actions": historical_count, "stop_actions": stop_count,
        "full_return_result_rows_checked": full_return_rows,
        "native_action_identity_unique": True, "all_admissible_forward_branches_present": True,
        "deployable_feature_leakage": False, "raw_outcomes_retained": True,
        "note": "Pilot validates capture identity/serialization. Immediate branch metrics are raw diagnostics; no progress label or model fit is performed.",
        "dataset_sha256": digest(dataset),
        "git_commit_at_execution": subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip(),
        "git_status_at_execution": subprocess.check_output(["git", "status", "--porcelain"]).decode(),
        "command_argv": __import__("sys").argv,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
