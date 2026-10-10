#!/usr/bin/env python3
"""Build the frozen inventory for the dense single-override causal dataset.

This tool deliberately refuses to turn dense full-return trajectories into
single-override labels.  Valid pairs must be supplied by a replay runner that
passes the prefix and post-event fidelity gate.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def action_sig(option):
    action = option["action"]
    return json.dumps(action, sort_keys=True, separators=(",", ":"))


def load_jsonl(path):
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def freeze_inventory(a0_options, a1_options, a0_traces, a1_traces):
    a0 = {(str(r["episode_id"]), int(r["high_level_step"])): r for r in a0_options}
    a1 = {(str(r["episode_id"]), int(r["high_level_step"])): r for r in a1_options}
    t0 = {(str(r["episode_id"]), int(r["high_level_step"])): r for r in a0_traces}
    t1 = {(str(r["episode_id"]), int(r["high_level_step"])): r for r in a1_traces}
    rows = []
    for key in sorted(set(a0) & set(a1) & set(t0) & set(t1)):
        o0, o1, r0, r1 = a0[key], a1[key], t0[key], t1[key]
        if r0.get("pre_pose") != r1.get("pre_pose") or r0.get("observation_hashes") != r1.get("observation_hashes"):
            continue
        native = [o for o in o0["options"] if o.get("admissible")]
        native_sigs = {action_sig(o) for o in native}
        dense = [o for o in o1["options"] if o.get("admissible") and action_sig(o) not in native_sigs and int(o["action"].get("act", -1)) == 4]
        if not dense:
            continue
        dense.sort(key=lambda o: (-float(o.get("logit", float("-inf"))), int(o["index"])))
        chosen = dense[0]
        native_index = int(o0["effective_index"])
        native_option = next((o for o in o0["options"] if int(o.get("index", -1)) == native_index), None)
        if native_option is None:
            continue
        native_logit = float(o0["logits"][native_index])
        rows.append({
            "state_key": f"{key[0]}:{key[1]}", "episode_id": key[0],
            "scene_id": o0["scene_id"], "high_level_step": key[1],
            "pre_pose": r0.get("pre_pose"), "observation_hashes": r0.get("observation_hashes"),
            "native_action_signature": action_sig(native_option),
            "dense_action_signature": action_sig(chosen), "dense_index_a1": int(chosen["index"]),
            "native_logit": native_logit,
            "dense_logit": float(chosen["logit"]),
            "dense_native_margin": float(chosen["logit"]) - native_logit,
            "native_provenance": "A0", "dense_provenance": "A1_new_candidate",
            "pool": "native_success_control" if bool(r0.get("post_metrics", {}).get("success", False)) else "native_failure_or_unknown",
            "replay_status": "not_run_no_snapshot",
        })
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--a0", type=Path, required=True)
    p.add_argument("--a1", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    a0o, a1o = load_jsonl(args.a0 / "graph_options/graph_options.jsonl"), load_jsonl(args.a1 / "graph_options/graph_options.jsonl")
    a0t, a1t = load_jsonl(args.a0 / "traces/worker_seed100.jsonl"), load_jsonl(args.a1 / "traces/worker_seed100.jsonl")
    inventory = freeze_inventory(a0o, a1o, a0t, a1t)
    with (out / "state_inventory.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(inventory[0]) if inventory else ["state_key"])
        writer.writeheader(); writer.writerows(inventory)
    with (out / "replay_fidelity.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["state_key", "status", "reason"]); writer.writeheader()
        for row in inventory:
            writer.writerow({"state_key": row["state_key"], "status": "FAIL", "reason": "no exact simulator/GraphMap snapshot runner executed"})
    (out / "single_override_pairs.jsonl").write_text("")
    summary = {"eligible_states": len(inventory), "valid_causal_pairs": 0, "replay_failures": len(inventory), "benefit": 0, "harm": 0, "uncertain": 0, "labels_frozen": True, "status": "blocked_until_fidelity_runner"}
    (out / "label_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for name, key in [("per_scene_summary.csv", "scene_id"), ("per_route_summary.csv", "episode_id")]:
        counts = {}
        for row in inventory: counts.setdefault(row[key], 0); counts[row[key]] += 1
        with (out / name).open("w", newline="") as f:
            w = csv.writer(f); w.writerow([key, "eligible_states", "valid_pairs", "replay_failures"])
            for k, n in sorted(counts.items()): w.writerow([k, n, 0, n])
    manifest = {"tool": str(Path(__file__)), "a0": str(args.a0), "a1": str(args.a1), "a0_options_sha256": digest(args.a0 / "graph_options/graph_options.jsonl"), "a1_options_sha256": digest(args.a1 / "graph_options/graph_options.jsonl"), "semantic_guard": "dense full-return outcomes are not used as single-override labels"}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
