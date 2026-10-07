#!/usr/bin/env python3
"""Transfer the frozen graph preference baseline to a separate training cohort."""
import argparse
import collections
import hashlib
import json
from pathlib import Path

import numpy as np

from graph_value_feasibility import FEATURE_NAMES, action_features, dominates, fit_pairwise


def critical_records(plan_path, run_path):
    plan = json.loads(Path(plan_path).read_text())
    graph_path = Path(plan["config"]["source_root"]) / "capture_001/graph_options/graph_options.jsonl"
    graphs = {}
    for line in graph_path.open():
        row = json.loads(line); graphs[(str(row["episode_id"]), int(row["high_level_step"]))] = row
    out = []
    for line in (Path(run_path) / "results.jsonl").open():
        result = json.loads(line); c = result["case"]
        if c["mode"] != "action": continue
        key = (str(c["episode_id"]), int(c["high_level_step"]))
        row = graphs[key]; option = next(o for o in row["options"] if int(o["index"]) == int(c["action_index"]))
        metrics = dict(result["metrics"]); metrics["primitive_action_count"] = int(round(metrics.get("primitive_action_count", metrics["steps_taken"])))
        out.append({"scene_id": row["scene_id"], "episode_id": key[0], "step": key[1], "index": int(c["action_index"]),
                    "features": action_features(row, option), "metrics": metrics, "logit": float(option["logit"])})
    return out


def confirmation_records(root):
    root = Path(root); summary = json.loads((root / "full_return_analysis_001/summary.json").read_text())
    graph_path = root / "capture_001/graph_options/graph_options.jsonl"; graphs = {}
    for line in graph_path.open():
        row = json.loads(line); graphs[(str(row["episode_id"]), int(row["high_level_step"]))] = row
    events = summary["events"]; schedule = { (str(e["episode_id"]), int(e["high_level_step"])): e
                 for e in json.loads((root / "schedule_top_logit_001.json").read_text())["events"] }
    grouped = collections.defaultdict(dict)
    for event in events:
        key = (str(event["episode_id"]), int(event["step"])); row = graphs[key]
        for idx, metrics in [(None, event["baseline"]), (event["alternative_index"], event["alternative"])]:
            if idx is None:
                idx = schedule[(str(event["episode_id"]), int(event["step"]))]["baseline_index"]
            option = next(o for o in row["options"] if int(o["index"]) == int(idx))
            mm = dict(metrics); mm["primitive_action_count"] = int(round(mm.get("primitive_action_count", mm["steps_taken"])))
            grouped[key][int(idx)] = {"scene_id": row["scene_id"], "episode_id": key[0], "step": key[1], "index": int(idx),
                                      "features": action_features(row, option), "metrics": mm, "logit": float(option["logit"])}
    return list(grouped.values())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--train-plan", type=Path, required=True); ap.add_argument("--train-run", type=Path, required=True)
    ap.add_argument("--confirmation-root", type=Path, required=True); ap.add_argument("--output", type=Path, required=True); args = ap.parse_args()
    train_items = critical_records(args.train_plan, args.train_run); train_mean = np.mean([x["features"] for x in train_items], axis=0)
    train_scale = np.std([x["features"] for x in train_items], axis=0); train_scale[train_scale < 1e-8] = 1.0
    for x in train_items: x["z"] = (x["features"] - train_mean) / train_scale
    train_groups = collections.defaultdict(list)
    for x in train_items: train_groups[(x["episode_id"], x["step"])].append(x)
    train_pairs = []
    for items in train_groups.values():
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                if dominates(items[i]["metrics"], items[j]["metrics"]): train_pairs.append((items[i], items[j]))
                elif dominates(items[j]["metrics"], items[i]["metrics"]): train_pairs.append((items[j], items[i]))
    w = fit_pairwise(train_pairs)
    confirmation = confirmation_records(args.confirmation_root); pair_correct = {"learned": [], "native_logit": []}; groups = []
    for item_map in confirmation:
        items = list(item_map.values())
        for x in items: x["z"] = (x["features"] - train_mean) / train_scale
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                if dominates(items[i]["metrics"], items[j]["metrics"]): better, worse = items[i], items[j]
                elif dominates(items[j]["metrics"], items[i]["metrics"]): better, worse = items[j], items[i]
                else: continue
                pair_correct["learned"].append(float(np.dot(w, better["z"] - worse["z"]) > 0))
                pair_correct["native_logit"].append(float(better["logit"] > worse["logit"]))
        learned = max(items, key=lambda x: float(np.dot(w, x["z"]))); native = max(items, key=lambda x: x["logit"])
        dominating = any(dominates(other["metrics"], items[0]["metrics"]) for other in items)
        groups.append({"scene_id": items[0]["scene_id"], "episode_id": items[0]["episode_id"], "step": items[0]["step"],
                       "strict_pair_count": sum(dominates(a["metrics"], b["metrics"]) or dominates(b["metrics"], a["metrics"])
                                                 for i, a in enumerate(items) for b in items[i + 1:]),
                       "has_dominating_choice": dominating, "learned_index": learned["index"], "native_index": native["index"],
                       "learned_is_dominated": any(dominates(other["metrics"], learned["metrics"]) for other in items),
                       "native_is_dominated": any(dominates(other["metrics"], native["metrics"]) for other in items)})
    support_scenes = sorted({x["scene_id"] for x in train_items if any(dominates(a["metrics"], b["metrics"]) or dominates(b["metrics"], a["metrics"])
                                                                         for a in [x] for b in train_items if b["episode_id"] == x["episode_id"] and b["step"] == x["step"] and b is not x)})
    result = {"analysis": "transfer of frozen pairwise graph preference to separate training cohort", "feature_names": list(FEATURE_NAMES),
              "train_states": len(train_groups), "train_actions": len(train_items), "train_strict_pairs": len(train_pairs),
              "train_pair_support_scenes": len(support_scenes), "confirmation_states": len(confirmation),
              "confirmation_scenes": len({x["scene_id"] for g in groups for x in [{"scene_id": g["scene_id"]}]}),
              "confirmation_strict_pairs": sum(g["strict_pair_count"] for g in groups),
              "pair_accuracy": {k: (float(np.mean(v)) if v else None) for k, v in pair_correct.items()},
              "pair_accuracy_counts": {k: len(v) for k, v in pair_correct.items()}, "groups": groups,
              "uses_unseen_labels": False, "uses_future_features": False, "learned_model_used_in_navigation": False,
              "decision": "CONDITIONAL GO for a larger prospective confirmation" if len({x["scene_id"] for x in train_items}) >= 6 else "NO-GO",
              "limits": ["Confirmation is an existing separate training diagnostic, not untouched validation.", "Only two frozen alternatives per state; predicted choices were not executed.", "No deployable gain or benchmark claim."],
              "hashes": {str(args.train_plan): hashlib.sha256(args.train_plan.read_bytes()).hexdigest(), str(args.train_run / "results.jsonl"): hashlib.sha256((args.train_run / "results.jsonl").read_bytes()).hexdigest(), str(args.confirmation_root / "full_return_analysis_001/summary.json"): hashlib.sha256((args.confirmation_root / "full_return_analysis_001/summary.json").read_bytes()).hexdigest(), str(Path(__file__)): hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as f: json.dump(result, f, indent=2); f.write("\n")
    print(json.dumps({k: result[k] for k in ["train_states", "train_strict_pairs", "confirmation_states", "confirmation_scenes", "confirmation_strict_pairs", "pair_accuracy", "decision"]}, indent=2))


if __name__ == "__main__": main()
