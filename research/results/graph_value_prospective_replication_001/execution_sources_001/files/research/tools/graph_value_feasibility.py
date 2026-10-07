#!/usr/bin/env python3
"""Outcome-blind feature feasibility for full-return graph-action preference.

This is deliberately a small diagnostic.  It fits a linear pairwise preference
model only on the frozen training failure cohort and evaluates leave-one-scene-
out.  Full-return metrics provide labels, but never enter the feature vector.
Mixed and tied action outcomes are retained and excluded from strict-dominance
pair fitting rather than being forced into a scalar reward.
"""
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path

import numpy as np


METRICS = ("success", "spl", "ndtw", "distance_to_goal", "path_length", "primitive_action_count")
FEATURE_NAMES = (
    "graph_logit", "logit_rank_percentile", "policy_entropy", "stop_probability",
    "candidate_count", "action_is_stop", "current_proposal", "ghost_distance_m",
    "back_path_length_m", "back_path_nodes", "embedding_norm", "embedding_mean",
    "embedding_std",
)
TOL = 1e-6


def val(value):
    if isinstance(value, dict) and "__array__" in value:
        return np.asarray(value["__array__"], dtype=np.float64)
    if isinstance(value, (list, tuple)):
        return np.asarray(value, dtype=np.float64)
    return np.asarray(value, dtype=np.float64)


def position(value):
    return val(value).reshape(-1)[:3]


def entropy(logits):
    x = np.asarray(logits, dtype=np.float64)
    x = x - np.max(x)
    p = np.exp(x)
    p /= p.sum()
    return float(-(p * np.log(np.maximum(p, 1e-12))).sum()), p


def action_features(row, option):
    if row.get("privileged_labels_in_features", False) or not option.get("admissible"):
        raise ValueError("Features require an admissible, nonprivileged option")
    admissible = [o for o in row["options"] if o.get("admissible")]
    logits = [float(o["logit"]) for o in admissible]
    if not np.all(np.isfinite(logits)):
        raise ValueError("Nonfinite admissible logits")
    logit = float(option["logit"])
    ordered = sorted(logits)
    rank = ordered.index(logit) if logit in ordered else sum(x <= logit for x in ordered) - 1
    rank_pct = float(rank / max(1, len(ordered) - 1))
    ent, probs = entropy(logits)
    # A state feature, identical for every candidate; index only the same
    # admissible sequence used to form the softmax (visited nodes may be absent).
    prob = sum(float(p) for o, p in zip(admissible, probs) if int(o["action"]["act"]) == 0)
    action = option["action"]
    stop = int(action.get("act", 4)) == 0
    if stop:
        ghost_distance = 0.0
    else:
        ghost_distance = float(np.linalg.norm(position(action["ghost_pos"]) - position(action["front_pos"])))
    back = action.get("back_path", []) or []
    back_len = 0.0
    # Native back_path is ordered from the current pose, including for STOP.
    prev = position(row["graph_position"])
    for item in back:
        pair = item.get("__tuple__", [])
        if prev is not None and len(pair) == 2:
            nxt = position(pair[1]); back_len += float(np.linalg.norm(nxt - prev)); prev = nxt
    emb = option.get("embedding", {}).get("__array__", [])
    e = np.asarray(emb, dtype=np.float64).reshape(-1)
    return np.asarray([
        logit, rank_pct, ent, prob, len(logits), float(stop),
        float(bool(option.get("current_proposal", False))), ghost_distance,
        back_len, len(back), float(np.linalg.norm(e)) if e.size else 0.0,
        float(e.mean()) if e.size else 0.0, float(e.std()) if e.size else 0.0,
    ], dtype=np.float64)


def dominates(a, b):
    # success/SPL/nDTW are higher; distance/path/primitive cost are lower.
    better_or_equal = (
        a["success"] >= b["success"] - TOL and a["spl"] >= b["spl"] - TOL and
        a["ndtw"] >= b["ndtw"] - TOL and a["distance_to_goal"] <= b["distance_to_goal"] + TOL and
        a["path_length"] <= b["path_length"] + TOL and
        a["primitive_action_count"] <= b["primitive_action_count"]
    )
    strict = (
        a["success"] > b["success"] + TOL or a["spl"] > b["spl"] + TOL or
        a["ndtw"] > b["ndtw"] + TOL or a["distance_to_goal"] < b["distance_to_goal"] - TOL or
        a["path_length"] < b["path_length"] - TOL or
        a["primitive_action_count"] < b["primitive_action_count"]
    )
    return bool(better_or_equal and strict)


def fit_pairwise(rows, field="z"):
    x, y = [], []
    for left, right in rows:
        d = left[field] - right[field]
        x.append(d); y.append(1.0)
    for left, right in rows:
        d = right[field] - left[field]
        x.append(d); y.append(-1.0)
    if not x:
        return None
    xx = np.asarray(x); yy = np.asarray(y)
    return np.linalg.solve(xx.T @ xx + 10.0 * np.eye(xx.shape[1]), xx.T @ yy)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", type=Path, required=True)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    plan = json.loads(args.plan.read_text())
    graph_path = Path(plan["config"]["source_root"]) / "capture_001/graph_options/graph_options.jsonl"
    graphs = {}
    for line in graph_path.open():
        row = json.loads(line); graphs[(str(row["episode_id"]), int(row["high_level_step"]))] = row
    results = [json.loads(line) for line in (args.run / "results.jsonl").open()]
    action_rows = [r for r in results if r["case"]["mode"] == "action"]
    by_state = collections.defaultdict(list)
    for result in action_rows:
        c = result["case"]; key = (str(c["episode_id"]), int(c["high_level_step"]))
        row = graphs[key]; option = next(o for o in row["options"] if int(o["index"]) == int(c["action_index"]))
        metrics = dict(result["metrics"])
        # Critical census metrics call this quantity `steps_taken`; the
        # preference contract treats primitive count as the same executed count.
        metrics["primitive_action_count"] = int(round(metrics.get("primitive_action_count", metrics["steps_taken"])))
        by_state[key].append({"episode_id": key[0], "scene_id": row["scene_id"], "step": key[1],
                              "index": int(c["action_index"]), "features": action_features(row, option),
                              "metrics": metrics, "delta": result["delta"],
                              "logit": float(option["logit"])})
    all_items = [item for items in by_state.values() for item in items]
    mean = np.mean([x["features"] for x in all_items], axis=0)
    scale = np.std([x["features"] for x in all_items], axis=0); scale[scale < 1e-8] = 1.0
    for item in all_items: item["z"] = (item["features"] - mean) / scale
    pair_rows = []
    for key, items in by_state.items():
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                if dominates(items[i]["metrics"], items[j]["metrics"]): pair_rows.append((items[i], items[j]))
                elif dominates(items[j]["metrics"], items[i]["metrics"]): pair_rows.append((items[j], items[i]))
    scenes = sorted({item["scene_id"] for item in all_items})
    folds = []
    pair_accuracy = {"learned": [], "native_logit": []}
    state_summary = []
    for held in scenes:
        train_pairs = [(a, b) for a, b in pair_rows if a["scene_id"] != held and b["scene_id"] != held]
        val_pairs = [(a, b) for a, b in pair_rows if a["scene_id"] == held or b["scene_id"] == held]
        train_items = [x for x in all_items if x["scene_id"] != held]
        if not train_pairs:
            folds.append({"held_out_scene": held, "status": "unsupported_no_train_pairs", "train_pairs": 0, "val_pairs": len(val_pairs)})
            continue
        # Standardization is refit inside each fold; labels never affect features.
        fold_mean = np.mean([x["features"] for x in train_items], axis=0)
        fold_scale = np.std([x["features"] for x in train_items], axis=0); fold_scale[fold_scale < 1e-8] = 1.0
        for x in all_items: x["fold_z"] = (x["features"] - fold_mean) / fold_scale
        train_pairs_z = [(a, b) for a, b in train_pairs]
        w = fit_pairwise(train_pairs_z, field="fold_z")
        for a, b in val_pairs:
            pair_accuracy["learned"].append(float(np.dot(w, a["fold_z"] - b["fold_z"]) > 0))
            pair_accuracy["native_logit"].append(float(a["logit"] > b["logit"]))
        held_states = [items for items in by_state.values() if items[0]["scene_id"] == held]
        dominated_learned = dominated_native = 0; quality_learned = quality_native = 0; available = 0
        for items in held_states:
            learned_pick = max(items, key=lambda x: float(np.dot(w, x["fold_z"])))
            native_pick = max(items, key=lambda x: x["logit"])
            best_quality = [x for x in items if x["metrics"]["success"] >= 1.0 and x["delta"]["ndtw"] >= -TOL]
            if best_quality: available += 1
            if any(dominates(other["metrics"], learned_pick["metrics"]) for other in items): dominated_learned += 1
            if any(dominates(other["metrics"], native_pick["metrics"]) for other in items): dominated_native += 1
            quality_learned += int(learned_pick in best_quality)
            quality_native += int(native_pick in best_quality)
        folds.append({"held_out_scene": held, "status": "evaluated", "train_pairs": len(train_pairs), "val_pairs": len(val_pairs),
                      "states": len(held_states), "states_with_route_quality_action": available,
                      "learned_dominated_choice_rate": dominated_learned / len(held_states) if held_states else None,
                      "native_dominated_choice_rate": dominated_native / len(held_states) if held_states else None,
                      "learned_route_quality_pick_rate": quality_learned / available if available else None,
                      "native_route_quality_pick_rate": quality_native / available if available else None})
    support_scenes = sorted({a["scene_id"] for a, b in pair_rows} | {b["scene_id"] for a, b in pair_rows})
    result = {
        "experiment_id": plan["config"]["experiment_id"], "analysis": "scene-grouped linear pairwise preference feasibility",
        "feature_names": list(FEATURE_NAMES), "feature_schema_version": 2,
        "feature_contract": "CRITICAL_GRAPH_VALUE_TRAINNEW_PROTOCOL.md",
        "states": len(by_state), "actions": len(all_items), "strict_dominance_pairs": len(pair_rows),
        "pair_support_scenes": len(support_scenes), "pair_support_scene_ids": support_scenes,
        "folds": folds, "pair_accuracy": {k: (float(np.mean(v)) if v else None) for k, v in pair_accuracy.items()},
        "pair_accuracy_counts": {k: len(v) for k, v in pair_accuracy.items()},
        "learning_support_gate_passed": len(support_scenes) >= 4,
        "uses_unseen_labels": False, "uses_future_features": False, "learned_model_used_in_navigation": False,
        "decision": "CONDITIONAL GO for a separately registered held-out confirmation" if len(support_scenes) >= 4 else "NO-GO for fitting; strict-dominance support is too sparse",
        "limits": ["One seed and retrospective failure cohort; this is feasibility evidence only.", "Strict dominance discards mixed/tied outcomes.", "Predicted actions were not executed; no deployable gain is claimed.", "Unseen outcomes were not read."],
        "hashes": {str(args.plan): hashlib.sha256(args.plan.read_bytes()).hexdigest(), str(graph_path): hashlib.sha256(graph_path.read_bytes()).hexdigest(), str(args.run / "results.jsonl"): hashlib.sha256((args.run / "results.jsonl").read_bytes()).hexdigest(), str(Path(__file__)): hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream: stream.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ["states", "actions", "strict_dominance_pairs", "pair_support_scenes", "pair_accuracy", "learning_support_gate_passed", "decision"]}, indent=2))


if __name__ == "__main__": main()
