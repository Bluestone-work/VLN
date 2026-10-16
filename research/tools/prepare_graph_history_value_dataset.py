#!/usr/bin/env python3
"""Build an outcome-blind graph/history waypoint-value dataset.

The source is the frozen native single-action full-return table.  Only fields
available before executing an action are used as features.  BENEFIT/HARM are
targets and are never copied into the feature vector.
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def numeric_id(value):
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def graph_features(row, state_rows):
    f = row["features"]
    ids = row.get("graph_ids") or []
    nonnull = [x for x in ids if x is not None]
    visited = [x for x in nonnull if not str(x).startswith("g")]
    ghosts = [x for x in nonnull if str(x).startswith("g")]
    current = numeric_id(row.get("current_vp"))
    idx = int(row["alternative_index"])
    candidate = ids[idx] if 0 <= idx < len(ids) else None
    cand_num = numeric_id(candidate)
    cand_ghost = str(candidate).startswith("g") if candidate is not None else False
    cand_visited = cand_num is not None
    cand_current = cand_num is not None and current is not None and cand_num == current
    cand_back = cand_num is not None and current is not None and cand_num < current
    age = (current - cand_num) if cand_num is not None and current is not None else 0
    ghost_num = numeric_id(str(candidate)[1:]) if cand_ghost else None
    # These are structural graph/history fields only.  Native logits and the
    # visual/language embedding stay in F0 and are added by the probe script.
    return [
        float(len(visited)),
        float(len(ghosts)),
        float(len(nonnull)),
        float(current if current is not None else -1),
        float(row["high_level_step"]),
        float(f.get("back_path_length", 0.0)),
        float(len(state_rows) - 1),
        float(idx),
        float(idx / max(len(ids) - 1, 1)),
        float(candidate is None),
        float(cand_visited),
        float(cand_current),
        float(cand_back),
        float(cand_ghost),
        float(age),
        float(ghost_num if ghost_num is not None else -1),
        float(bool(f.get("current_proposal", False))),
        float(bool(f.get("is_stop", False))),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    rows = [json.loads(line) for line in args.pairs.open() if line.strip()]
    groups = defaultdict(list)
    for row in rows:
        if row["label"] == "UNCERTAIN":
            continue
        key = (row["scene_id"], str(row["episode_id"]), int(row["high_level_step"]))
        groups[key].append(row)
    out_rows = []
    for key, state_rows in sorted(groups.items()):
        scene, episode, step = key
        for row in sorted(state_rows, key=lambda x: int(x["alternative_index"])):
            f = row["features"]
            native = [
                f["native_logit"], f["native_selected_logit"],
                f["native_logit_margin"], float(f.get("is_stop", False)),
                float(f.get("current_proposal", False)), f["graph_size"],
                f.get("back_path_length", 0.0),
            ] + list(f.get("native_embedding", []))
            bearing = f.get("bearing_deg")
            if bearing is None:
                geometry = [0.0, 0.0, f.get("distance", 0.0), 0.0]
            else:
                import math
                theta = math.radians(float(bearing))
                geometry = [math.sin(theta), math.cos(theta), f.get("distance", 0.0), 0.0]
            out_rows.append({
                "pair_id": row["pair_id"], "scene_id": scene,
                "episode_id": episode, "high_level_step": step,
                "state_key": "{}|{}|{}".format(scene, episode, step),
                "label": row["label"], "native_action_index": row["alternative_index"],
                "native_selected_index": row["native_index"],
                "graph": graph_features(row, state_rows),
                "native": native, "geometry": geometry,
                "graph_id": (row.get("graph_ids") or [None])[int(row["alternative_index"])]
                             if int(row["alternative_index"]) < len(row.get("graph_ids") or []) else None,
            })
    if not out_rows:
        raise RuntimeError("No usable rows")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "rows.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in out_rows))
    import numpy as np
    np.savez_compressed(str(args.output / "features.npz"),
                        graph=np.asarray([r["graph"] for r in out_rows], dtype=np.float32),
                        native=np.asarray([r["native"] for r in out_rows], dtype=np.float32),
                        geometry=np.asarray([r["geometry"] for r in out_rows], dtype=np.float32),
                        labels=np.asarray([r["label"] == "BENEFIT" for r in out_rows], dtype=np.uint8))
    states = sorted({r["state_key"] for r in out_rows})
    scenes = sorted({r["scene_id"] for r in out_rows})
    manifest = {
        "schema_version": 1, "source_pairs": str(args.pairs),
        "rows": len(out_rows), "states": len(states), "scenes": len(scenes),
        "scene_ids": scenes, "labels": Counter(r["label"] for r in out_rows),
        "mixed_states": sum({r["label"] for r in rs} == {"BENEFIT", "HARM"}
                             for rs in groups.values()),
        "feature_dimensions": {"graph": len(out_rows[0]["graph"]),
                                "native": len(out_rows[0]["native"]),
                                "geometry": len(out_rows[0]["geometry"])},
        "graph_feature_names": [
            "visited_count", "ghost_count", "nonnull_graph_size", "current_vp_index",
            "high_level_step", "back_path_length", "state_candidate_count",
            "action_index", "action_index_fraction", "candidate_is_stop",
            "candidate_is_visited", "candidate_is_current", "candidate_is_backtrack",
            "candidate_is_ghost", "candidate_history_age", "candidate_ghost_index",
            "current_proposal", "is_stop"],
        "split": "leave-one-scene-out; states never split across folds",
        "label_features_separation": "BENEFIT/HARM are targets only; no endpoint, goal, reference, or final outcome feature",
    }
    (args.output / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2, default=list) + "\n")
    print(json.dumps(manifest, indent=2, default=list))


if __name__ == "__main__":
    main()
