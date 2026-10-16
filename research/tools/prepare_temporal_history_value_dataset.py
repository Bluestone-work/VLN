#!/usr/bin/env python3
"""Add short native pre-action history context to the graph-value dataset."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from prepare_graph_history_value_dataset import graph_features


def state_summary(rows):
    first = rows[0]
    f = first["features"]
    ids = first.get("graph_ids") or []
    selected = int(first["native_index"])
    candidate = ids[selected] if 0 <= selected < len(ids) else None
    return [float(f.get("graph_size", 0.0)), float(f.get("back_path_length", 0.0)),
            float(sum(x is not None and not str(x).startswith("g") for x in ids)),
            float(sum(x is not None and str(x).startswith("g") for x in ids)),
            float(first["high_level_step"]), float(selected),
            float(candidate is None), float(candidate is not None and str(candidate).startswith("g")),
            float(f.get("native_selected_logit", 0.0)), float(f.get("native_logit_margin", 0.0)),
            float(f.get("current_proposal", False)), float(f.get("is_stop", False))]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--pairs", type=Path, required=True); ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args(); source = [json.loads(x) for x in args.pairs.open() if x.strip()]
    states = defaultdict(list)
    for row in source:
        if row["label"] == "UNCERTAIN": continue
        states[(row["scene_id"], str(row["episode_id"]), int(row["high_level_step"]))].append(row)
    by_episode = defaultdict(list)
    for key, rows in states.items(): by_episode[(key[0], key[1])].append((key[2], rows))
    history = {}
    for episode, seq in by_episode.items():
        seq.sort(); past = []
        for step, rows in seq:
            history[(episode[0], episode[1], step)] = list(past[-2:])
            past.append(state_summary(rows))
    out = []
    for key, rows in sorted(states.items()):
        for row in sorted(rows, key=lambda x: int(x["alternative_index"])):
            base = graph_features(row, rows)
            hist = []
            for summary in history[key]: hist.extend(summary)
            while len(hist) < 24: hist.append(0.0)
            f = row["features"]
            native = [f["native_logit"], f["native_selected_logit"], f["native_logit_margin"],
                      float(f.get("is_stop", False)), float(f.get("current_proposal", False)),
                      f["graph_size"], f.get("back_path_length", 0.0)] + list(f.get("native_embedding", []))
            import math
            bearing = f.get("bearing_deg")
            geometry = [0.0, 0.0, f.get("distance", 0.0), 0.0] if bearing is None else [
                math.sin(math.radians(float(bearing))), math.cos(math.radians(float(bearing))), f.get("distance", 0.0), 0.0]
            out.append({"pair_id": row["pair_id"], "scene_id": key[0], "episode_id": key[1],
                        "high_level_step": key[2], "state_key": "{}|{}|{}".format(*key),
                        "label": row["label"], "native": native, "geometry": geometry,
                        "graph": base + hist})
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "rows.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in out))
    import numpy as np
    np.savez_compressed(str(args.output / "features.npz"), graph=np.asarray([r["graph"] for r in out], dtype=np.float32),
                        native=np.asarray([r["native"] for r in out], dtype=np.float32), geometry=np.asarray([r["geometry"] for r in out], dtype=np.float32),
                        labels=np.asarray([r["label"] == "BENEFIT" for r in out], dtype=np.uint8))
    manifest = {"schema_version": 1, "source_pairs": str(args.pairs), "rows": len(out),
                "states": len(states), "scenes": len({r["scene_id"] for r in out}),
                "history": "previous one and two native pre-action state summaries; zero padded",
                "feature_dimensions": {"graph": len(out[0]["graph"]), "native": len(out[0]["native"]), "geometry": len(out[0]["geometry"])},
                "label_features_separation": "outcomes and endpoint observations are targets only"}
    (args.output / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__": main()
