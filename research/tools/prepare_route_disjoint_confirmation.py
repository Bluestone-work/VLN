#!/usr/bin/env python3
"""Metadata-only route-disjoint sample for a scene-overlap confirmation."""
import argparse
import collections
import gzip
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def route_key(episode):
    return (episode["scene_id"].split("data/scene_datasets/")[-1], str(episode["trajectory_id"]))


def listed_records(path):
    path = Path(path)
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.open()]
    with (gzip.open(str(path), "rt") if path.suffix == ".gz" else path.open()) as stream:
        obj = json.load(stream)
    if isinstance(obj, dict):
        records = obj.get("route_episode_ids", obj.get("episodes", []))
        if isinstance(records, int):
            return []
        return records
    return obj


def record_route(record, by_episode, source_path):
    """Resolve one exclusion record without silently dropping provenance."""
    if record.get("scene_id") and record.get("trajectory_id") is not None:
        return route_key(record)
    if record.get("episode_id") is not None and str(record["episode_id"]) in by_episode:
        matched = by_episode[str(record["episode_id"])]
        if record.get("scene_id") and route_key(dict(matched, scene_id=record["scene_id"])) != route_key(matched):
            raise ValueError("Episode/scene identity mismatch in " + str(source_path))
        return route_key(matched)
    raise ValueError("Cannot resolve excluded route in " + str(source_path) + ": " + repr(record))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text())
    source = Path(cfg["source"])
    with gzip.open(str(source), "rt") as stream:
        data = json.load(stream)
    all_episodes = data["episodes"]
    by_episode = {str(e["episode_id"]): e for e in all_episodes}
    excluded = set()
    source_meta = {}
    for raw_path in cfg["exclude_sources"]:
        path = Path(raw_path)
        if not path.exists():
            raise ValueError("Missing route exclusion source: " + str(path))
        records = listed_records(path)
        local = set()
        for record in records:
            local.add(record_route(record, by_episode, path))
        if not local:
            raise ValueError("Empty route exclusion source: " + str(path))
        excluded |= local
        source_meta[str(path)] = {"records": len(records), "routes": len(local), "sha256": digest(path)}

    available = [e for e in all_episodes if route_key(e) not in excluded]
    grouped = collections.defaultdict(lambda: collections.defaultdict(list))
    for episode in available:
        grouped[episode["scene_id"]][str(episode["trajectory_id"])].append(episode)
    routes_per_scene = int(cfg["routes_per_scene"])
    scenes = [scene for scene, routes in grouped.items() if len(routes) >= routes_per_scene]
    scenes.sort(key=lambda scene: hashlib.sha256(
        "{}|{}".format(cfg["selection_seed"], scene).encode()).hexdigest())
    scenes = scenes[: int(cfg["scenes"])]
    if len(scenes) < int(cfg["scenes"]):
        raise ValueError("Not enough scenes with route-disjoint support")
    selected = []
    for scene in scenes:
        routes = grouped[scene]
        ordered = sorted(routes, key=lambda route: hashlib.sha256(
            "{}|{}|{}".format(cfg["selection_seed"], scene, route).encode()).hexdigest())
        for route in ordered[:routes_per_scene]:
            selected.append(min(routes[route], key=lambda e: str(e["episode_id"])))
    selected_keys = {route_key(e) for e in selected}
    if selected_keys & excluded:
        raise AssertionError("Route exclusion failed")
    if len(selected_keys) != len(selected):
        raise AssertionError("Duplicate selected route")

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=False)
    dataset_path = output_dir / cfg.get("dataset_filename", "route_disjoint_subset.json.gz")
    subset = dict(data)
    subset["episodes"] = selected
    with dataset_path.open("xb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as zipped:
            zipped.write(json.dumps(subset, ensure_ascii=False).encode("utf8"))
    manifest = {
        "experiment_id": cfg["experiment_id"],
        "source": str(source),
        "source_sha256": digest(source),
        "dataset_path": str(dataset_path),
        "dataset_sha256": digest(dataset_path),
        "selection_seed": cfg["selection_seed"],
        "scenes": scenes,
        "routes_per_scene": routes_per_scene,
        "selected_routes": len(selected_keys),
        "selected_episodes": len(selected),
        "excluded_routes": len(excluded),
        "selection_rule": "metadata-only seeded route order after exact prior-route exclusion; no outcomes used",
        "scene_overlap_allowed": True,
        "route_overlap": len(selected_keys & excluded),
        "scene_counts": dict(collections.Counter(e["scene_id"] for e in selected)),
        "episodes": [{"episode_id": str(e["episode_id"]), "scene_id": e["scene_id"],
                      "trajectory_id": str(e["trajectory_id"])} for e in selected],
        "excluded_sources": source_meta,
        "source_script_sha256": digest(Path(__file__)),
        "config_sha256": digest(args.config),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "episodes"}, indent=2))


if __name__ == "__main__":
    main()
