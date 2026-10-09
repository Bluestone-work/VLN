#!/usr/bin/env python3
"""Metadata-only audit for a fresh long-horizon graph-value route pool."""
import argparse
import collections
import gzip
import hashlib
import json
import subprocess
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def scene(value):
    return str(value).split("data/scene_datasets/")[-1]


def route(row):
    return scene(row["scene_id"]), str(row["trajectory_id"])


def read_rows(path):
    path = Path(path)
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.open() if line.strip()]
    with (gzip.open(str(path), "rt") if path.suffix == ".gz" else path.open()) as f:
        value = json.load(f)
    if isinstance(value, dict):
        rows = value.get("episodes", value.get("route_episode_ids", []))
        return rows if isinstance(rows, list) else []
    return value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text())
    source = Path(cfg["sampling"]["source"])
    source_rows = read_rows(source)
    excluded_routes = set()
    excluded_scenes = set()
    source_summary = []
    for item in cfg["sampling"]["exclude_sources"]:
        path = Path(item)
        rows = read_rows(path)
        metadata_scenes = set()
        if path.suffix != ".jsonl":
            with path.open() as stream:
                raw = json.load(stream)
            if isinstance(raw, dict):
                metadata_scenes = {scene(x) for x in raw.get("scenes", [])}
                metadata_scenes.update(scene(x) for x in raw.get("excluded_scenes", []))
        local_routes = set()
        local_scenes = set(metadata_scenes)
        for row in rows:
            if row.get("scene_id") and row.get("trajectory_id") is not None:
                local_routes.add(route(row)); local_scenes.add(scene(row["scene_id"]))
            elif row.get("scene_id"):
                local_scenes.add(scene(row["scene_id"]))
        excluded_routes |= local_routes
        excluded_scenes |= local_scenes
        source_summary.append({"source": str(path), "sha256": sha256(path),
                               "rows": len(rows), "routes": len(local_routes),
                               "scenes": len(local_scenes)})
    available = [row for row in source_rows if route(row) not in excluded_routes and scene(row["scene_id"]) not in excluded_scenes]
    grouped = collections.defaultdict(set)
    for row in available:
        grouped[scene(row["scene_id"])].add(str(row["trajectory_id"]))
    eligible = {s: sorted(routes) for s, routes in grouped.items()
                if len(routes) >= int(cfg["sampling"]["routes_per_scene"])}
    result = {
        "experiment_id": "LONG-HORIZON-GRAPH-VALUE-POOL-AUDIT-002",
        "status": "blocked_before_sampling" if len(eligible) < int(cfg["sampling"]["minimum_scenes"]) else "pool_available",
        "source": str(source), "source_sha256": sha256(source),
        "requested_minimum_scenes": cfg["sampling"]["minimum_scenes"],
        "requested_minimum_routes": cfg["sampling"]["minimum_routes"],
        "configured_routes_per_scene": cfg["sampling"]["routes_per_scene"],
        "excluded_scene_count": len(excluded_scenes), "excluded_route_count": len(excluded_routes),
        "available_scene_count": len(grouped), "available_route_count": sum(len(x) for x in grouped.values()),
        "eligible_scene_count": len(eligible), "eligible_routes_by_scene": eligible,
        "sources": source_summary,
        "decision": "NO-GO for independent scene-disjoint confirmation with current R2R train pool" if len(eligible) < int(cfg["sampling"]["minimum_scenes"]) else "Pool satisfies metadata gate; freeze sample before outcomes",
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip(),
        "config_sha256": sha256(args.config), "script_sha256": sha256(Path(__file__)),
    }
    args.output_dir.mkdir(parents=False, exist_ok=False)
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
