#!/usr/bin/env python3
"""Audit a sampled route manifest against every declared exclusion source."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def route_key(record):
    scene = str(record["scene_id"]).split("data/scene_datasets/")[-1]
    return scene, str(record["trajectory_id"])


def records(path):
    path = Path(path)
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.open() if line.strip()]
    with (gzip.open(str(path), "rt") if path.suffix == ".gz" else path.open()) as f:
        value = json.load(f)
    if isinstance(value, dict):
        return value.get("route_episode_ids", value.get("episodes", []))
    return value


def source_records(path, by_episode):
    for row in records(path):
        if row.get("scene_id") and row.get("trajectory_id") is not None:
            yield row
        elif row.get("episode_id") is not None and str(row["episode_id"]) in by_episode:
            matched = by_episode[str(row["episode_id"])]
            if row.get("scene_id") and route_key(dict(matched, scene_id=row['scene_id'])) != route_key(matched):
                raise ValueError('Episode/scene identity mismatch in ' + str(path))
            yield matched
        else:
            yield row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--extra", type=Path, action="append", default=[])
    args = ap.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if args.output.exists():
        raise ValueError('Audit output already exists')
    if sha256(manifest['source']) != manifest['source_sha256']:
        raise ValueError('Source dataset hash mismatch')
    if sha256(manifest['dataset_path']) != manifest['dataset_sha256']:
        raise ValueError('Sample dataset hash mismatch')
    with gzip.open(manifest['dataset_path'], 'rt') as stream:
        actual = json.load(stream)['episodes']
    identity = lambda e: (str(e['episode_id']), route_key(e))
    if sorted(map(identity, actual)) != sorted(map(identity, manifest['episodes'])):
        raise ValueError('Dataset/manifest population mismatch')
    selected = {route_key(x) for x in manifest["episodes"]}
    if len(selected) != len(actual):
        raise ValueError('Repeated routes')
    source_dataset = Path(manifest["source"])
    with gzip.open(str(source_dataset), "rt") as f:
        source_data = json.load(f)
    by_episode = {str(x["episode_id"]): x for x in source_data["episodes"]}
    sources = list(manifest.get("excluded_sources", {}).keys()) + [str(x) for x in args.extra]
    overlaps = []
    source_summary = []
    for source in sources:
        p = Path(source)
        if not p.exists():
            source_summary.append({"source": source, "missing": True})
            continue
        declared = manifest.get('excluded_sources', {}).get(source, {}).get('sha256')
        if declared and declared != sha256(p):
            raise ValueError('Exclusion source hash mismatch: ' + source)
        local = set()
        unresolved = []
        for row in source_records(p, by_episode):
            if row.get("scene_id") and row.get("trajectory_id") is not None:
                local.add(route_key(row))
            else:
                unresolved.append({"episode_id": row.get("episode_id"), "keys": sorted(row)})
        hit = sorted(selected & local)
        overlaps.extend({"source": source, "scene_id": s, "trajectory_id": t} for s, t in hit)
        source_summary.append({"source": source, "records": len(records(p)), "routes": len(local),
                               "overlap_routes": len(hit), "unresolved_records": len(unresolved),
                               "sha256": sha256(p)})
    result = {
        "manifest": str(args.manifest),
        "manifest_sha256": sha256(args.manifest),
        "selected_routes": len(selected),
        "sources": source_summary,
        "overlaps": overlaps,
        "overlap_route_count": len({(x["scene_id"], x["trajectory_id"]) for x in overlaps}),
        "valid_for_confirmation": not overlaps and all(not x.get("missing") and not x.get("unresolved_records") for x in source_summary),
        "audit_script_sha256": sha256(__file__),
        "interpretation": "Validity covers declared source routes only, not scene independence or checkpoint training holdout. Nonzero overlap invalidates route-disjoint confirmation.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["valid_for_confirmation"] else 2)


if __name__ == "__main__":
    main()
