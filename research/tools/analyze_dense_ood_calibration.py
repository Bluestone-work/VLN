#!/usr/bin/env python3
"""Decompose A0/A1 dense harm into native contamination and new-action ranking.

Only matched prefix states before the first selected-action divergence are used.
This is a read-only analysis of accepted A0/A1 captures; no outcomes enter the
feature computation or any threshold.
"""
import argparse
import csv
import hashlib
import json
import math
import subprocess
from pathlib import Path


def rows(path):
    with Path(path).open() as f:
        return [json.loads(x) for x in f if x.strip()]


def target(option):
    a = option["action"]
    def pos(key):
        value = a[key]["__array__"]
        return tuple(round(float(x), 5) for x in value)
    if int(a["act"]) == 0:
        return ("stop", pos("stop_pos"))
    return ("move", pos("ghost_pos"))


def vector(option):
    return [float(x) for x in option.get("embedding", {}).get("__array__", [])]


def l2(a, b):
    if not a or not b or len(a) != len(b):
        return None
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def maxabs(a, b):
    if not a or not b or len(a) != len(b):
        return None
    return max(abs(x - y) for x, y in zip(a, b))


def mean(values):
    values = [x for x in values if x is not None]
    return sum(values) / len(values) if values else None


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def analyze_pair(a0_dir, a1_dir, route_roles, cohort):
    left = {(str(x["episode_id"]), int(x["high_level_step"])): x for x in rows(a0_dir / "graph_options/graph_options.jsonl")}
    right = {(str(x["episode_id"]), int(x["high_level_step"])): x for x in rows(a1_dir / "graph_options/graph_options.jsonl")}
    out = []
    for episode in sorted({k[0] for k in left} & {k[0] for k in right}, key=int):
        keys = sorted(k for k in left if k[0] == episode and k in right)
        for key in keys:
            x, y = left[key], right[key]
            xa = {target(o): o for o in x["options"] if o.get("admissible")}
            ya = {target(o): o for o in y["options"] if o.get("admissible")}
            common = set(xa) & set(ya)
            native_targets = {target(o) for o in x["options"] if o.get("admissible")}
            selected_same = target(next(o for o in x["options"] if int(o["index"]) == int(x["effective_index"]))) == target(next(o for o in y["options"] if int(o["index"]) == int(y["effective_index"])))
            role = route_roles.get(episode, "unknown")
            native_deltas = []
            for ident in common:
                a, b = xa[ident], ya[ident]
                native_deltas.append({
                    "target": repr(ident), "logit_delta": float(b["logit"]) - float(a["logit"]),
                    "embedding_l2": l2(vector(a), vector(b)), "embedding_maxabs": maxabs(vector(a), vector(b)),
                    "is_native": ident in native_targets,
                })
            out.append({
                "cohort": cohort, "episode_id": episode, "high_level_step": key[1], "role": role,
                "selected_same": selected_same, "native_action_count": len(native_targets),
                "dense_action_count": sum(1 for o in y["options"] if o.get("admissible") and target(o) not in native_targets),
                "common_action_count": len(common), "native_deltas": native_deltas,
            })
            if not selected_same:
                break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    manifest = json.loads((args.root / "manifest.json").read_text())
    route_roles = {str(x["episode_id"]): x.get("role", "unknown") for x in manifest.get("route_rows", [])}
    pairs = [
        ("train", args.root / "runs/train_a0", args.root / "runs/train_a1"),
        ("val_unseen", args.root / "runs/unseen_a0", args.root / "runs/unseen_a1"),
    ]
    state_rows = []
    source_files = [args.root / "manifest.json"]
    for cohort, a0, a1 in pairs:
        state_rows.extend(analyze_pair(a0, a1, route_roles, cohort))
        source_files.extend([a0 / "graph_options/graph_options.jsonl", a1 / "graph_options/graph_options.jsonl"])
    native = [d for r in state_rows for d in r["native_deltas"] if d["is_native"]]
    all_common = [d for r in state_rows for d in r["native_deltas"]]
    divergent = [r for r in state_rows if not r["selected_same"]]
    dense_selected = [r for r in divergent if r["dense_action_count"] > 0]
    # Optional route-level outcome labels are only used for post-hoc stratified
    # reporting, never for the contamination measurements themselves.
    route_csv = args.root / "analysis_001/per_route.csv"
    if route_csv.exists():
        with route_csv.open() as f:
            for row in csv.DictReader(f):
                route_roles[str(row["episode_id"])] = "destroyed" if float(row["delta_success"]) < 0 else "retained"
    grouped = {}
    for group in ("retained", "destroyed"):
        subset = [r for r in state_rows if route_roles.get(r["episode_id"]) == group]
        vals = [d for r in subset for d in r["native_deltas"] if d["is_native"]]
        grouped[group] = {"states": len(subset), "divergent_states": sum(not r["selected_same"] for r in subset),
                          "native_option_comparisons": len(vals),
                          "native_logit_delta_abs_mean": mean([abs(x["logit_delta"]) for x in vals]),
                          "native_embedding_l2_mean": mean([x["embedding_l2"] for x in vals])}
    summary = {
        "experiment_id": "DENSE-OOD-CALIBRATION-AUDIT-001",
        "question": "H1 native representation contamination versus H2 new-action ranking/calibration",
        "state_count_before_first_divergence": len(state_rows),
        "divergent_states": len(divergent), "divergent_states_with_dense_action": len(dense_selected),
        "native_option_comparisons": len(native), "all_common_option_comparisons": len(all_common),
        "native_logit_delta_abs_mean": mean([abs(x["logit_delta"]) for x in native]),
        "native_logit_delta_abs_max": max([abs(x["logit_delta"]) for x in native], default=None),
        "native_embedding_l2_mean": mean([x["embedding_l2"] for x in native]),
        "native_embedding_l2_max": max([x["embedding_l2"] for x in native], default=None),
        "native_embedding_maxabs_mean": mean([x["embedding_maxabs"] for x in native]),
        "native_embedding_maxabs_max": max([x["embedding_maxabs"] for x in native], default=None),
        "posthoc_success_stratification": grouped,
        "method": "Matched prefixes only; stop at first selected-action divergence; no rollout or label fitting",
        "interpretation": "H1 supported if native embeddings/logits shift materially before divergence; H2 supported if native values remain stable while dense actions outrank them",
        "source_sha256": {str(p): digest(p) for p in source_files},
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip(),
    }
    args.output_dir.mkdir(parents=False, exist_ok=False)
    with (args.output_dir / "matched_states.jsonl").open("x") as f:
        for row in state_rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
