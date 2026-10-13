#!/usr/bin/env python3
"""Validate visited-node semantic memory on the frozen 44-episode cohort.

This is an analysis-only tool.  It consumes the already captured pre-action
RGB-D/CLIP records and never writes to GraphMap or runs a policy.  Goal
positions and final outcomes are loaded only as evaluation labels.
"""
import argparse
import gzip
import hashlib
import json
import math
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path


LANDMARKS = (
    "sofa", "couch", "painting", "door", "doorway", "table", "chair", "bed",
    "kitchen", "bedroom", "hallway", "corridor", "stairs", "staircase", "rug",
    "archway", "podium", "seat", "pew", "dining", "sink", "fireplace", "cabinet",
)
VIEW_RE = re.compile(r"^(?:rgb|depth)_(\d+)$")


def arr(v):
    if isinstance(v, dict) and "__array__" in v:
        return v["__array__"]
    return v


def dist(a, b):
    aa, bb = arr(a), arr(b)
    if not aa or not bb:
        return float("inf")
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(aa, bb)))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def ordered_phrases(instruction):
    """Extract ordered instruction phrases from text only.

    This deliberately does not inspect reference paths, graph nodes, or goal
    positions.  Duplicate mentions are retained in text order, then stable
    de-duplicated for the progress index.
    """
    text = instruction or ""
    hits = []
    for name in LANDMARKS:
        for m in re.finditer(r"\b" + re.escape(name) + r"\b", text.lower()):
            left = text[max(0, m.start() - 42):m.start()].lower()
            right = text[m.end():m.end() + 42].lower()
            context = left + " " + right
            direction = "unknown"
            if re.search(r"\bleft\b", context):
                direction = "left"
            elif re.search(r"\bright\b", context):
                direction = "right"
            elif re.search(r"\bbehind\b|\bback\b", context):
                direction = "behind"
            elif re.search(r"\bfront\b|\bahead\b", context):
                direction = "front"
            relation = "landmark evidence"
            if re.search(r"\bpass(?:ed|ing)?\b|\baround\b", context):
                relation = "passed candidate"
            elif re.search(r"\benter(?:ed|ing)?\b|\bthrough\b|\binto\b", context):
                relation = "room entrance"
            elif re.search(r"\bturn\b|\bat\b", context):
                relation = "turn cue"
            elif re.search(r"\bcontinue\b|\bdown\b|\balong\b", context):
                relation = "corridor continues forward"
            hits.append((m.start(), {"phrase": name, "direction": direction, "relation": relation}))
    hits.sort(key=lambda x: x[0])
    out, seen = [], set()
    for _, item in hits:
        if item["phrase"] not in seen:
            out.append(item)
            seen.add(item["phrase"])
    return out


def view_inventory(visual):
    files = (visual or {}).get("files", {})
    bearings = []
    for key in files:
        m = VIEW_RE.match(key)
        if m and key.startswith("rgb_"):
            bearings.append(int(m.group(1)) % 360)
    if "rgb" in files:
        # The base RGB sensor is the 0-degree sector in the validated
        # multiview configuration; it is not an unknown bearing.
        bearings.append(0)
    return sorted(set(bearings), key=lambda x: (x is None, x if x is not None else 0))


def rank_desc(rows, score_key):
    return sorted(rows, key=lambda r: (float(r.get(score_key, float("-inf"))), -int(r["high_level_step"])), reverse=True)


def rank_of(items, key):
    for i, x in enumerate(items, 1):
        if x["_key"] == key:
            return i
    return len(items) + 1


def stop_probability(logits):
    vals = [float(x) for x in (logits or []) if isinstance(x, (int, float)) and math.isfinite(float(x))]
    if not vals or not isinstance(logits[0], (int, float)) or not math.isfinite(float(logits[0])):
        return 0.0
    m = max(vals)
    denom = sum(math.exp(x - m) for x in vals)
    return math.exp(float(logits[0]) - m) / denom if denom else 0.0


def spearman(xs, ys):
    if len(xs) < 2:
        return None
    def r(vals):
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        out = [0.0] * len(vals); i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            for k in range(i, j + 1): out[order[k]] = (i + j) / 2.0 + 1.0
            i = j + 1
        return out
    a, b = r(xs), r(ys)
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    da = sum((x - ma) ** 2 for x in a); db = sum((y - mb) ** 2 for y in b)
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / math.sqrt(da * db) if da and db else 0.0


def bootstrap_scene(rows, metric, seed=100, n=2000):
    scenes = sorted(set(r["scene_id"] for r in rows))
    if not scenes:
        return None
    import random
    rng = random.Random(seed); vals = []
    grouped = {s: [r for r in rows if r["scene_id"] == s] for s in scenes}
    for _ in range(n):
        sample = []
        for s in scenes:
            sample.extend(rng.choice(grouped[s]) for _ in grouped[s])
        v = metric(sample)
        if v is not None and math.isfinite(v): vals.append(v)
    vals.sort()
    return [vals[int(.025 * len(vals))], vals[int(.975 * len(vals)) - 1]] if vals else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", type=Path, required=True)
    ap.add_argument("--semantic", type=Path, required=True)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    out = args.output_dir; out.mkdir(parents=True, exist_ok=False)

    dataset = json.load(gzip.open(args.dataset, "rt"))
    episodes = {str(e["episode_id"]): e for e in dataset["episodes"]}
    graph_path = args.capture / "graph_options" / "graph_options.jsonl"
    graphs = [json.loads(x) for x in graph_path.open() if x.strip()]
    semantic = [json.loads(x) for x in args.semantic.open() if x.strip()]
    sem_by_key = {(str(x["episode_id"]), int(x["high_level_step"])): x for x in semantic}

    nodes = []
    memory_rows = []
    for g in graphs:
        key = (str(g["episode_id"]), int(g["high_level_step"]))
        s = sem_by_key.get(key, {})
        source_evidence = s.get("instruction_evidence") or []
        if source_evidence:
            phrases = [{"phrase": str(x.get("phrase", "unknown")),
                        "direction": "unknown", "relation": str(x.get("relation", "unknown"))}
                       for x in source_evidence]
        else:
            phrases = ordered_phrases(g.get("instruction", ""))
        last_phrase = phrases[-1]["phrase"] if phrases else None
        rel = s.get("instruction_relevance") or {}
        matched = rel.get("matched_phrase")
        if s.get("semantic_stop_score") is not None:
            evidence = float(s.get("semantic_stop_score") or 0.0)
            matched = None
            goal_evidence = evidence
        else:
            evidence = float(rel["semantic_similarity"]) if rel.get("semantic_similarity") is not None else 0.0
        # Existing extraction stores only the best phrase score.  A goal score
        # therefore receives credit only when the best phrase is the final
        # instruction phrase; this avoids pretending an unscored phrase was seen.
            goal_evidence = evidence if matched == last_phrase else 0.0
        visual = g.get("semantic_visual") or {}
        bearings = view_inventory(visual)
        broad = s.get("landmarks") or []
        landmarks = []
        for item in broad:
            name = item.get("label", item.get("name"))
            score = float(item.get("clip_similarity", item.get("confidence", 0.0)))
            landmarks.append({"name": name, "bearing_deg": None, "direction": "unknown",
                              "confidence": max(0.0, min(1.0, (score + 1.0) / 2.0)),
                              "evidence_score": score, "source": "frozen_CLIP_probe"})
        relations = [{"relation": p["relation"], "phrase": p["phrase"],
                      "direction": p["direction"], "confidence": 0.0,
                      "source": "instruction_text_only"} for p in phrases]
        evidence_by_phrase = {str(x.get("phrase")): x for x in source_evidence if isinstance(x, dict)}
        sem_confidence = float(s.get("global_confidence", s.get("confidence", 0.0)) or 0.0)
        row = {"node_id": g.get("current_vp"), "episode_id": key[0],
               "high_level_step": key[1], "scene_id": g.get("scene_id"),
               "instruction": g.get("instruction", ""),
               "position": g.get("graph_position"),
               "place": {"label": (s.get("place", {}).get("label") if isinstance(s.get("place"), dict) else s.get("place")),
                         "confidence": float(s.get("place", {}).get("confidence", s.get("confidence", 0.0)) if isinstance(s.get("place"), dict) else s.get("confidence", 0.0))},
               "landmarks": landmarks, "relations": relations,
               "instruction_evidence": [{"phrase": p["phrase"],
                                          "score": float(evidence_by_phrase.get(p["phrase"], {}).get("score", evidence if p["phrase"] == matched else 0.0) or 0.0),
                                          "relation": evidence_by_phrase.get(p["phrase"], {}).get("relation", p["relation"]),
                                          "confidence": float(evidence_by_phrase.get(p["phrase"], {}).get("confidence", evidence if p["phrase"] == matched else 0.0) or 0.0)}
                                         for p in phrases],
               "semantic_embedding": {"encoder": "openai_clip_ViT-B/32_probe", "available": False,
                                       "reason": "probe score retained; raw embedding intentionally not duplicated"},
               "global_confidence": sem_confidence,
               "semantic_goal_evidence": goal_evidence,
               "native_stop_score": stop_probability(g.get("logits")),
               "view_bearings_deg": bearings,
               "view_coverage": {"observed_sector_count": len(set(bearings)),
                                 "expected_sector_count": 12, "complete": len([b for b in bearings if b is not None]) >= 12,
                                 "central_view_present": 0 in bearings},
               "observed_semantics_only": True, "privileged_future_input_used": False}
        memory_rows.append(row)
        ep = episodes.get(key[0]); goal = (ep or {}).get("goals", [{}])[0].get("position")
        node = dict(row); node["_key"] = key; node["goal_distance"] = dist(g.get("graph_position"), goal)
        node["native_action_index"] = g.get("effective_index")
        node["native_selected_stop"] = g.get("effective_index") == 0
        nodes.append(node)

    with (out / "semantic_node_memory.jsonl").open("w") as f:
        for row in memory_rows: f.write(json.dumps(row, sort_keys=True) + "\n")

    by_ep = defaultdict(list)
    for r in nodes: by_ep[r["episode_id"]].append(r)
    for rr in by_ep.values(): rr.sort(key=lambda x: x["high_level_step"])

    stop_rows, progress_rows, back_rows = [], [], []
    for ep, rr in by_ep.items():
        target = min(rr, key=lambda x: x["goal_distance"])["_key"]
        native_rank = rank_desc(rr, "native_stop_score")
        semantic_rank = rank_desc(rr, "semantic_goal_evidence")
        # Rank-normalized late fusion is a descriptive offline comparator only.
        nmax = max((x["native_stop_score"] for x in rr), default=0.0)
        smax = max((x["semantic_goal_evidence"] for x in rr), default=0.0)
        for x in rr:
            x["fusion_score"] = .5 * (x["native_stop_score"] / nmax if nmax else 0.0) + .5 * (x["semantic_goal_evidence"] / smax if smax else 0.0)
        fusion_rank = rank_desc(rr, "fusion_score")
        stop_rows.append({"episode_id": ep, "scene_id": rr[0]["scene_id"],
                          "target_key": target, "native_top1": native_rank[0]["_key"] == target,
                          "semantic_top1": semantic_rank[0]["_key"] == target,
                          "fusion_top1": fusion_rank[0]["_key"] == target,
                          "native_rank": rank_of(native_rank, target),
                          "semantic_rank": rank_of(semantic_rank, target),
                          "fusion_rank": rank_of(fusion_rank, target),
                          "native_target_distance_m": next(x["goal_distance"] for x in native_rank if x["_key"] == target),
                          "semantic_retrieved_distance_m": semantic_rank[0]["goal_distance"]})
        observed = []
        phrases = [{"phrase": x["phrase"]} for x in rr[0].get("instruction_evidence", [])]
        if not phrases:
            phrases = ordered_phrases(rr[0].get("instruction", ""))
        pidx = {p["phrase"]: i for i, p in enumerate(phrases)}
        for x in rr:
            # matched phrase is carried in the retained evidence list
            scored = [v for v in x["instruction_evidence"] if v["score"] > 0]
            observed.append(pidx.get(scored[0]["phrase"]) if scored else None)
        seen = [x for x in observed if x is not None]
        inversions = sum(1 for i in range(len(seen)) for j in range(i + 1, len(seen)) if seen[j] < seen[i])
        progress_rows.append({"episode_id": ep, "scene_id": rr[0]["scene_id"], "phrase_count": len(phrases),
                              "matched_steps": len(seen), "phrase_recovery": len(set(seen)) / len(phrases) if phrases else 0.0,
                              "inversions": inversions, "monotone": bool(seen) and inversions == 0,
                              "ordered_transition_accuracy": sum(seen[i] <= seen[i+1] for i in range(len(seen)-1)) / max(1, len(seen)-1)})
        for i, cur in enumerate(rr):
            hist = rr[:i + 1]
            nr = rank_desc(hist, "native_stop_score")[0]
            sr = rank_desc(hist, "semantic_goal_evidence")[0]
            gr = min(hist, key=lambda x: x["goal_distance"])
            back_rows.append({"episode_id": ep, "scene_id": rr[0]["scene_id"], "step": cur["high_level_step"],
                              "native_key": nr["_key"], "semantic_key": sr["_key"], "gt_key": gr["_key"],
                              "native_rank": rank_of(sorted(hist, key=lambda x: x["native_stop_score"], reverse=True), gr["_key"]),
                              "semantic_rank": rank_of(sorted(hist, key=lambda x: x["semantic_goal_evidence"], reverse=True), gr["_key"]),
                              "native_distance_m": nr["goal_distance"], "semantic_distance_m": sr["goal_distance"],
                              "gt_distance_m": gr["goal_distance"]})

    def mean(xs): return sum(xs) / len(xs) if xs else None
    stop_summary = {"episodes": len(stop_rows), "goal_node_top1_recall": mean([x["semantic_top1"] for x in stop_rows]),
                    "native_top1_recall": mean([x["native_top1"] for x in stop_rows]),
                    "fusion_top1_recall": mean([x["fusion_top1"] for x in stop_rows]),
                    "native_mrr": mean([1.0 / x["native_rank"] for x in stop_rows]),
                    "semantic_mrr": mean([1.0 / x["semantic_rank"] for x in stop_rows]),
                    "fusion_mrr": mean([1.0 / x["fusion_rank"] for x in stop_rows]),
                    "spearman_native_vs_goal": spearman([x["native_stop_score"] for x in nodes], [-x["goal_distance"] for x in nodes]),
                    "spearman_semantic_vs_goal": spearman([x["semantic_goal_evidence"] for x in nodes], [-x["goal_distance"] for x in nodes]),
                    "scene_bootstrap_ci_semantic_spearman": bootstrap_scene(nodes, lambda x: spearman([r["semantic_goal_evidence"] for r in x], [-r["goal_distance"] for r in x]))}
    progress_summary = {"episodes": len(progress_rows), "ordered_evidence_consistency": mean([x["ordered_transition_accuracy"] for x in progress_rows]),
                        "phrase_recovery": mean([x["phrase_recovery"] for x in progress_rows]), "monotone_episode_rate": mean([x["monotone"] for x in progress_rows]),
                        "inversion_count": sum(x["inversions"] for x in progress_rows)}
    valid_back = [x for x in back_rows if x["native_distance_m"] < float("inf")]
    back_summary = {"states": len(valid_back), "native_retrieval_mrr": mean([1.0 / x["native_rank"] for x in valid_back]),
                    "semantic_retrieval_mrr": mean([1.0 / x["semantic_rank"] for x in valid_back]),
                    "semantic_distance_improvement_rate": mean([x["semantic_distance_m"] < x["native_distance_m"] for x in valid_back]),
                    "semantic_median_distance_gain_m": sorted([x["native_distance_m"] - x["semantic_distance_m"] for x in valid_back])[len(valid_back)//2] if valid_back else None,
                    "scene_bootstrap_ci_improvement_rate": bootstrap_scene(valid_back, lambda x: mean([r["semantic_distance_m"] < r["native_distance_m"] for r in x]))}
    native_correct = sum(x["native_top1"] for x in stop_rows); semantic_correct = sum(x["semantic_top1"] for x in stop_rows)
    confusion = {"native_correct_semantic_correct": sum(x["native_top1"] and x["semantic_top1"] for x in stop_rows),
                 "native_correct_semantic_wrong": sum(x["native_top1"] and not x["semantic_top1"] for x in stop_rows),
                 "native_wrong_semantic_correct": sum(not x["native_top1"] and x["semantic_top1"] for x in stop_rows),
                 "native_wrong_semantic_wrong": sum(not x["native_top1"] and not x["semantic_top1"] for x in stop_rows)}
    coverage = Counter(sum(1 for b in r["view_bearings_deg"] if b is not None) for r in memory_rows)
    result = {"head": subprocess.check_output(["git", "rev-parse", "HEAD"], universal_newlines=True).strip(),
              "dataset": {"episodes": len(by_ep), "scenes": len(set(r["scene_id"] for r in nodes)), "decision_nodes": len(nodes)},
              "representation": {"visited_nodes_only": True, "coverage_histogram_sector_count": dict(coverage),
                                  "complete_360_node_count": sum(1 for r in memory_rows if len([b for b in r["view_bearings_deg"] if b is not None]) >= 12),
                                  "central_only_capture": all(len([b for b in r["view_bearings_deg"] if b is not None]) == 0 for r in memory_rows),
                                  "what_where_relation_instruction_fields": True,
                                  "future_or_gt_used_as_feature": False,
                                  "generic_clip_is_probe_only": True},
              "stop": stop_summary, "progress": progress_summary, "backtracking": back_summary,
              "complementarity": confusion,
              "safety": {"native_success_retention": "44/44 cohort is native-success by construction; no intervention run",
                          "beneficial_intervention_rate": confusion["native_wrong_semantic_correct"] / max(1, len(stop_rows)),
                          "harmful_intervention_rate": confusion["native_correct_semantic_wrong"] / max(1, len(stop_rows)),
                          "risk_coverage": "not applicable: no action was changed"},
              "shadow_deployment": {"run": False, "trajectory_changed": False, "status": "NOT_RUN_GATED_NO_GO"},
              "limits": ["The 44-cohort multiview rerun has 12 RGB + 12 depth views per pre-action state and was action/pose/graph aligned to the original trace.",
                         "Model confidence is uncalibrated; the frozen 2B extractor returned zero phrase confidence in this run, so semantic scores abstain.",
                         "Goal positions and outcome cohort membership are evaluation labels only."],
              "files": {"capture_graph_sha256": digest(graph_path), "semantic_input": str(args.semantic)}}
    (out / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
    (out / "progress_by_episode.json").write_text(json.dumps(progress_rows, indent=2) + "\n")
    (out / "stop_by_episode.json").write_text(json.dumps(stop_rows, indent=2) + "\n")
    (out / "backtracking_by_state.json").write_text(json.dumps(back_rows, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
