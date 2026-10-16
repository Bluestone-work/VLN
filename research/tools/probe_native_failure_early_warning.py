#!/usr/bin/env python3
"""Development probe for early native-trajectory failure warning.

Only pre-action graph/controller state is used as a feature. Final native
success is an evaluation label. The available cohorts are outcome-enriched and
must not be treated as benchmark data.
"""
import argparse
import glob
import json
import math
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


def arr(v):
    return v.get("__array__") if isinstance(v, dict) and "__array__" in v else v


def logits_summary(logits, selected):
    values = sorted(float(x) for x in logits if x is not None)
    chosen = float(logits[selected]) if selected < len(logits) and logits[selected] is not None else 0.0
    others = [x for x in values if abs(x - chosen) > 1e-12]
    second = max(others) if others else chosen
    margin = chosen - second
    z = np.asarray(values, dtype=float); z -= np.max(z); p = np.exp(z); p /= p.sum()
    entropy = float(-(p * np.log(np.maximum(p, 1e-12))).sum())
    return chosen, margin, entropy, len(values)


def state_feature(graph, trace=None):
    ids = graph.get("graph_ids") or []
    selected = int(graph["effective_index"])
    chosen, margin, entropy, option_count = logits_summary(graph.get("logits") or [], selected)
    visited = sum(x is not None and not str(x).startswith("g") for x in ids)
    ghosts = sum(x is not None and str(x).startswith("g") for x in ids)
    options = graph.get("options") or []
    selected_option = next((o for o in options if int(o["index"]) == selected), None)
    back_len = len((selected_option or {}).get("action", {}).get("back_path") or [])
    pre = (trace or {}).get("pre_metrics") or {}
    collisions = pre.get("collisions")
    collision_count = float(collisions.get("count", 0.0)) if isinstance(collisions, dict) else 0.0
    return [float(graph.get("high_level_step", 0)), float(len(ids)), float(visited), float(ghosts),
            float(option_count), chosen, margin, entropy, float(selected == 0), float(back_len),
            float((pre or {}).get("steps_taken", 0.0)), collision_count]


def load_rows(success_pairs, success_graph, success_trace, failure_cases):
    states = {}
    episode_labels = {}
    success_keys = set()
    for line in success_pairs.open():
        row = json.loads(line)
        key = (row["scene_id"], str(row["episode_id"]), int(row["high_level_step"]))
        success_keys.add(key)
        episode_labels[(key[0], key[1])] = 1
    success_graphs = {}
    for line in success_graph.open():
        g = json.loads(line); key = (g["scene_id"], str(g["episode_id"]), int(g["high_level_step"]))
        if key in success_keys: success_graphs[key] = g
    success_traces = {}
    for line in success_trace.open():
        t = json.loads(line); key = (t["scene_id"], str(t["episode_id"]), int(t["high_level_step"]))
        if key in success_keys: success_traces[key] = t
    for key in sorted(success_keys):
        if key not in success_graphs or key not in success_traces: continue
        states[key] = {"scene_id": key[0], "episode_id": key[1], "step": key[2],
                       "features": state_feature(success_graphs[key], success_traces[key]),
                       "label": 1, "source": "native_success_cohort"}
    # Failure cohort: graph/options and pre-action trace are enough to rebuild
    # the deployed state features; critical branch outcomes are not read.
    for result_path in glob.glob(str(failure_cases / "e*_s*_a*" / "result.json")):
        result = json.loads(Path(result_path).read_text()); c = result["case"]
        if c.get("mode") != "action": continue
        graph_path = Path(result_path).parent / "graph" / "graph_options.jsonl"
        trace_path = Path(result_path).parent / "trace.jsonl"
        graph = next(g for g in (json.loads(x) for x in graph_path.open())
                     if int(g["high_level_step"]) == int(c["high_level_step"]))
        trace = next(t for t in (json.loads(x) for x in trace_path.open())
                     if int(t["high_level_step"]) == int(c["high_level_step"]))
        key = (graph["scene_id"], str(c["episode_id"]), int(c["high_level_step"]))
        states[key] = {"scene_id": key[0], "episode_id": key[1], "step": key[2],
                       "features": state_feature(graph, trace), "label": 0,
                       "source": "native_failure_cohort"}
        episode_labels[(key[0], key[1])] = 0
    return list(states.values()), episode_labels


def fit_predict(x, y, weights, xt):
    mu = x.mean(0); sd = x.std(0); sd[sd < 1e-8] = 1.0
    x = (x - mu) / sd; xt = (xt - mu) / sd; keep = np.std(x, axis=0) > 1e-8
    x = x[:, keep]; xt = xt[:, keep]; weights = weights / max(weights.mean(), 1e-8)
    prior = float(np.average(y, weights=weights)); init = np.zeros(x.shape[1] + 1)
    init[-1] = math.log(max(prior, 1e-6) / max(1 - prior, 1e-6))
    def fun(v):
        z = x.dot(v[:-1]) + v[-1]; e = expit(z) - y
        loss = np.average(np.logaddexp(0, z) - y * z, weights=weights) + .01 * np.dot(v[:-1], v[:-1])
        grad = np.r_[((weights * e)[:, None] * x).sum(0) / weights.sum() + .02 * v[:-1], (weights * e).sum() / weights.sum()]
        return loss, grad
    fit = minimize(fun, init, jac=True, method="L-BFGS-B", options={"maxiter": 500, "ftol": 1e-10})
    if not fit.success: raise RuntimeError(fit.message)
    return expit(xt.dot(fit.x[:-1]) + fit.x[-1])


def auc_ap(y, p):
    y = np.asarray(y, bool); p = np.asarray(p, float); pos = int(y.sum()); neg = len(y) - pos
    if not pos or not neg: return None, None
    order = np.argsort(-p, kind="mergesort"); yy = y[order]; pp = p[order]
    ends = np.r_[np.flatnonzero(np.diff(pp) != 0), len(p) - 1]; tp = np.cumsum(yy)[ends]; fp = ends + 1 - tp
    auc = float(np.trapz(np.r_[0., tp] / pos, np.r_[0., fp] / neg)); ap = float(np.sum(np.diff(np.r_[0., tp]) / pos * tp / (ends + 1)))
    return auc, ap


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--success-pairs", type=Path, required=True); ap.add_argument("--success-graph", type=Path, required=True); ap.add_argument("--success-trace", type=Path, required=True); ap.add_argument("--failure-cases", type=Path, required=True); ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args(); rows, episodes = load_rows(args.success_pairs, args.success_graph, args.success_trace, args.failure_cases)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "rows.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    scenes = np.asarray([r["scene_id"] for r in rows]); y = np.asarray([r["label"] for r in rows], dtype=bool); x = np.asarray([r["features"] for r in rows], dtype=float)
    windows = {"all_states": np.ones(len(rows), bool), "early_step_le3": np.asarray([r["step"] <= 3 for r in rows])}
    summary = {"cohort_status": "outcome-enriched development cohorts; not an unbiased benchmark", "states": len(rows), "episodes": len(episodes), "scenes": len(set(scenes)), "episode_labels": dict(Counter(episodes.values())), "models": {}}
    for window, keep in windows.items():
        pred = np.zeros(len(rows)); usable = keep
        for scene in sorted(set(scenes[usable])):
            test = (scenes == scene) & usable; train = (scenes != scene) & usable
            # Equalize each episode before class balancing.
            w = np.zeros(len(rows)); groups = defaultdict(list)
            for i, r in enumerate(rows):
                if train[i]: groups[(r["scene_id"], r["episode_id"])].append(i)
            for ix in groups.values(): w[ix] = 1.0 / len(ix)
            for cls in (0, 1):
                total = w[train & (y == cls)].sum()
                if total: w[train & (y == cls)] *= .5 / total
            if train.any() and test.any(): pred[test] = fit_predict(x[train], y[train], w[train], x[test])
        auc, apv = auc_ap(y[usable], pred[usable]); high = pred[usable] >= .5
        summary["models"][window] = {"auroc": auc, "auprc": apv, "predicted_failure_precision": float((high & y[usable]).sum() / max(high.sum(), 1)), "predicted_failure_recall": float((high & y[usable]).sum() / max(y[usable].sum(), 1)), "coverage": float(high.mean())}
        with (args.output / (window + "_oof.jsonl")).open("w") as f:
            for i, r in enumerate(rows):
                if usable[i]: f.write(json.dumps({"state_key": "{}|{}|{}".format(r["scene_id"], r["episode_id"], r["step"]), "label": int(y[i]), "prediction": float(pred[i])}) + "\n")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n"); print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
