#!/usr/bin/env python3
"""Development-only probe for a pre-action failure recovery gate."""
import argparse
import glob
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


def arr(v):
    if isinstance(v, dict) and "__array__" in v:
        return v["__array__"]
    return v


def compare(metrics, baseline):
    for key, sign, eps in [("success", 1, 1e-6), ("ndtw", 1, 1e-6), ("sdtw", 1, 1e-6),
                           ("path_length", -1, 1e-5), ("steps_taken", -1, 1e-5),
                           ("collisions", -1, 1e-6)]:
        d = float(metrics.get(key, 0.0)) - float(baseline.get(key, 0.0))
        if abs(d) > eps:
            return 1 if sign * d > 0 else -1
    return 0


def load_rows(cases):
    rows = []
    for result_path in glob.glob(str(cases / "e*_s*_a*" / "result.json")):
        result = json.loads(Path(result_path).read_text())
        c = result["case"]
        if c.get("mode") != "action":
            continue
        graph_path = Path(result_path).parent / "graph" / "graph_options.jsonl"
        graphs = [json.loads(x) for x in graph_path.open()]
        graph = next(g for g in graphs if int(g["high_level_step"]) == int(c["high_level_step"]))
        trace_path = Path(result_path).parent / "trace.jsonl"
        trace_rows = [json.loads(x) for x in trace_path.open()]
        trace = next(t for t in trace_rows if int(t["high_level_step"]) == int(c["high_level_step"]))
        option = next(o for o in graph["options"] if int(o["index"]) == int(c["action_index"]))
        label = compare(result["metrics"], result["baseline_metrics"])
        if label == 0:
            continue  # native/equal branch, no positive-vs-negative target
        ids = graph.get("graph_ids") or []
        idx = int(option["index"])
        graph_id = option.get("graph_id")
        position = np.asarray(arr(graph["graph_position"]), dtype=float)
        action = option["action"]
        target = arr(action.get("ghost_pos")) if action.get("act") != 0 else arr(action.get("stop_pos"))
        target = np.asarray(target if target is not None else position, dtype=float)
        delta = target - position
        distance = float(np.linalg.norm(delta[[0, 2]]))
        native_index = int(graph["effective_index"])
        native_logit = float(graph["logits"][idx]) if graph["logits"][idx] is not None else 0.0
        selected_logit = float(graph["logits"][native_index]) if graph["logits"][native_index] is not None else 0.0
        current = str(graph.get("current_vp"))
        is_ghost = graph_id is not None and str(graph_id).startswith("g")
        is_visited = graph_id is not None and not is_ghost
        current_num = int(current) if current.isdigit() else -1
        cand_num = int(str(graph_id)) if is_visited and str(graph_id).isdigit() else -1
        features = [
            native_logit, selected_logit, native_logit - selected_logit,
            float(idx == 0), float(option.get("current_proposal", False)),
            float(len(ids)), float(len(action.get("back_path") or [])),
            float(idx), float(idx / max(len(ids) - 1, 1)), distance,
            float(is_ghost), float(is_visited), float(cand_num < current_num and cand_num >= 0),
            float(graph_id is None), float(graph.get("effective_index") == 0),
            float(c["high_level_step"]),
            float((trace.get("pre_metrics") or {}).get("steps_taken", 0.0)),
            float(((trace.get("pre_metrics") or {}).get("collisions") or {}).get("count", 0.0)
                  if isinstance((trace.get("pre_metrics") or {}).get("collisions"), dict) else 0.0),
        ]
        rows.append({"scene_id": graph["scene_id"], "episode_id": str(c["episode_id"]),
                     "high_level_step": int(c["high_level_step"]),
                     "state_key": "{}|{}|{}".format(graph["scene_id"], c["episode_id"], c["high_level_step"]),
                     "action_index": idx, "label": "BENEFIT" if label > 0 else "HARM",
                     "features": features, "native_index": native_index,
                     "native_stop": bool(graph.get("effective_index") == 0),
                     "native_success": bool(result["baseline_metrics"].get("success", 0) > 0.5)})
    return rows


def fit_predict(x, y, w, xt):
    mu = x.mean(0); sd = x.std(0); sd[sd < 1e-8] = 1.
    x = (x - mu) / sd; xt = (xt - mu) / sd
    keep = np.std(x, axis=0) > 1e-8; x = x[:, keep]; xt = xt[:, keep]
    w = w / max(w.mean(), 1e-8); init = np.zeros(x.shape[1] + 1)
    prior = float(np.average(y, weights=w)); init[-1] = math.log(max(prior, 1e-6) / max(1 - prior, 1e-6))
    def fun(v):
        z = x.dot(v[:-1]) + v[-1]; e = expit(z) - y
        loss = np.average(np.logaddexp(0, z) - y * z, weights=w) + .01 * np.dot(v[:-1], v[:-1])
        grad = np.r_[((w * e)[:, None] * x).sum(0) / w.sum() + .02 * v[:-1], (w * e).sum() / w.sum()]
        return loss, grad
    fit = minimize(fun, init, jac=True, method="L-BFGS-B", options={"maxiter": 500, "ftol": 1e-10})
    if not fit.success: raise RuntimeError(fit.message)
    return expit(xt.dot(fit.x[:-1]) + fit.x[-1])


def auc_ap(y, p):
    y = np.asarray(y, bool); p = np.asarray(p, float); pos = int(y.sum()); neg = len(y) - pos
    if not pos or not neg: return None, None
    order = np.argsort(-p, kind="mergesort"); yy = y[order]; pp = p[order]
    ends = np.r_[np.flatnonzero(np.diff(pp) != 0), len(p) - 1]
    tp = np.cumsum(yy)[ends]; fp = ends + 1 - tp
    auc = float(np.trapz(np.r_[0., tp] / pos, np.r_[0., fp] / neg))
    ap = float(np.sum(np.diff(np.r_[0., tp]) / pos * tp / (ends + 1)))
    return auc, ap


def pairwise(rows, y, p):
    groups = defaultdict(list)
    for i, r in enumerate(rows): groups[r["state_key"]].append(i)
    win = total = 0.0
    for ix in groups.values():
        pos = [i for i in ix if y[i]]; neg = [i for i in ix if not y[i]]
        for i in pos:
            for j in neg:
                win += (p[i] > p[j]) + .5 * (p[i] == p[j]); total += 1
    return win / total if total else None


def state_rescue(rows, y, p):
    groups = defaultdict(list)
    for i, r in enumerate(rows): groups[r["state_key"]].append(i)
    opportunity = rescued = harmful = 0
    for ix in groups.values():
        has_benefit = any(y[i] for i in ix); opportunity += has_benefit
        top = ix[int(np.argmax(p[ix]))]
        rescued += bool(has_benefit and y[top]); harmful += bool(not has_benefit and not y[top])
    return {"states": len(groups), "opportunity_states": opportunity,
            "top1_rescue": rescued, "top1_harm_on_no_opportunity": harmful}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cases", type=Path, required=True); ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args(); rows = load_rows(args.cases)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "rows.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    y = np.asarray([r["label"] == "BENEFIT" for r in rows], dtype=bool)
    scenes = np.asarray([r["scene_id"] for r in rows]); x = np.asarray([r["features"] for r in rows], dtype=float)
    # Native-only keeps fields that a released policy already computes; the
    # full block adds candidate graph geometry/topology controls.
    blocks = {"native_margin_stop": x[:, :5], "native_plus_graph_geometry": x}
    pred = {n: np.zeros(len(rows)) for n in blocks}
    for name, xx in blocks.items():
        for scene in sorted(set(scenes)):
            test = scenes == scene; train = ~test
            groups = defaultdict(list)
            for i, r in enumerate(rows):
                if train[i]: groups[r["state_key"]].append(i)
            w = np.zeros(len(rows))
            for ix in groups.values(): w[ix] = 1.0 / len(ix)
            for cls in (0, 1):
                total = w[train & (y == cls)].sum()
                if total: w[train & (y == cls)] *= .5 / total
            pred[name][test] = fit_predict(xx[train], y[train], w[train], xx[test])
    result = {"cohort_status": "outcome-enriched development diagnostic; not an unbiased benchmark",
              "rows": len(rows), "states": len(set(r["state_key"] for r in rows)), "scenes": len(set(scenes)),
              "labels": dict(Counter("BENEFIT" if z else "HARM" for z in y)), "models": {}, "scene_split": "leave-one-scene-out"}
    for name, p in pred.items():
        auc, ap = auc_ap(y, p); result["models"][name] = {"auroc": auc, "auprc": ap,
            "within_state_pairwise": pairwise(rows, y, p), "state_rescue": state_rescue(rows, y, p)}
    with (args.output / "oof.jsonl").open("w") as f:
        for i, r in enumerate(rows): f.write(json.dumps({"state_key": r["state_key"], "pair": r["action_index"], "label": r["label"], "predictions": {n: float(p[i]) for n, p in pred.items()}}) + "\n")
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
