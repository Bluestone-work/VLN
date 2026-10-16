#!/usr/bin/env python3
"""Scene-disjoint probe for explicit graph/history waypoint value."""
import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


def auc_ap(y, p):
    y = np.asarray(y, bool); p = np.asarray(p, float)
    pos = int(y.sum()); neg = len(y) - pos
    if not pos or not neg: return None, None
    order = np.argsort(-p, kind="mergesort"); yy = y[order]
    ends = np.r_[np.flatnonzero(np.diff(p[order]) != 0), len(p) - 1]
    tp = np.cumsum(yy)[ends]; fp = ends + 1 - tp
    auc = float(np.trapz(np.r_[0., tp] / pos, np.r_[0., fp] / neg))
    ap = float(np.sum(np.diff(np.r_[0., tp]) / pos * tp / (ends + 1)))
    return auc, ap


def fit_predict(x, y, weights, xt):
    mu = x.mean(0); sd = x.std(0); sd[sd < 1e-8] = 1.
    x = (x - mu) / sd; xt = (xt - mu) / sd
    keep = np.std(x, axis=0) > 1e-8; x = x[:, keep]; xt = xt[:, keep]
    weights = weights / max(weights.mean(), 1e-8)
    init = np.zeros(x.shape[1] + 1)
    prior = float(np.average(y, weights=weights))
    init[-1] = math.log(max(prior, 1e-6) / max(1 - prior, 1e-6))
    def fun(w):
        z = x.dot(w[:-1]) + w[-1]; e = expit(z) - y
        loss = np.average(np.logaddexp(0, z) - y * z, weights=weights) + .01 * np.dot(w[:-1], w[:-1])
        grad = np.r_[((weights * e)[:, None] * x).sum(0) / weights.sum() + .02 * w[:-1],
                     (weights * e).sum() / weights.sum()]
        return loss, grad
    result = minimize(fun, init, jac=True, method="L-BFGS-B", options={"maxiter": 1000, "ftol": 1e-10})
    if not result.success: raise RuntimeError(result.message)
    return expit(xt.dot(result.x[:-1]) + result.x[-1])


def weights(rows, y, train):
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        if train[i]: groups[r["state_key"]].append(i)
    w = np.zeros(len(rows), float)
    for ix in groups.values():
        for i in ix: w[i] = 1. / len(ix)
    for cls in (0, 1):
        total = w[train & (y == cls)].sum()
        if total: w[train & (y == cls)] *= .5 / total
    return w


def pairwise(rows, y, p):
    groups = defaultdict(list)
    for i, r in enumerate(rows): groups[r["state_key"]].append(i)
    num = den = 0.0
    for ix in groups.values():
        pos = [i for i in ix if y[i]]; neg = [i for i in ix if not y[i]]
        for i in pos:
            for j in neg:
                num += (p[i] > p[j]) + .5 * (p[i] == p[j]); den += 1
    return float(num / den) if den else None


def metrics(rows, y, p):
    auc, ap = auc_ap(y, p)
    pred = p >= .5
    return {"auroc": auc, "auprc": ap, "within_state_pairwise": pairwise(rows, y, p),
            "benefit_precision": float(np.sum(pred & y) / max(np.sum(pred), 1)),
            "benefit_recall": float(np.sum(pred & y) / max(np.sum(y), 1)),
            "predicted_benefit": int(pred.sum()), "brier": float(np.mean((p - y) ** 2))}


def rescue(rows, y, p):
    groups = defaultdict(list)
    for i, r in enumerate(rows): groups[r["state_key"]].append(i)
    out = {"states": len(groups), "rescue_opportunity_states": 0,
           "native_wrong_model_correct": 0, "native_correct_model_wrong": 0}
    for ix in groups.values():
        has_benefit = bool(np.any(y[ix])); out["rescue_opportunity_states"] += has_benefit
        top = ix[int(np.argmax(p[ix]))]
        if has_benefit and y[top]: out["native_wrong_model_correct"] += 1
        if not has_benefit and not y[top]: out["native_correct_model_wrong"] += 1
    out["rescue_rate"] = out["native_wrong_model_correct"] / max(out["rescue_opportunity_states"], 1)
    out["destruction_rate"] = out["native_correct_model_wrong"] / max(len(groups) - out["rescue_opportunity_states"], 1)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", type=Path, required=True); ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args(); d = np.load(str(args.dataset / "features.npz"))
    rows = [json.loads(x) for x in (args.dataset / "rows.jsonl").open()]
    y = d["labels"].astype(bool); scenes = np.asarray([r["scene_id"] for r in rows])
    blocks = {"F0_native": d["native"], "F0_plus_geometry": np.c_[d["native"], d["geometry"]],
              "graph_history_only": d["graph"], "F0_plus_graph_history": np.c_[d["native"], d["graph"]],
              "F0_plus_geometry_plus_graph_history": np.c_[d["native"], d["geometry"], d["graph"]]}
    preds = {n: np.zeros(len(rows), float) for n in blocks}
    for name, x in blocks.items():
        for scene in sorted(set(scenes)):
            test = scenes == scene; train = ~test
            w = weights(rows, y, train)
            preds[name][test] = fit_predict(x[train], y[train], w[train], x[test])
        print("fitted", name, flush=True)
    out = {"dataset": str(args.dataset), "rows": len(rows), "states": len(set(r["state_key"] for r in rows)),
           "scenes": len(set(scenes)), "labels": dict(Counter("BENEFIT" if z else "HARM" for z in y)),
           "split": "leave-one-scene-out", "models": {}, "paired_deltas": {}}
    for name, p in preds.items(): out["models"][name] = dict(metrics(rows, y, p), rescue=rescue(rows, y, p))
    rng = np.random.RandomState(20261016); unique = sorted(set(scenes)); boot = {n: defaultdict(list) for n in blocks}
    for _ in range(2000):
        sample = rng.choice(unique, len(unique), replace=True)
        for name, p in preds.items():
            ix = np.concatenate([np.where(scenes == s)[0] for s in sample])
            a, ap = auc_ap(y[ix], p[ix]); boot[name]["auroc"].append(a); boot[name]["auprc"].append(ap)
            boot[name]["pairwise"].append(pairwise([rows[i] for i in ix], y[ix], p[ix]))
    for name in blocks:
        out["models"][name]["scene_bootstrap_ci"] = {k: np.percentile(v, [2.5, 97.5]).tolist() for k, v in boot[name].items()}
    for other, base in [("F0_plus_geometry", "F0_native"), ("graph_history_only", "F0_native"),
                        ("F0_plus_graph_history", "F0_native"), ("F0_plus_geometry_plus_graph_history", "F0_plus_geometry")]:
        out["paired_deltas"][other + " minus " + base] = {}
        for metric in ("auroc", "auprc", "pairwise"):
            out["paired_deltas"][other + " minus " + base][metric] = np.percentile(
                np.asarray(boot[other][metric]) - np.asarray(boot[base][metric]), [2.5, 97.5]).tolist()
    with (args.output.parent / "graph_history_value_oof.jsonl").open("w") as stream:
        for i, r in enumerate(rows):
            stream.write(json.dumps({"pair_id": r["pair_id"], "state_key": r["state_key"], "scene_id": r["scene_id"],
                                     "label": r["label"], "predictions": {n: float(p[i]) for n, p in preds.items()}}) + "\n")
    args.output.write_text(json.dumps(out, indent=2) + "\n"); print(json.dumps(out, indent=2))


if __name__ == "__main__": main()
