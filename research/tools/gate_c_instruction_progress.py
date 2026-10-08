#!/usr/bin/env python3
"""Frozen Gate C representation diagnostic.

Builds nested F0/F1/F2/F3 native-relative features from the existing Gate B
samples and graph captures. Token representations are reconstructed with the
same frozen ETPNav language encoder/checkpoint; no navigation rollout is run.
"""
from __future__ import print_function

import argparse
import csv
import hashlib
import json
import math
import os
import random
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LABELS = {"KEEP": 0.0, "INTERVENE": 1.0}
THRESHOLDS = [0.50, 0.70, 0.80, 0.90, 0.95]
EPS = 1e-8


def unpack(v):
    if isinstance(v, dict) and "__array__" in v:
        return np.asarray(v["__array__"], dtype=np.float32)
    if isinstance(v, dict) and "__scalar__" in v:
        return float(v["__scalar__"])
    return v


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def read_graph(path):
    out = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            out[(str(r["episode_id"]), int(r["high_level_step"]))] = r
    return out


def valid_embedding(option):
    e = unpack(option.get("embedding"))
    return e if isinstance(e, np.ndarray) and e.shape == (768,) and np.isfinite(e).all() else None


def cosine(a, b):
    da = float(np.linalg.norm(a)); db = float(np.linalg.norm(b))
    return float(np.dot(a, b) / max(da * db, EPS))


def span_stats(vecs, mask, emb):
    """Return stable scalar alignment statistics for a token span."""
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return {"cos": 0.0, "mass": 0.0, "entropy": 0.0, "expected": 0.0, "peak": 0.0}
    sims = np.asarray([cosine(emb, vecs[i]) for i in idx], dtype=np.float64)
    probs = np.exp(sims - sims.max()); probs /= max(probs.sum(), EPS)
    pos = idx.astype(np.float64)
    return {"cos": float(sims.mean()), "mass": float(probs.sum()),
            "entropy": float(-(probs * np.log(np.maximum(probs, EPS))).sum()),
            "expected": float((probs * pos).sum()), "peak": float(pos[int(np.argmax(sims))])}


def load_tokens(paths):
    """Map episode id to padded BERT indices from the frozen source cohorts."""
    result = {}
    for path in paths:
        import gzip
        with gzip.open(path, "rt") as f:
            data = json.load(f)
        for ep in data["episodes"]:
            result[str(ep["episode_id"])] = np.asarray(ep["instruction"]["instruction_tokens"], dtype=np.int64)
    return result


def extract_txt(tokens, checkpoint, base_pretrained, batch_size=32):
    """Use ETPNav's frozen forward_txt without constructing a simulator."""
    import torch
    from types import SimpleNamespace
    from vlnce_baselines.models.etp.vlnbert_init import get_vlnbert_models

    cfg = SimpleNamespace(
        pretrained_path=str(base_pretrained), task_type="r2r", use_depth_embedding=True,
        use_sprels=True, fix_lang_embedding=False, fix_pano_embedding=False)
    model = get_vlnbert_models(config=cfg)
    # The released navigation checkpoint is a trainer wrapper.  Load only its
    # exact frozen ETP `vln_bert` state on top of the released pretrain model,
    # matching RLTrainer.load_checkpoint's key namespace.
    import torch
    wrapper = torch.load(str(checkpoint), map_location="cpu")
    state = wrapper.get("state_dict", wrapper)
    vln_state = {k[len("net.module.vln_bert."):]: v for k, v in state.items()
                 if k.startswith("net.module.vln_bert.")}
    if not vln_state:
        raise RuntimeError("navigation checkpoint has no net.module.vln_bert state")
    missing = model.load_state_dict(vln_state, strict=False)
    if missing.missing_keys or missing.unexpected_keys:
        raise RuntimeError("frozen vln_bert state mismatch: %s %s" %
                           (missing.missing_keys, missing.unexpected_keys))
    model.eval()
    ids = sorted(tokens)
    output = {}
    with torch.no_grad():
        for start in range(0, len(ids), batch_size):
            keys = ids[start:start + batch_size]
            x = torch.from_numpy(np.stack([tokens[k] for k in keys], axis=0)).long()
            mask = x.ne(0)
            y = model.forward_txt(x, mask).detach().cpu().numpy().astype(np.float32)
            for k, v in zip(keys, y):
                # Keep the exact dataset mask; padded BERT hidden states can
                # have nonzero norms and must never be treated as instruction
                # suffix tokens.
                output[k] = (v, tokens[k] != 0)
    return output


def f1_features(native, alt):
    d = alt - native
    absd = np.abs(d)
    return {
        "f1_native_norm": float(np.linalg.norm(native)),
        "f1_alt_norm": float(np.linalg.norm(alt)),
        "f1_diff_norm": float(np.linalg.norm(d)),
        "f1_native_alt_cos": cosine(native, alt),
        "f1_native_mean": float(native.mean()), "f1_alt_mean": float(alt.mean()),
        "f1_diff_mean": float(d.mean()), "f1_diff_std": float(d.std()),
        "f1_absdiff_mean": float(absd.mean()), "f1_absdiff_std": float(absd.std()),
        "f1_native_std": float(native.std()), "f1_alt_std": float(alt.std()),
    }


def progress_features(native, alt, txt, token_mask):
    # BERT index 0 is [CLS], 0 padding; preserve all non-padding positions.
    mask = np.asarray(token_mask, dtype=bool)
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        idx = np.asarray([0]); mask = np.zeros(len(txt), dtype=bool); mask[0] = True
    # Current state proxy is STOP graph embedding; it is available pre-action.
    state = native
    state_sims = np.asarray([cosine(state, txt[i]) for i in idx], dtype=np.float64)
    p = np.exp(state_sims - state_sims.max()); p /= max(p.sum(), EPS)
    pos = idx.astype(np.float64); n = max(float(idx[-1]), 1.0)
    expected = float((p * pos).sum() / n); peak = float(pos[int(np.argmax(state_sims))] / n)
    cut = int(round(float(idx[-1] + 1) * 0.5))
    prefix = idx < max(cut, 1); suffix = ~prefix
    pref_mass = float(p[prefix].sum()); suff_mass = float(p[suffix].sum())
    ent = float(-(p * np.log(np.maximum(p, EPS))).sum() / max(math.log(len(p)), 1.0))
    out = {
        "f2_progress_expected": expected, "f2_progress_peak": peak,
        "f2_progress_entropy": ent, "f2_prefix_mass": pref_mass,
        "f2_suffix_mass": suff_mass, "f2_progress_margin": suff_mass - pref_mass,
    }
    # Candidate/state alignment with the token stream is the explicit semantic
    # diagnostic. P3 spans are fixed by token position, never by outcomes.
    for name, emb in (("native", native), ("alt", alt)):
        sims = np.asarray([cosine(emb, txt[i]) for i in idx], dtype=np.float64)
        q = np.exp(sims - sims.max()); q /= max(q.sum(), EPS)
        out[f"f3_{name}_instruction_cos"] = float(sims.mean())
        out[f"f3_{name}_prefix_cos"] = float(sims[prefix].mean())
        out[f"f3_{name}_suffix_cos"] = float(sims[suffix].mean())
        out[f"f3_{name}_suffix_mass"] = float(q[suffix].sum())
        out[f"f3_{name}_token_expected"] = float((q * pos).sum() / n)
    for key in ("instruction_cos", "prefix_cos", "suffix_cos", "suffix_mass", "token_expected"):
        out[f"f3_delta_{key}"] = out[f"f3_alt_{key}"] - out[f"f3_native_{key}"]
    return out


def build_rows(samples, graphs, txt_by_ep):
    rows = []; missing = []
    for src in samples:
        key = (str(src["episode_id"]), int(src["high_level_step"]))
        graph = graphs.get(key)
        if graph is None or key[0] not in txt_by_ep:
            missing.append(key); continue
        by_idx = {int(o["index"]): o for o in graph["options"]}
        ni, ai = int(src["native_index"]), int(src["candidate_index"])
        no, ao = by_idx.get(ni), by_idx.get(ai)
        ne, ae = valid_embedding(no or {}), valid_embedding(ao or {})
        if ne is None or ae is None:
            missing.append(key); continue
        base = dict(src)
        # F0 is re-derived from the authoritative Gate B CSV and checked.
        for name in ("candidate_logit", "native_logit", "logit_difference", "candidate_rank_norm", "native_rank_norm", "candidate_distance_m", "native_distance_m", "candidate_path_estimate_m", "native_path_estimate_m", "candidate_count", "high_level_step_norm", "policy_entropy", "top1_top2_margin", "visited_or_history"):
            base[name] = float(src[name])
        base.update(f1_features(ne, ae))
        # F2/F3 use the graph's STOP/current state as an explicit frozen state
        # representation and the same instruction encoder output for the route.
        txt, token_mask = txt_by_ep[key[0]]
        base.update(progress_features(ne, ae, txt, token_mask))
        rows.append(base)
    return rows, missing


def numeric_features(rows, level):
    # Exact Gate B feature order, including categorical STOP/proposal flags and
    # raw decision index. Keeping this list literal prevents post-hoc feature
    # selection when comparing F0 against the semantic additions.
    f0 = ["high_level_step", "candidate_is_stop", "native_is_stop",
          "candidate_current_proposal", "native_current_proposal",
          "candidate_logit", "native_logit", "logit_difference",
          "candidate_rank_norm", "native_rank_norm", "candidate_distance_m",
          "native_distance_m", "candidate_path_estimate_m",
          "native_path_estimate_m", "candidate_count", "high_level_step_norm",
          "policy_entropy", "top1_top2_margin", "visited_or_history"]
    f1 = [k for k in rows[0] if k.startswith("f1_")]
    f2 = [k for k in rows[0] if k.startswith("f2_")]
    f3 = [k for k in rows[0] if k.startswith("f3_")]
    names = f0 + (f1 if level >= 1 else []) + (f2 if level >= 2 else []) + (f3 if level >= 3 else [])
    return names, np.asarray([[float(r[k]) for k in names] for r in rows], dtype=np.float64)


def fit_predict(xtr, ytr, xte, model_name):
    mean = xtr.mean(0); scale = xtr.std(0); scale[scale < 1e-8] = 1.0
    a = (xtr - mean) / scale; b = (xte - mean) / scale
    if model_name == "ridge":
        xb = np.c_[np.ones(len(a)), a]; eye = np.eye(xb.shape[1]); eye[0, 0] = 0
        w = np.linalg.solve(xb.T @ xb + eye, xb.T @ ytr)
        return 1 / (1 + np.exp(-np.clip(np.c_[np.ones(len(b)), b] @ w, -30, 30)))
    import torch
    torch.manual_seed(0)
    if model_name == "logistic": model = torch.nn.Linear(a.shape[1], 1)
    else: model = torch.nn.Sequential(torch.nn.Linear(a.shape[1], 16), torch.nn.ReLU(), torch.nn.Linear(16, 1))
    opt = torch.optim.Adam(model.parameters(), lr=0.03 if model_name == "logistic" else 0.01, weight_decay=1e-3)
    xt = torch.from_numpy(a).float(); yt = torch.from_numpy(ytr[:, None]).float()
    for _ in range(400 if model_name == "logistic" else 500):
        loss = torch.nn.functional.binary_cross_entropy_with_logits(model(xt), yt)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad(): return torch.sigmoid(model(torch.from_numpy(b).float())).numpy().ravel()


def evaluate(rows, probs, threshold):
    grouped = defaultdict(list)
    for r, p in zip(rows, probs): grouped[(r["episode_id"], int(r["high_level_step"]))].append((r, float(p)))
    selected = []
    for key, vals in grouped.items():
        r, p = max(vals, key=lambda x: x[1])
        if p >= threshold: selected.append((key, r, p))
    beneficial = sum(r["label"] == "INTERVENE" for _, r, _ in selected)
    harmful = sum(r["label"] == "KEEP" for _, r, _ in selected)
    ambiguous = sum(r["label"] == "AMBIGUOUS" for _, r, _ in selected)
    scenes = sorted({r["scene_id"] for _, r, _ in selected if r["label"] == "INTERVENE"})
    routes = sorted({r["episode_id"] for _, r, _ in selected if r["label"] == "INTERVENE"})
    states = len(grouped)
    return {"threshold": threshold, "eligible_states": states, "interventions": len(selected),
            "coverage": len(selected) / max(states, 1), "beneficial_interventions": beneficial,
            "harmful_keep_interventions": harmful, "ambiguous_interventions": ambiguous,
            "intervention_precision": beneficial / max(len(selected), 1),
            "intervention_recall": beneficial / max(sum(r["label"] == "INTERVENE" for r in rows), 1),
            "harm_rate": harmful / max(len(selected), 1),
            "ambiguous_intervention_rate": ambiguous / max(len(selected), 1),
            "positive_route_count": len(routes), "positive_scene_count": len(scenes),
            "positive_routes": routes, "positive_scenes": scenes}


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def main():
    p = argparse.ArgumentParser(); p.add_argument("--samples", type=Path, required=True)
    p.add_argument("--train-graph", type=Path, required=True); p.add_argument("--val-graph", type=Path, required=True)
    p.add_argument("--train-tokens", type=Path, required=True); p.add_argument("--val-tokens", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--base-pretrained", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=False)
    samples = read_csv(args.samples); train_s = [r for r in samples if r["cohort"] == "train"]; val_s = [r for r in samples if r["cohort"] == "val_unseen"]
    train_g, val_g = read_graph(args.train_graph), read_graph(args.val_graph)
    tokens = load_tokens([args.train_tokens, args.val_tokens])
    target_eps = {str(r["episode_id"]) for r in samples}; tokens = {k: v for k, v in tokens.items() if k in target_eps}
    txt = extract_txt(tokens, args.checkpoint, args.base_pretrained)
    train_rows, miss_tr = build_rows(train_s, train_g, txt); val_rows, miss_va = build_rows(val_s, val_g, txt)
    if len(train_rows) != len(train_s) or len(val_rows) != len(val_s):
        raise RuntimeError("Graph/token join incomplete: train missing %d, val missing %d" % (len(miss_tr), len(miss_va)))
    write_csv(args.output_dir / "per_sample.csv", train_rows + val_rows)
    summary = {"protocol": "Gate C instruction-progress representation diagnostic", "sample_count": len(train_rows) + len(val_rows), "state_count": len({(r['episode_id'], r['high_level_step']) for r in train_rows + val_rows}), "class_counts": dict(Counter(r['label'] for r in train_rows + val_rows)), "split": "train versus val_unseen scene-disjoint", "checkpoint": str(args.checkpoint), "checkpoint_sha256": sha256(args.checkpoint), "levels": {}, "reproduction_gate_b": {"samples": 1377, "states": 117, "routes": 31, "scenes": 18, "INTERVENE": 76, "KEEP": 533, "AMBIGUOUS": 768}, "missing": {"train": len(miss_tr), "val_unseen": len(miss_va)}}
    pred_rows = []
    for level in range(4):
        names, xtr = numeric_features(train_rows, level); _, xva = numeric_features(val_rows, level)
        fit = [r for r in train_rows if r["label"] in LABELS]; test = val_rows
        _, xf = numeric_features(fit, level); _, xt = numeric_features(test, level)
        y = np.asarray([LABELS[r["label"]] for r in fit], dtype=np.float64)
        level_out = {"feature_names": names, "feature_count": len(names), "train_fit_rows": len(fit), "heldout_rows": len(test), "models": {}}
        for model_name in ("logistic", "ridge", "small_mlp"):
            probs = fit_predict(xf, y, xt, model_name)
            level_out["models"][model_name] = {str(t): evaluate(test, probs, t) for t in THRESHOLDS}
            for r, prob in zip(test, probs):
                rr = {"level": "F%d" % level, "model": model_name, "probability": float(prob)}; rr.update(r); pred_rows.append(rr)
        summary["levels"]["F%d" % level] = level_out
    write_csv(args.output_dir / "heldout_predictions.csv", pred_rows)
    (args.output_dir / "feature_schema.json").write_text(json.dumps({"F0": numeric_features(train_rows,0)[0], "F1": numeric_features(train_rows,1)[0], "F2": numeric_features(train_rows,2)[0], "F3": numeric_features(train_rows,3)[0]}, indent=2) + "\n")
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
