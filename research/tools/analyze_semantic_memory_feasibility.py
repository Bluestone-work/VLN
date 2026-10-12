#!/usr/bin/env python3
"""Descriptive, outcome-blind diagnostics for captured semantic memory."""
import argparse, json, math, random, re
from collections import defaultdict
from pathlib import Path


def rank(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]: j += 1
        v = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1): out[order[k]] = v
        i = j + 1
    return out


def corr(a, b):
    if len(a) < 2: return None
    ra, rb = rank(a), rank(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    da = sum((x - ma) ** 2 for x in ra); db = sum((x - mb) ** 2 for x in rb)
    return sum((x - ma) * (y - mb) for x, y in zip(ra, rb)) / math.sqrt(da * db) if da and db else 0.0


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--semantic', type=Path, required=True); ap.add_argument('--graph', type=Path, required=True); ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    sem = [json.loads(x) for x in args.semantic.open() if x.strip()]
    graph = [json.loads(x) for x in args.graph.open() if x.strip()]
    nodes = [x for x in sem if x.get('native_logits') is not None]
    by_ep = defaultdict(list); by_scene = defaultdict(list)
    for x in nodes:
        by_ep[str(x['episode_id'])].append(x); by_scene[str(x.get('scene_id'))].append(x)
    stop, evidence, agreement = [], [], []
    for x in nodes:
        vals = [v for v in (x.get('native_logits') or []) if isinstance(v, (int, float)) and math.isfinite(v)]
        stop_score = float(vals[0]) if vals else None
        sem_score = x.get('instruction_relevance', {}).get('semantic_similarity')
        if stop_score is not None and sem_score is not None:
            stop.append(stop_score); evidence.append(float(sem_score))
        broad = x.get('landmarks') or []
        phrases = x.get('instruction_landmark_hypotheses') or []
        if broad and phrases:
            agreement.append(broad[0]['label'] in phrases or broad[0]['label'].replace('a ', '') in phrases)
    progress = []
    for ep, seq in by_ep.items():
        seq = sorted(seq, key=lambda x: int(x['high_level_step']))
        phrase_order = seq[0].get('instruction_landmark_hypotheses') or []
        indices = {p: i for i, p in enumerate(phrase_order)}
        observed = [indices.get(x.get('instruction_relevance', {}).get('matched_phrase')) for x in seq]
        observed = [i for i in observed if i is not None]
        inversions = sum(1 for i in range(len(observed)) for j in range(i + 1, len(observed)) if observed[j] < observed[i])
        progress.append({'episode_id': ep, 'steps': len(seq), 'phrase_count': len(phrase_order), 'matched_steps': len(observed), 'inversions': inversions, 'monotone': inversions == 0 and bool(observed)})
    rng = random.Random(100)
    scene_corr = []
    for _ in range(10000):
        sampled = [rng.choice(list(by_scene)) for _ in by_scene]
        aa, bb = [], []
        for scene in sampled:
            for x in by_scene[scene]:
                vals = [v for v in (x.get('native_logits') or []) if isinstance(v, (int, float)) and math.isfinite(v)]
                if vals and x.get('instruction_relevance', {}).get('semantic_similarity') is not None:
                    aa.append(float(vals[0])); bb.append(float(x['instruction_relevance']['semantic_similarity']))
        c = corr(aa, bb)
        if c is not None: scene_corr.append(c)
    scene_corr.sort()
    out = {
        'decision_nodes': len(nodes), 'episodes': len(by_ep), 'scenes': len(by_scene),
        'stop_vs_instruction_semantic_spearman': corr(stop, evidence),
        'stop_vs_instruction_semantic_pairs': len(stop),
        'scene_bootstrap_10000_spearman_ci': [scene_corr[250], scene_corr[-251]] if scene_corr else None,
        'top_broad_label_matches_instruction_phrase_rate': sum(agreement) / len(agreement) if agreement else None,
        'broad_label_pairs': len(agreement),
        'progress_episode_count': len(progress),
        'progress_monotone_episode_rate': sum(x['monotone'] for x in progress) / len(progress) if progress else None,
        'progress_total_inversions': sum(x['inversions'] for x in progress),
        'native_ghost_selection_tested': False,
        'native_ghost_selection_reason': 'single current panorama is not action-conditioned; no per-ghost visual corridor or endpoint observation was captured',
        'future_outcomes_used_as_input': False,
        'notes': ['STOP comparison is correlation only; no privileged goal-node semantic label is inferred.', 'CLIP prompt scores are probes and are not object, relation, or instruction-completion ground truth.', 'Progress order uses lexical phrase order from the instruction and is descriptive, not a learned label.']
    }
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps({'summary': out, 'progress_by_episode': progress}, indent=2) + '\n'); print(json.dumps(out, indent=2))


if __name__ == '__main__': main()
