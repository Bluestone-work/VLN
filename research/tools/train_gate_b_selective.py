#!/usr/bin/env python3
"""Fit only small Gate B feasibility models and evaluate selective intervention."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


LABELS = {'KEEP': 0.0, 'INTERVENE': 1.0}
EXCLUDE = {
    'cohort', 'split', 'episode_id', 'scene_id', 'label', 'native_success',
    'native_failure', 'native_full_success', 'alternative_full_success',
}


def read(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream))


def features(rows):
    names = [k for k in rows[0] if k not in EXCLUDE]
    names = [k for k in names if k not in {'candidate_index', 'native_index'}]
    matrix = np.asarray([[float(r[k]) for k in names] for r in rows], dtype=np.float32)
    return names, matrix


def fit_logistic(x, y, seed=0):
    torch.manual_seed(seed)
    model = torch.nn.Linear(x.shape[1], 1)
    opt = torch.optim.Adam(model.parameters(), lr=0.03, weight_decay=1e-4)
    xt = torch.from_numpy(x); yt = torch.from_numpy(y[:, None])
    for _ in range(400):
        loss = torch.nn.functional.binary_cross_entropy_with_logits(model(xt), yt)
        opt.zero_grad(); loss.backward(); opt.step()
    return lambda z: torch.sigmoid(model(torch.from_numpy(z))).detach().numpy().ravel()


def fit_ridge(x, y, ridge=1.0):
    xb = np.concatenate([np.ones((len(x), 1), dtype=np.float32), x], axis=1)
    eye = np.eye(xb.shape[1], dtype=np.float32); eye[0, 0] = 0.0
    w = np.linalg.solve(xb.T.dot(xb) + ridge * eye, xb.T.dot(y))
    return lambda z: 1.0 / (1.0 + np.exp(-np.clip(np.concatenate([np.ones((len(z), 1), dtype=np.float32), z], axis=1).dot(w), -30, 30)))


def fit_mlp(x, y, seed=0):
    torch.manual_seed(seed)
    model = torch.nn.Sequential(torch.nn.Linear(x.shape[1], 16), torch.nn.ReLU(), torch.nn.Linear(16, 1))
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-3)
    xt = torch.from_numpy(x); yt = torch.from_numpy(y[:, None])
    for _ in range(500):
        loss = torch.nn.functional.binary_cross_entropy_with_logits(model(xt), yt)
        opt.zero_grad(); loss.backward(); opt.step()
    return lambda z: torch.sigmoid(model(torch.from_numpy(z))).detach().numpy().ravel()


def evaluate(rows, probs, threshold):
    grouped = defaultdict(list)
    for row, prob in zip(rows, probs):
        grouped[(row['episode_id'], int(row['high_level_step']))].append((row, float(prob)))
    interventions = []
    for key, items in grouped.items():
        row, prob = max(items, key=lambda item: item[1])
        if prob >= threshold:
            interventions.append((key, row, prob))
    beneficial = sum(row['label'] == 'INTERVENE' for _, row, _ in interventions)
    harmful = sum(row['label'] == 'KEEP' for _, row, _ in interventions)
    ambiguous = sum(row['label'] == 'AMBIGUOUS' for _, row, _ in interventions)
    positives = sum(row['label'] == 'INTERVENE' for row in rows)
    state_count = len(grouped)
    return {
        'threshold': threshold, 'eligible_states': state_count,
        'interventions': len(interventions), 'coverage': len(interventions) / max(state_count, 1),
        'beneficial_interventions': beneficial, 'harmful_keep_interventions': harmful,
        'ambiguous_interventions': ambiguous,
        'intervention_precision': beneficial / max(len(interventions), 1),
        'intervention_recall': beneficial / max(positives, 1),
        'harm_rate': harmful / max(len(interventions), 1),
        'ambiguous_rate': ambiguous / max(len(interventions), 1),
        'native_success_destruction_rate': None,
        'native_success_destruction_note': 'Not estimable: audited Gate B cohorts contain only native-failure routes.',
    }


def main():
    p = argparse.ArgumentParser(); p.add_argument('--samples', type=Path, required=True); p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=False)
    all_rows = read(args.samples)
    fit_rows = [r for r in all_rows if r['label'] in LABELS]
    train_rows = [r for r in fit_rows if r['cohort'] == 'train']
    test_rows = [r for r in all_rows if r['cohort'] == 'val_unseen']
    names, x_train = features(train_rows); _, x_test = features(test_rows)
    mean = x_train.mean(0); scale = x_train.std(0); scale[scale < 1e-6] = 1.0
    x_train = (x_train - mean) / scale; x_test = (x_test - mean) / scale
    y = np.asarray([LABELS[r['label']] for r in train_rows], dtype=np.float32)
    models = {'logistic': fit_logistic(x_train, y), 'ridge': fit_ridge(x_train, y), 'small_mlp': fit_mlp(x_train, y)}
    thresholds = [0.50, 0.70, 0.80, 0.90, 0.95]
    output = {
        'feature_names': names,
        'train_rows': len(train_rows),
        'train_rows_excluded_ambiguous': sum(r['label'] == 'AMBIGUOUS' for r in all_rows if r['cohort'] == 'train'),
        'test_rows_all_labels': len(test_rows),
        'held_out_split': 'scene-disjoint cohort val_unseen',
        'training_policy': 'fit binary models on INTERVENE/KEEP only; evaluate all labels and report AMBIGUOUS separately',
        'models': {},
    }
    pred_rows = []
    for model_name, predict in models.items():
        train_prob = predict(x_train); test_prob = predict(x_test)
        output['models'][model_name] = {'train': {str(t): evaluate(train_rows, train_prob, t) for t in thresholds}, 'held_out_val_unseen_all_labels': {str(t): evaluate(test_rows, test_prob, t) for t in thresholds}}
        for r, prob in zip(test_rows, test_prob):
            rr = dict(r); rr['model'] = model_name; rr['probability'] = float(prob); pred_rows.append(rr)
    fields = list(pred_rows[0].keys())
    with (args.output_dir / 'heldout_predictions.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(pred_rows)
    (args.output_dir / 'summary.json').write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps(output, indent=2))


if __name__ == '__main__': main()
