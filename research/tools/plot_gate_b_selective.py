#!/usr/bin/env python3
"""Plot the pre-registered Gate B held-out precision/coverage sweep."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()

    import matplotlib.pyplot as plt

    summary = json.loads(args.summary.read_text())
    thresholds = [0.50, 0.70, 0.80, 0.90, 0.95]
    fig, (ax_precision, ax_ambiguity) = plt.subplots(1, 2, figsize=(10, 4), dpi=150)
    for model_name, model in summary['models'].items():
        rows = model['held_out_val_unseen_all_labels']
        coverage = [rows[str(t)]['coverage'] for t in thresholds]
        precision = [rows[str(t)]['intervention_precision'] for t in thresholds]
        ambiguity = [rows[str(t)]['ambiguous_rate'] for t in thresholds]
        ax_precision.plot(coverage, precision, marker='o', label=model_name)
        ax_ambiguity.plot(coverage, ambiguity, marker='o', label=model_name)
        for x, y, threshold in zip(coverage, precision, thresholds):
            ax_precision.annotate(f'{threshold:.2g}', (x, y), fontsize=7,
                                  xytext=(3, 3), textcoords='offset points')

    ax_precision.set_title('Held-out intervention precision')
    ax_precision.set_xlabel('Coverage (states intervened)')
    ax_precision.set_ylabel('Beneficial interventions / interventions')
    ax_precision.set_xlim(0, 1.02)
    ax_precision.set_ylim(0, 1.02)
    ax_precision.grid(alpha=0.25)
    ax_precision.legend(fontsize=8)
    ax_ambiguity.set_title('Held-out ambiguity among interventions')
    ax_ambiguity.set_xlabel('Coverage (states intervened)')
    ax_ambiguity.set_ylabel('Ambiguous interventions / interventions')
    ax_ambiguity.set_xlim(0, 1.02)
    ax_ambiguity.set_ylim(0, 1.02)
    ax_ambiguity.grid(alpha=0.25)
    fig.suptitle('Gate B: native-relative selective intervention')
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)
    plt.close(fig)


if __name__ == '__main__':
    main()
