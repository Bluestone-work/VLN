#!/usr/bin/env python3
"""Visualize descriptive cost components; no fitted curves or inference claims."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args(); data = json.loads((args.root / 'analysis_001/summary.json').read_text())
    labels = ['Exploratory\nFULL', 'Replication\nFULL', 'Replication\nCOST-own', 'Replication\nCOST-matched']
    names = ['exploratory_full', 'replication_full', 'replication_cost_own', 'replication_cost_matched']
    fig, ax = plt.subplots(figsize=(10, 5.8), constrained_layout=True)
    x = np.arange(len(names)); positive = np.zeros(len(names)); negative = positive.copy()
    for group, label, color in [('immediate', 'Changed option', '#2374ab'),
                                ('later_navigation', 'Later navigation', '#d8842b'),
                                ('later_stop', 'Later STOP execution', '#8e5ca6')]:
        values = np.asarray([data['batches'][n]['all_events']['cost_groups'][group] / 96. for n in names])
        bottom = np.where(values >= 0, positive, negative)
        ax.bar(x, values, bottom=bottom, width=.62, label=label, color=color)
        positive += np.maximum(values, 0); negative += np.minimum(values, 0)
    total = [data['batches'][n]['all_events']['primitive_delta'] / 96. for n in names]
    ax.scatter(x, total, marker='D', color='black', s=45, label='Complete episode', zorder=3)
    for i, value in enumerate(total):
        ax.annotate('{:+.3f}'.format(value), (i, value), xytext=(15, 0), textcoords='offset points', va='center')
    ax.axhline(0, color='black', lw=.8); ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel('Primitive change / route (96 routes per arm)')
    ax.set_title('Immediate savings and later navigation costs\nRetrospective accounting; cohorts separate, control arms overlap')
    ax.legend(loc='upper left', fontsize=9); ax.grid(axis='y', alpha=.2)
    output = args.root / 'figures_001'; output.mkdir(exist_ok=False)
    fig.savefig(str(output / 'cost_components.png'), dpi=180)
    fig.savefig(str(output / 'cost_components.pdf'))
    plt.close(fig)


if __name__ == '__main__':
    main()
