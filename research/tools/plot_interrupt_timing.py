#!/usr/bin/env python3
"""Display every registered single-cut return without selecting favorable cuts."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser(); p.add_argument('--analysis', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); a = p.parse_args()
    rows = json.loads((a.analysis/'cases.json').read_text())
    a.output.mkdir(parents=True, exist_ok=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharex=True, sharey=True)
    groups = [(False, False, 'Failed -> failed', '#999999', '.'),
              (False, True, 'Failed -> success', '#1976d2', '*'),
              (True, True, 'Success -> success', '#2e7d32', 'o'),
              (True, False, 'Success -> failed', '#c62828', 'x')]
    for ax, mode, title in zip(axes, ['interrupt_consume', 'interrupt_retain'], ['Consume pending ghost', 'Retain pending ghost']):
        for before, after, label, color, marker in groups:
            group = [r for r in rows if r['case']['mode'] == mode and bool(r['baseline']['success']) == before and bool(r['metrics']['success']) == after]
            ax.scatter([r['delta']['steps_taken'] for r in group], [100*r['delta']['ndtw'] for r in group],
                       s=40 if marker == '*' else 18, marker=marker, color=color, alpha=0.65, label=label)
        ax.axhline(0, color='black', linewidth=0.6); ax.axvline(0, color='black', linewidth=0.6)
        ax.set_title(title); ax.set_xlabel('Primitive count change vs native')
        ax.grid(alpha=0.15)
    axes[0].set_ylabel('nDTW change (percentage points)')
    axes[1].legend(fontsize=8)
    fig.suptitle('Retrospective timing oracle: all 175 correlated cuts per arm', fontsize=11)
    fig.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(str(a.output/('all_cut_returns.'+ext)), dpi=160, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
