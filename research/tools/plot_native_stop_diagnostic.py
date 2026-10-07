#!/usr/bin/env python3
"""Visualize the recorded interior-arrival exception, not an altered rollout."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);args=p.parse_args()
    data=json.loads(args.source.read_text())
    rows=[r for r in data['episodes'] if r['route_audit']['oracle_arrival_without_boundary']]
    if len(rows)!=1:raise ValueError('Plot contract expects the one observed exception')
    r=rows[0];pts=r['primitives'];fig,ax=plt.subplots(figsize=(8,4))
    ax.plot([p['episode_primitive'] for p in pts],[p['distance_m'] for p in pts],label='Recorded primitive endpoint')
    ends=[p for p in pts if p['is_option_end']]
    ax.scatter([p['episode_primitive'] for p in ends],[p['distance_m'] for p in ends],color='black',label='High-level option end',zorder=3)
    inside=[p for p in pts if p['within_3m']]
    ax.scatter([p['episode_primitive'] for p in inside],[p['distance_m'] for p in inside],color='orange',label='Inside 3 m (3 points)',zorder=4)
    ax.axhline(3,color='red',linestyle='--',label='Native success distance')
    ax.set(xlabel='Executed primitive count',ylabel='Geodesic goal distance (m)',title='Unseen episode 1593: proximity occurs between decisions')
    ax.legend(fontsize=8);fig.tight_layout();args.output_dir.mkdir(parents=True,exist_ok=False)
    fig.savefig(str(args.output_dir/'interior_arrival.png'),dpi=180);fig.savefig(str(args.output_dir/'interior_arrival.pdf'))


if __name__=='__main__':main()
