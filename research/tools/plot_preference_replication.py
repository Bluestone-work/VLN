#!/usr/bin/env python3
"""Standalone paired-effect plot; raw values and intervals come from frozen analysis."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--summary',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    result=json.loads(a.summary.read_text());arms=list(result['arms'])
    labels={'full':'FULL','cost_own':'COST (own timing)','cost_matched':'COST (matched timing)'}
    names=[labels.get(n,'Random '+n.split('_')[-1][-2:]) for n in arms]
    colors=['#2364aa','#d78027','#168a80']+['#888888']*3
    fig,axes=plt.subplots(2,2,figsize=(12,7),constrained_layout=True)
    for ax,(metric,label,scale) in zip(axes.flat,[('success','SR change (pp)',100),('spl','SPL change (pp)',100),('ndtw','nDTW change (pp)',100),('primitive_action_count','Primitive count change / route',1)]):
        for i,(name,color) in enumerate(zip(arms,colors)):
            m=result['arms'][name]['metrics'][metric];center=m['delta']*scale;lo,hi=[v*scale for v in m['scene_cluster_ci95']]
            ax.plot([lo,hi],[i,i],color=color,lw=2)
            ax.plot(center,i,'o',color=color,markersize=6)
        ax.axvline(0,color='black',lw=.8,ls='--');ax.set_yticks(range(len(arms)));ax.set_yticklabels(names);ax.invert_yaxis()
        ax.set_xlabel(label);ax.grid(axis='x',alpha=.2)
    fig.suptitle('Prospective R2R training-route replication: 96 routes, 8 overlapping scenes\nPaired changes vs native; exploratory 95% scene-bootstrap intervals',fontsize=12)
    a.output.mkdir(exist_ok=False)
    fig.savefig(str(a.output/'paired_effects.png'),dpi=180)
    fig.savefig(str(a.output/'paired_effects.pdf'))
    plt.close(fig)

if __name__=='__main__':main()
