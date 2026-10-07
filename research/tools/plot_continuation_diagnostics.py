#!/usr/bin/env python3
"""Plot all new intervention cases, including unfavorable and delayed outcomes."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from run_option_capture import digest


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--summary',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args()
    data=json.loads(args.summary.read_text())
    if not data['all_accounting_checks_passed']:raise ValueError('Cost accounting must pass')
    rows=[r for r in data['events'] if r['batch']=='prospective_train5']
    args.output_dir.mkdir(parents=True,exist_ok=False)
    fig,axes=plt.subplots(1,2,figsize=(12,4.8))
    for r in rows:
        h=r['descriptive_horizon_profile']
        x=[v['horizon'] for v in h]
        label='Episode '+r['episode_id']
        axes[0].plot(x,[v['extra_goal_progress_m'] for v in h],marker='o',markersize=3,label=label)
        axes[1].plot(x,[v['primitive_delta'] for v in h],marker='o',markersize=3,label=label)
    for ax in axes:
        ax.axhline(0,color='#333333',linewidth=.8)
        ax.axvline(2,color='#555555',linestyle='--',linewidth=1,label='Registered H=2')
        ax.set_xlabel('High-level decisions since intervention (including replacement)')
        ax.grid(alpha=.2)
    axes[0].set_ylabel('Extra goal progress (m; positive is closer)')
    axes[1].set_ylabel('Cumulative primitive delta (negative uses fewer)')
    axes[0].legend(fontsize=8)
    fig.suptitle('Short-window labels and actual continuations: all five new training cases')
    fig.text(.5,.015,'Post-hoc descriptive curves; privileged diagnostics. STOP is absorbing offline. Decision windows have unequal costs.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.045,1,.94))
    for ext in ['png','pdf']:fig.savefig(str(args.output_dir/('all_five_horizon_profiles.'+ext)),dpi=170)
    plt.close(fig)
    out={'all_new_cases_plotted':[r['episode_id'] for r in rows],
         'analysis_type':'post-hoc descriptive figure; no horizon tuning',
         'hashes':{str(f):digest(f) for f in [args.summary,Path(__file__)]}}
    (args.output_dir/'manifest.json').write_text(json.dumps(out,indent=2)+'\n')


if __name__=='__main__':main()
