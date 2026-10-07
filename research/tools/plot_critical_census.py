#!/usr/bin/env python3
"""Standalone figure of overlapping rescue counts and matched interrupt outcomes."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
 p=argparse.ArgumentParser();p.add_argument('--summary',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();d=json.loads(a.summary.read_text());a.output.mkdir(parents=True,exist_ok=False)
 fig,axes=plt.subplots(1,2,figsize=(10,4.2))
 vals=[d['ranking_opportunity']['count'],d['ranking_sr_and_ndtw_opportunity']['count'],d['termination_opportunity']['count'],d['unresolved']['count']]
 axes[0].barh(['Native move replacement','+ nDTW nondegradation','Termination replacement','Unresolved'],vals,color=['#4878a8','#5e9b72','#b99247','#999999'])
 total=d['control_routes']
 axes[0].set_xlim(0,total);axes[0].set_xlabel('Routes out of {} baseline failures'.format(total))
 axes[0].invert_yaxis();axes[0].set_title('Rescue opportunities overlap')
 for i,v in enumerate(vals):axes[0].text(v+.08,i,str(v),va='center')
 events=d['interrupt_results'];eps=sorted({r['case']['episode_id'] for r in events});colors=['#999999','#4878a8','#b99247']
 for mode,color,label in zip(['sense_only','interrupt_consume','interrupt_retain'],colors,['Sense only','Interrupt: consume ghost','Interrupt: retain ghost']):
  rows={r['case']['episode_id']:r for r in events if r['case']['mode']==mode}
  ys=[rows[ep]['delta']['ndtw']*100 for ep in eps];axes[1].plot(eps,ys,'o-',label=label,color=color)
 axes[1].axhline(0,color='#555555',linewidth=.7);axes[1].set_xlabel('Diagnostic route (all remain failed)');axes[1].set_ylabel('nDTW change (percentage points)');axes[1].set_title('Four frozen collision cuts; 0 rescues');axes[1].legend(fontsize=8)
 fig.suptitle('Privileged diagnostics, unchanged checkpoint / STOP / 15 decisions',fontsize=11)
 fig.tight_layout();fig.savefig(str(a.output/'critical_census.png'),dpi=180);fig.savefig(str(a.output/'critical_census.pdf'));plt.close(fig)

if __name__=='__main__':main()
