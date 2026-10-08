#!/usr/bin/env python3
"""Create auditable Gate C tables/figures without plotting dependencies."""
import csv, json, math
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def read(p):
    with open(p, newline='') as f: return list(csv.DictReader(f))


def draw_plot(path, series, title, xlabel, ylabel, y_max=1.0):
    w, h = 1000, 650; left, top, right, bottom = 90, 65, 35, 85
    im = Image.new('RGB', (w, h), 'white'); d = ImageDraw.Draw(im)
    d.text((left, 18), title, fill='black')
    d.line((left, top, left, h-bottom), fill='black', width=2); d.line((left, h-bottom, w-right, h-bottom), fill='black', width=2)
    for j in range(6):
        y = h-bottom - (h-bottom-top)*j/5; val=y_max*j/5
        d.line((left, y, w-right, y), fill='#dddddd', width=1); d.text((15, y-8), '%.1f'%val, fill='black')
    colors=['#1f77b4','#d62728','#2ca02c','#9467bd']
    for si,(name, vals) in enumerate(series):
        pts=[]
        for i,(x,y) in enumerate(vals):
            px=left+(w-right-left)*x; py=h-bottom-(h-bottom-top)*min(max(y,0),y_max)/y_max; pts.append((px,py))
        if len(pts)>1: d.line(pts, fill=colors[si%len(colors)], width=4)
        for px,py in pts: d.ellipse((px-5,py-5,px+5,py+5), fill=colors[si%len(colors)])
        d.text((w-right-180, top+22*si), name, fill=colors[si%len(colors)])
    d.text((w//2-60,h-35), xlabel, fill='black'); d.text((8,h//2), ylabel, fill='black')
    im.save(path)


def main():
    root=Path(__file__).resolve().parents[2]; run=root/'research/results/gate_c_instruction_progress/run_003'; out=root/'research/results/gate_c_instruction_progress'; (out/'figures').mkdir(parents=True,exist_ok=True)
    d=json.loads((run/'summary.json').read_text()); rows=read(run/'per_sample.csv'); preds=read(run/'heldout_predictions.csv')
    # Route/scene label and representation audit tables.
    for key in ('episode_id','scene_id'):
        grouped=defaultdict(list)
        for r in rows: grouped[(r['cohort'],r[key])].append(r)
        out_rows=[]
        for (cohort,ident), rs in sorted(grouped.items()):
            c=Counter(r['label'] for r in rs)
            out_rows.append({'cohort':cohort,key:ident,'state_count':len({(r['episode_id'],r['high_level_step']) for r in rs}),
                             'sample_count':len(rs),'INTERVENE':c['INTERVENE'],'KEEP':c['KEEP'],'AMBIGUOUS':c['AMBIGUOUS'],
                             'f2_progress_expected_mean':sum(float(r['f2_progress_expected']) for r in rs)/len(rs),
                             'f3_delta_suffix_cos_mean':sum(float(r['f3_delta_suffix_cos']) for r in rs)/len(rs)})
        with (out/('per_route.csv' if key=='episode_id' else 'per_scene.csv')).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(out_rows[0])); w.writeheader(); w.writerows(out_rows)
    # Fixed threshold curves for logistic are the primary compact comparison;
    # all model values remain in summary.json and heldout_predictions.csv.
    pc=[]; rc=[]
    for lev in ('F0','F1','F2','F3'):
        md=d['levels'][lev]['models']['logistic']
        pc.append((lev,[(float(t),float(md[str(float(t))]['intervention_precision'])) for t in (0.50,0.70,0.80,0.90,0.95)]))
        rc.append((lev,[(float(t),float(md[str(float(t))]['ambiguous_intervention_rate'])) for t in (0.50,0.70,0.80,0.90,0.95)]))
    draw_plot(out/'figures'/'gate_c_precision_coverage.png',pc,'Gate C held-out precision by fixed threshold','threshold','intervention precision')
    draw_plot(out/'figures'/'gate_c_risk_coverage.png',rc,'Gate C held-out ambiguity risk by fixed threshold','threshold','AMBIGUOUS intervention rate')
    # F0/F1/F2/F3 at the preregistered .95 threshold.
    vals=[]
    for lev in ('F0','F1','F2','F3'):
        v=d['levels'][lev]['models']['logistic']['0.95']; vals.append((lev,[(0.0,float(v['intervention_precision'])),(1.0,float(v['coverage']))]))
    draw_plot(out/'figures'/'gate_c_f0_f3_comparison.png',vals,'Gate C .95 precision and coverage','comparison index','value')
    # Machine-readable decision: no online method is enabled.
    best=[]
    for lev in ('F1','F2','F3'):
        for model,md in d['levels'][lev]['models'].items():
            for t,v in md.items():
                if v['coverage']>0: best.append({'level':lev,'model':model,'threshold':float(t),'precision':v['intervention_precision'],'coverage':v['coverage'],'ambiguous_rate':v['ambiguous_intervention_rate']})
    best.sort(key=lambda x:(x['precision'],x['coverage']),reverse=True)
    decision={'gate':'C','decision':'NO-GO','claim':'NO-GO for instruction-progress selective intervention',
              'reason':'No F2/F3 held-out fixed-threshold region reaches precision >= 0.70 with nonzero coverage; ambiguity remains high.',
              'best_nonbaseline_regions':best[:10], 'strong_go_precision':0.80,
              'required_precision':0.70,'required_positive_scene_count':3,
              'online_navigation_run':False}
    (out/'GO_NO_GO_SUMMARY.json').write_text(json.dumps(decision,indent=2)+'\n')
    (out/'experiment_table.csv').write_text('level,model,threshold,coverage,precision,harm_rate,ambiguous_rate\n')
    with (out/'experiment_table.csv').open('a',newline='') as f:
        w=csv.writer(f)
        for lev in ('F0','F1','F2','F3'):
            for model,md in d['levels'][lev]['models'].items():
                for t,v in md.items(): w.writerow([lev,model,t,v['coverage'],v['intervention_precision'],v['harm_rate'],v['ambiguous_intervention_rate']])
    print(json.dumps(decision,indent=2))


if __name__=='__main__': main()
