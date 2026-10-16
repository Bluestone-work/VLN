#!/usr/bin/env python3
"""Render a compact visual audit of the latent transition/value experiment."""
import argparse, html, json, math
from pathlib import Path
from collections import OrderedDict
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]


def font(size, bold=False):
    p = '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc' if bold else '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    return ImageFont.truetype(p, size)


def unpack_array(x):
    return np.asarray(x['__array__'] if isinstance(x, dict) and '__array__' in x else x)


def nearest_sector(b):
    return min(range(0, 360, 30), key=lambda x: abs((x-b+180) % 360-180))


def load_rgb(path):
    a=np.load(str(path), allow_pickle=False)[...,:3]
    return Image.fromarray(a.astype(np.uint8)).convert('RGB')


def save_case(out, row, pair, graph, title):
    sec=pair['features'].get('sector_deg')
    files=graph['semantic_visual']['files']
    rgbkey='rgb_{}'.format(int(sec)) if sec else 'rgb'
    current=load_rgb(files[rgbkey]).resize((448,448),Image.BICUBIC)
    endpoint=load_rgb(ROOT / row['endpoint_files']['rgb']).resize((448,448),Image.BICUBIC)
    canvas=Image.new('RGB',(960,650),'#101a2b'); d=ImageDraw.Draw(canvas)
    d.text((24,18),title,font=font(28,True),fill='white')
    d.text((24,62),'Episode {} · step {} · candidate {} · graph {}'.format(row['episode_id'],row['high_level_step'],row['index'],row['graph_id']),font=font(18),fill='#c8d5e8')
    d.text((24,92),'label: {}  |  bearing: {:.1f}° → sector {}°  |  distance: {:.2f} m'.format(pair['label'],pair['features']['bearing_deg'],sec,pair['features']['distance']),font=font(18),fill='#ffbd68' if pair['label']=='BENEFIT' else '#ff7d8b')
    canvas.paste(current,(24,145));canvas.paste(endpoint,(488,145))
    d.text((28,602),'当前 pre-action sector（线上可用输入）',font=font(17),fill='#77d5cc')
    d.text((492,602),'执行 candidate 后 endpoint（仅 dynamics target）',font=font(17),fill='#ffbd68')
    path=out/'cases'/('{}__{}.png'.format(pair['label'].lower(),pair['episode_id']))
    canvas.save(path);return path.name


def save_context_case(out, row, pair, graph, title):
    """Show all pre-action sectors so apparent zoom is visibly separated from motion."""
    files=graph['semantic_visual']['files']; sec=int(pair['features']['sector_deg'])
    sheet=Image.new('RGB',(900,560),'#101a2b');d=ImageDraw.Draw(sheet)
    d.text((20,16),title+'：相机没有缩放，机器人执行了 {:.2f} m 的候选位移'.format(pair['features']['distance']),font=font(25,True),fill='white')
    for j,b in enumerate(range(0,360,30)):
        key='rgb_{}'.format(b) if b else 'rgb'; im=load_rgb(files[key]).resize((210,145),Image.BICUBIC)
        x=20+(j%4)*220;y=65+(j//4)*165
        sheet.paste(im,(x,y)); color='#ffbd68' if b==sec else '#60758e';d.rectangle((x,y,x+210,y+145),outline=color,width=5 if b==sec else 2)
        d.text((x+7,y+7),'bearing {}°{}'.format(b,'  ← candidate' if b==sec else ''),font=font(15,True if b==sec else False),fill='#ffbd68' if b==sec else 'white')
    outp=out/'cases'/('{}__context.png'.format(pair['label'].lower()));sheet.save(outp)
    return outp.name


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);args=ap.parse_args();out=args.output.resolve();(out/'cases').mkdir(parents=True,exist_ok=True)
    summary=json.loads((ROOT/'research/results/latent_transition_value_full_001/summary.json').read_text())
    pairs={(str(x['episode_id']),int(x['high_level_step']),int(x['alternative_index'])):x for x in map(json.loads,(ROOT/'research/results/native_single_action_full_return_44_002/directional_pairs_002/pairs.jsonl').open())}
    graphs={(str(x['episode_id']),int(x['high_level_step'])):x for x in map(json.loads,(ROOT/'research/results/semantic_node_memory_multiview_001/graph_options/graph_options.jsonl').open())}
    endpoint_rows=[json.loads(x) for x in (ROOT/'research/results/native_transition_endpoint_full_001/native_endpoint_options.jsonl').open()]
    selected=OrderedDict()
    for er in endpoint_rows:
        k=(str(er['episode_id']),int(er['high_level_step']),int(er['index']));p=pairs.get(k)
        if p and er['action_type']==4 and p['label'] not in selected:selected[p['label']]=(er,p,graphs[(k[0],k[1])])
        if len(selected)==2:break
    cases=[]
    context_cases=[]
    for label,(er,p,g) in selected.items():
        cases.append(save_case(out,er,p,g,'{} example'.format(label)))
        context_cases.append(save_context_case(out,er,p,g,'{} example'.format(label)))

    names=['F0','F0 + geometry','F0 + predicted\ntransition','F0 + oracle\nendpoint']
    keys=['F0','F0+geometry','F0+predicted_transition','F0+oracle_endpoint']; metrics=['auroc','auprc','pairwise'];
    vals={m:[summary['models'][k][m] for k in keys] for m in metrics}
    fig,ax=plt.subplots(figsize=(11,5.5));x=np.arange(len(names));w=.24;colors=['#667eea','#43c6c7','#ffbd68']
    for j,m in enumerate(metrics):ax.bar(x+(j-1)*w,vals[m],w,label=m.upper(),color=colors[j])
    ax.set_ylim(0,1);ax.set_xticks(x);ax.set_xticklabels(names);ax.set_ylabel('score');ax.set_title('Full 44-route scene-disjoint probe');ax.grid(axis='y',alpha=.25);ax.legend(frameon=False,ncol=3);fig.tight_layout();fig.savefig(out/'metrics.png',dpi=160);plt.close(fig)
    # Paired delta confidence intervals for the actual transition arm.
    delta=summary['paired_deltas']['F0+predicted_transition minus F0'];fig,ax=plt.subplots(figsize=(8,4));labels=['AUROC','AUPRC','Pairwise'];mid=[(delta[k][0]+delta[k][1])/2 for k in ['auroc','auprc','pairwise']];lo=[mid[i]-delta[k][0] for i,k in enumerate(['auroc','auprc','pairwise'])];hi=[delta[k][1]-mid[i] for i,k in enumerate(['auroc','auprc','pairwise'])];ax.errorbar(labels,mid,yerr=[lo,hi],fmt='o',color='#ffbd68',ecolor='#ffbd68',capsize=6,ms=9);ax.axhline(0,color='#777',lw=1);ax.set_ylim(-.04,.04);ax.set_ylabel('transition − F0');ax.set_title('Scene-bootstrap 95% CI');ax.grid(axis='y',alpha=.25);fig.tight_layout();fig.savefig(out/'paired_delta_ci.png',dpi=160);plt.close(fig)
    instr=selected[next(iter(selected))][1].get('instruction','') if selected else ''
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>ETPNav latent transition visualization</title><style>body{max-width:1180px;margin:24px auto;padding:0 25px;background:#101a2b;color:#d9e5f5;font:17px/1.65 sans-serif}h1{color:#fff}h2{color:#77d5cc;margin-top:30px}.card{background:#17263b;border-radius:14px;padding:20px;margin:18px 0}img{max-width:100%;border-radius:10px;background:#fff}.flow{display:flex;align-items:center;gap:10px;flex-wrap:wrap}.box{padding:13px 16px;border:1px solid #3b617d;border-radius:10px;background:#203650}.arrow{font-size:25px;color:#ffbd68}.warn{color:#ffbd68}.good{color:#77d5cc}table{border-collapse:collapse;width:100%}td,th{padding:8px;border-bottom:1px solid #33465d;text-align:right}td:first-child,th:first-child{text-align:left}code{color:#ffbd68}</style>
<h1>原始 ETPNav：短期 latent transition/value probe</h1><div class="card"><p>这页展示的是“做了什么”，不是在线导航录像。原始 ETPNav policy、Graph Transformer、controller 和 STOP 都没有改变。</p><div class="flow"><div class="box">当前节点 pre-action RGB-D sector<br><small>线上可用</small></div><div class="arrow">＋</div><div class="box">candidate bearing / distance<br>native logit summary</div><div class="arrow">→</div><div class="box">预测下一时刻视觉 latent<br><small>PCA-16 ResNet latent</small></div><div class="arrow">→</div><div class="box">BENEFIT / HARM value probe</div></div><p class="warn">Endpoint 图像只用于训练 dynamics target，不能作为线上输入。右侧案例中的 endpoint 是“如果执行该 action 后看到的图像”，不是模型事先看到的输入。</p></div>
<h2>1. 数据规模</h2><div class="card"><p>4,251 个 exact endpoint branches · 373 states · 44 episodes · 11 scenes · 3,877 个 causal alternatives。</p><p>BENEFIT: 430，HARM: 3,447；所有 native selected sentinel 373/373 精确通过。</p></div>
<h2>2. 当前 sector 与 endpoint 对照</h2>'''
    for c,cc in zip(cases,context_cases):page+='<div class="card"><img src="cases/{}"><br><img src="cases/{}"></div>'.format(cc,c)
    page+='''<p>左图是 action 执行前对应 bearing 的当前 sector；右图是 exact replay 后的 endpoint RGB。它们用于说明 transition target 的含义。正式 value probe 不会读取右图。</p><h2>3. 指标对比</h2><div class="card"><img src="metrics.png"></div><table><tr><th>模型</th><th>AUROC</th><th>AUPRC</th><th>within-state pairwise</th></tr>'''
    for n,k in zip(names,keys):page+='<tr><td>{}</td><td>{:.3f}</td><td>{:.3f}</td><td>{:.3f}</td></tr>'.format(n.replace('\n',' '),summary['models'][k]['auroc'],summary['models'][k]['auprc'],summary['models'][k]['pairwise'])
    page+='''</table><h2>4. Transition 相对 native 的不确定性</h2><div class="card"><img src="paired_delta_ci.png"><p>横线穿过 0 表示 scene-bootstrap 没有证明 transition 分支稳定优于 F0。</p></div><h2>结论</h2><div class="card"><p><span class="warn">当前一步 endpoint RGB latent transition 没有通过 GO gate。</span> 它没有稳定提供 native ETPNav 之外的 waypoint value signal。下一步若继续，应预测 instruction-conditioned traversability 或 graph/history transition state，而不是继续生成未来 RGB。</p><p><a href="../LATENT_TRANSITION_VALUE_REPORT.md">打开完整报告</a></p></div></html>'''
    (out/'index.html').write_text(page);(out/'README.txt').write_text('Open index.html in a browser. Endpoint images are target-only diagnostics, not online inputs.\n')
    print(json.dumps({'output':str(out),'cases':cases,'metrics':str(out/'metrics.png'),'delta_ci':str(out/'paired_delta_ci.png')}))

if __name__=='__main__':main()
