#!/usr/bin/env python3
"""Select a fresh metadata-only route sample for full-return diagnostics."""
import argparse
import collections
import gzip
import hashlib
import json
from pathlib import Path

from prepare_balanced_calibration import select
from run_option_capture import digest


def norm(scene):
    return scene.split('data/scene_datasets/')[-1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);args=p.parse_args()
    cfg=json.loads(args.config.read_text());s=cfg['sampling'];source=Path(s['source'])
    with gzip.open(str(source),'rt') as f:data=json.load(f)
    excluded=set();excluded_counts={};source_hashes={}
    for path in map(Path,s['exclude_sources']):
        local=set()
        if path.suffix=='.jsonl':
            with path.open() as f:
                for line in f:
                    row=json.loads(line);local.add(norm(row['scene_id']))
        else:
            obj=json.loads(path.read_text())
            records=obj.get('episodes', obj.get('route_episode_ids', []))
            if isinstance(records, int):
                # Some historical manifests keep only a route count; their
                # sampled and explicitly excluded scenes are still enough to
                # enforce scene disjointness.
                local={norm(s) for s in obj.get('scenes', [])}
                local.update(norm(s) for s in obj.get('excluded_scenes', []))
            else:
                local={norm(e['scene_id']) for e in records}
        excluded |= local;excluded_counts[str(path)]=len(local);source_hashes[str(path)]=digest(path)
    available=[e for e in data['episodes'] if norm(e['scene_id']) not in excluded]
    grouped=collections.defaultdict(set)
    for e in available:grouped[norm(e['scene_id'])].add(str(e['trajectory_id']))
    scenes=sorted([scene for scene,routes in grouped.items() if len(routes)>=s['routes_per_scene']],
                  key=lambda scene:hashlib.sha256('{}|full-return|{}'.format(s['seed'],scene).encode()).hexdigest())[:s['scenes']]
    if len(scenes)!=s['scenes']:raise ValueError('Insufficient fresh scenes')
    rows=select([e for e in available if norm(e['scene_id']) in set(scenes)],s['routes_per_scene'],s['seed'])
    routes={(norm(e['scene_id']),str(e['trajectory_id'])) for e in rows}
    if len(rows)!=cfg['episodes'] or len(routes)!=len(rows):raise ValueError('Invalid sample')
    validation={}
    for split in ['val_seen','val_unseen']:
        path=Path('data/datasets/R2R_VLNCE_v1-2_preprocessed_BERTidx')/split/(split+'_bertidx.json.gz')
        with gzip.open(str(path),'rt') as f:records=json.load(f)['episodes']
        overlap=routes & {(norm(e['scene_id']),str(e['trajectory_id'])) for e in records}
        if overlap:raise ValueError('Validation route overlap')
        validation[split]={'route_overlap':0,'source_sha256':digest(path)}
    if any(not (Path('data/scene_datasets')/scene).is_file() for scene in scenes):raise ValueError('Missing scene')
    out=Path(cfg['dataset_path']);out.parent.mkdir(parents=True,exist_ok=False)
    subset=dict(data);subset['episodes']=rows
    with out.open('xb') as raw:
        with gzip.GzipFile(fileobj=raw,mode='wb',filename='',mtime=0) as gz:gz.write(json.dumps(subset,ensure_ascii=False).encode())
    for path in [source,args.config,Path(__file__),Path(cfg['continuation_protocol'])]:source_hashes[str(path)]=digest(path)
    manifest={'experiment_id':cfg['experiment_id'],'split':'train','seed':s['seed'],'episodes':len(rows),
              'distinct_routes':len(routes),'scenes':sorted(scenes),'excluded_scenes':sorted(excluded),
              'excluded_source_scene_counts':excluded_counts,'prior_scene_overlap':0,
              'validation_disjointness':validation,'selection_rule':'Metadata-only scene/route selection; no outcomes, goals, reference route or branch execution',
              'dataset_path':str(out),'dataset_sha256':digest(out),'hashes':source_hashes,
              'route_episode_ids':[{'scene_id':e['scene_id'],'episode_id':str(e['episode_id']),'trajectory_id':str(e['trajectory_id'])} for e in rows]}
    with Path(cfg['sampling_manifest']).open('x') as f:json.dump(manifest,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in manifest.items() if k not in ['route_episode_ids','hashes']},indent=2))

if __name__=='__main__':main()
