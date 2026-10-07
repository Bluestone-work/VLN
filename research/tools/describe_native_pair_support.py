#!/usr/bin/env python3
"""Post hoc class-balance description of already audited native-relative pairs."""
import argparse
import csv
import json
from pathlib import Path
from audit_preference_training_support import digest


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,required=True); args=ap.parse_args()
    pairs_path=args.root/'analysis_001/pairs.csv'; states_path=args.root/'analysis_001/states.json'
    states={(r['episode_id'],r['step']):r for r in json.loads(states_path.read_text())}
    groups={name:[] for name in ['native_better','alternative_better']}
    with pairs_path.open(newline='') as stream:
        for row in csv.DictReader(stream):
            if row['kind']!='native_move' or row['relation']!='strict':continue
            native=states[(row['episode_id'],int(row['step']))]['native_index']
            name='native_better' if int(row['winner_index'])==native else 'alternative_better'
            groups[name].append(row)
    summary={}
    for name,rows in groups.items():
        summary[name]={'pairs':len(rows),'states':len({(r['episode_id'],r['step']) for r in rows}),
                       'routes':len({r['episode_id'] for r in rows}), 'scenes':len({r['scene_id'] for r in rows}),
                       'frozen_correct':sum(r['frozen_correct']=='True' for r in rows),
                       'logit_correct':sum(r['logit_correct']=='True' for r in rows)}
    result={'analysis':'Post hoc directional split of registered native-relative strict pairs; no new fitting or gate.',
            'groups':summary,'limits':'In-sample and state-correlated. A strict improvement need not rescue success.',
            'hashes':{str(p):digest(p) for p in [Path(__file__),pairs_path,states_path]}}
    output=args.root/'directional_support_001'; output.mkdir(exist_ok=False)
    with (output/'summary.json').open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    with (output/'native_improvement_pairs.json').open('x') as stream:json.dump(groups['alternative_better'],stream,indent=2);stream.write('\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
