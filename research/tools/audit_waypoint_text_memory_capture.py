#!/usr/bin/env python3
import argparse, hashlib, json
from pathlib import Path
import numpy as np

EXPECTED = ('rgb','depth') + tuple('rgb_{}'.format(x) for x in range(30,360,30)) + tuple('depth_{}'.format(x) for x in range(30,360,30))

def digest(path): return hashlib.sha256(np.load(path, allow_pickle=False).tobytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--capture',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); a=ap.parse_args()
    rows=[json.loads(x) for x in (a.capture/'graph_options/graph_options.jsonl').open() if x.strip()]; missing=[]; bad=[]; count=0
    for row in rows:
        files=(row.get('semantic_visual') or {}).get('files',{})
        for sensor in EXPECTED:
            path=Path(files.get(sensor,'')); count+=1
            if not path.exists(): missing.append({'key':[row['episode_id'],row['high_level_step']],'sensor':sensor}); continue
            actual=digest(path); expected=(row['semantic_visual'].get('observation_hashes') or {}).get(sensor)
            if actual != expected: bad.append({'key':[row['episode_id'],row['high_level_step']],'sensor':sensor})
    result={'decision_rows':len(rows),'expected_sensor_records':count,'missing_sensor_records':len(missing),'hash_mismatches':len(bad),'views_per_state':len(EXPECTED),'status':'PASS' if not missing and not bad else 'FAIL','missing':missing[:10],'bad':bad[:10]}
    a.output.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2)); raise SystemExit(0 if result['status']=='PASS' else 1)
if __name__=='__main__': main()
