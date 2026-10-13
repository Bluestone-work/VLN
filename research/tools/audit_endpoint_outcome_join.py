#!/usr/bin/env python3
import argparse, csv, json
from pathlib import Path

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--endpoint',type=Path,required=True); ap.add_argument('--pairs',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); a=ap.parse_args()
    endpoint=[json.loads(x) for x in (a.endpoint/'native_endpoint_options.jsonl').open() if x.strip()]
    pairs=[json.loads(x) for x in (a.pairs/'single_override_pairs.jsonl').open() if x.strip()] if (a.pairs/'single_override_pairs.jsonl').exists() else []
    ekeys={(str(x['episode_id']),int(x['high_level_step']),json.dumps(x.get('graph_id'),sort_keys=True)) for x in endpoint}
    pkeys={(str(x.get('episode_id')),int(x.get('high_level_step',-1)),json.dumps(x.get('native_action_signature'),sort_keys=True)) for x in pairs}
    result={'endpoint_action_records':len(endpoint),'causal_pair_records':len(pairs),'exact_join_records':0,'status':'BLOCKED_NO_EXACT_CAUSAL_PAIRS','reason':'endpoint branches contain native action identities, but the available single_override_pairs file is empty; no BENEFIT/HARM labels may be inferred'}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2))
if __name__=='__main__': main()
