#!/usr/bin/env python3
"""Run one exact disabled/native or enabled/alternative continuation pair."""
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(str(ROOT))


def digest(path):
    import hashlib
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--schedule', type=Path, required=True)
    p.add_argument('--source-capture', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--exp-name', required=True)
    p.add_argument('--enabled', action='store_true')
    args = p.parse_args()
    source_manifest_path = args.source_capture / 'manifest.json'
    source = json.loads(source_manifest_path.read_text())
    schedule = json.loads(args.schedule.read_text())
    config = source['config']
    if config['split'] != 'val_unseen' or config['episodes'] <= 0:
        raise ValueError('This runner is frozen for the val-unseen source capture')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    overrides = dict(source['overrides'])
    overrides['TRAINER_NAME'] = 'SS-ETP-NativeFullReturn'
    if overrides.get('ENV_NAME') != 'VLNCEOptionTraceEnv':
        raise ValueError('Source capture is not an OptionTrace environment')
    os.environ['ETPNAV_OPTION_TRACE_DIR'] = str((args.output_dir / 'traces').resolve())
    os.environ['ETPNAV_NATIVE_FULL_RETURN_MANIFEST'] = str((args.output_dir / 'manifest.json').resolve())
    os.environ.pop('ETPNAV_GRAPH_OPTION_DIR', None)
    manifest = {'created_utc': datetime.now(timezone.utc).isoformat(),
                'schedule': schedule, 'schedule_path': str(args.schedule),
                'source_capture': str(args.source_capture), 'config': config,
                'enabled': args.enabled, 'output_dir': str(args.output_dir.resolve()),
                'exp_name': args.exp_name, 'overrides': overrides,
                'source_manifest_sha256': digest(source_manifest_path),
                'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip(),
                'git_status': subprocess.check_output(['git', 'status', '--porcelain']).decode(),
                'python': sys.version, 'argv': sys.argv}
    (args.output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    from run import run_exp
    from vlnce_baselines.adaptive_action.option_calibration import OptionTraceEnv
    from vlnce_baselines.adaptive_action.native_full_return import NativeFullReturnTrainer
    run_exp(args.exp_name, config['base_config'], 'eval',
            opts=[x for pair in overrides.items() for x in pair], local_rank=0)


if __name__ == '__main__':
    main()
