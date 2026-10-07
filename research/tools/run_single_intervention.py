#!/usr/bin/env python3
"""Execute the frozen one-action schedule with original evaluation settings."""
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
from run_option_capture import digest


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--schedule', type=Path, required=True)
    p.add_argument('--source-capture', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--exp-name', required=True)
    p.add_argument('--enabled', action='store_true')
    p.add_argument('--control-gate', type=Path)
    args = p.parse_args()
    if (Path('data/logs/eval_results') / args.exp_name).exists():
        raise ValueError('Native results already exist')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    schedule = json.loads(args.schedule.read_text())
    source_path = args.source_capture / 'manifest.json'
    source = json.loads(source_path.read_text())
    config = json.loads(Path(schedule['base_experiment_config']).read_text())
    if config != source['config'] or config['split'] != 'train' or config['episodes'] <= 0:
        raise ValueError('Frozen training population changed')
    for name, expected in list(schedule['hashes'].items()) + list(source['hashes'].items()):
        if digest(Path(name)) != expected:
            raise ValueError('Frozen source changed: ' + name)
    if args.enabled:
        if args.control_gate is None:
            raise ValueError('Disabled-interceptor gate required before enabling')
        gate = json.loads(args.control_gate.read_text())
        if not gate['all_required_gates_passed'] or gate['mode'] != 'disabled':
            raise ValueError('Disabled-interceptor gate failed')
        if gate['schedule_sha256'] != digest(args.schedule) or gate['source_capture_manifest_sha256'] != digest(source_path):
            raise ValueError('Control gate belongs to another protocol')
        for name, expected in gate['intervention_source_hashes'].items():
            if digest(Path(name)) != expected:
                raise ValueError('Intervention implementation changed after control')
    overrides = dict(source['overrides'])
    overrides['TRAINER_NAME'] = 'SS-ETP-SingleIntervention'
    # The source capture already uses the calibrated trace environment.
    if overrides['ENV_NAME'] != 'VLNCEOptionTraceEnv':
        raise ValueError('Source trace environment mismatch')
    os.environ['ETPNAV_OPTION_TRACE_DIR'] = str(args.output_dir.resolve() / 'traces')
    os.environ.pop('ETPNAV_GRAPH_OPTION_DIR', None)
    paths = [args.schedule, source_path, Path(__file__),
             Path('vlnce_baselines/adaptive_action/single_intervention.py'),
             Path('research/tools/run_option_capture.py')]
    hashes = dict(source['hashes'])
    hashes.update({str(path): digest(path) for path in paths})
    manifest = {'created_utc': datetime.now(timezone.utc).isoformat(), 'schedule': schedule,
                'schedule_path': str(args.schedule), 'source_capture': str(args.source_capture),
                'config': config, 'enabled': args.enabled, 'output_dir': str(args.output_dir.resolve()),
                'exp_name': args.exp_name, 'overrides': overrides, 'hashes': hashes,
                'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip(),
                'git_status': subprocess.check_output(['git', 'status', '--porcelain']).decode(),
                'hardware': subprocess.check_output(['nvidia-smi', '--query-gpu=name,uuid,driver_version', '--format=csv,noheader']).decode(),
                'python': sys.version, 'argv': sys.argv,
                'environment': {k: os.environ.get(k) for k in ['EGL_PLATFORM', 'CUDA_VISIBLE_DEVICES']},
                'control_gate': str(args.control_gate) if args.control_gate else None,
                'analysis_only_oracle': True}
    path = args.output_dir / 'manifest.json'
    path.write_text(json.dumps(manifest, indent=2) + '\n')
    os.environ['ETPNAV_SINGLE_MANIFEST'] = str(path.resolve())
    from run import run_exp
    from vlnce_baselines.adaptive_action.option_calibration import OptionTraceEnv
    from vlnce_baselines.adaptive_action.single_intervention import SingleInterventionTrainer
    run_exp(args.exp_name, config['base_config'], 'eval',
            opts=[x for pair in overrides.items() for x in pair], local_rank=0)


if __name__ == '__main__':
    main()
