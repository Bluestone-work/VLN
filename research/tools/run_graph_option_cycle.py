#!/usr/bin/env python3
"""Sequential research stages with explicit prerequisite checks and exclusive logs."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--exp-prefix', required=True)
    args = p.parse_args()
    provenance = args.root / 'cycle_001'
    provenance.mkdir(parents=True, exist_ok=False)
    cfg = json.loads(args.config.read_text())
    if not Path(cfg['dataset_path']).is_file() or not Path(cfg['sampling_manifest']).is_file():
        raise ValueError('Frozen sample is absent')
    (provenance / 'git_diff.patch').write_bytes(subprocess.check_output(['git', 'diff']))
    (provenance / 'git_status.txt').write_bytes(subprocess.check_output(['git', 'status', '--porcelain']))
    (provenance / 'pip_freeze.txt').write_bytes(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze']))
    files = [args.config, Path(cfg['sampling_manifest']), Path(cfg['route_protocol']), Path(__file__)]
    manifest = {'created_utc': now(), 'config': cfg, 'python': sys.version, 'argv': sys.argv,
                'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip(),
                'hardware': subprocess.check_output(['nvidia-smi', '--query-gpu=name,uuid,driver_version', '--format=csv,noheader']).decode(),
                'hashes': {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
                'stages': []}
    def save():
        (provenance / 'status.json').write_text(json.dumps(manifest, indent=2) + '\n')
    def stage(name, script, arguments, gate=None):
        if gate:
            path, flag = gate
            if not json.loads(path.read_text())[flag]:
                raise ValueError('Failed prerequisite gate: ' + flag)
        command = [sys.executable, 'research/tools/' + script] + [str(a) for a in arguments]
        entry = {'name': name, 'command': command, 'started_utc': now(), 'status': 'running'}
        manifest['stages'].append(entry)
        save()
        print('START ' + name, flush=True)
        with (provenance / (name + '.stdout.log')).open('xb') as out, (provenance / (name + '.stderr.log')).open('xb') as err:
            result = subprocess.call(command, stdout=out, stderr=err)
        entry.update(returncode=result, finished_utc=now(), status='passed' if result == 0 else 'failed')
        save()
        if result:
            raise RuntimeError('Stage {} exited {}'.format(name, result))
        print('PASS ' + name, flush=True)
    capture, control = args.root / 'capture_001', args.root / 'control_001'
    noninterference, probe = args.root / 'noninterference_001', args.root / 'probe_full001'
    for name, directory, trace in [('capture', capture, True), ('control', control, False)]:
        exp_name = args.exp_prefix + '_' + name
        if (Path('data/logs/eval_results') / exp_name).exists():
            raise ValueError('Native results already exist: ' + exp_name)
        options = ['--config', args.config, '--output-dir', directory, '--exp-name', exp_name]
        if trace:
            options.append('--trace')
        stage(name, 'run_option_capture.py', options)
    stage('noninterference', 'check_option_noninterference.py', ['--capture-dir', capture, '--control-dir', control, '--output-dir', noninterference])
    stage('probe', 'probe_graph_options.py', ['--capture-dir', capture, '--noninterference-dir', noninterference, '--output-dir', probe],
          (noninterference / 'summary.json', 'baseline_noninterference_gate_passed'))
    stage('integrity', 'audit_graph_option_probe.py', ['--probe-dir', probe, '--output-dir', args.root / 'integrity_001'],
          (probe / 'summary.json', 'all_required_gates_passed'))
    stage('analysis', 'analyze_graph_options.py', ['--probe-dir', probe, '--output-dir', args.root / 'analysis_001'],
          (args.root / 'integrity_001/summary.json', 'integrity_gate_passed'))
    manifest.update(status='complete', finished_utc=now())
    save()
    print('COMPLETE ' + cfg['experiment_id'], flush=True)


if __name__ == '__main__':
    main()
