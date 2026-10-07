#!/usr/bin/env python3
"""Versioned operational retry of replication 001 after a pre-episode fork failure.

Preserves the failed status, logs and run directory. Does not modify any frozen
experiment source, schedule, model or configuration. This is deliberately a
narrow continuation, not an automatic retry mechanism for failed outcome gates.
"""
import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    root = Path(cfg['replication']['root'])
    cycle = root / 'replication_cycle_001'
    state_path = cycle / 'status.json'
    old_bytes = state_path.read_bytes()
    state = json.loads(old_bytes.decode())
    if state['status'] != 'failed' or state['stages'][-1]['name'] != 'random_20261031_disabled':
        raise ValueError('Not the declared startup failure')
    error_path = cycle / 'random_20261031_disabled.stderr.log'
    if 'BrokenPipeError' not in error_path.read_text():
        raise ValueError('Unexpected failure')
    failed_run = cycle / 'random_20261031_disabled'
    if list((failed_run / 'traces').glob('*.jsonl')):
        raise ValueError('Unexpected episode traces in failed startup')
    decisions = failed_run / 'hook/decisions.jsonl'
    if decisions.exists() and decisions.stat().st_size:
        raise ValueError('Failed attempt contains decisions')
    reg = json.loads((root / 'registration_001.json').read_text())
    for path, sha in reg['hashes'].items():
        if digest(path) != sha:
            raise ValueError('Registered file changed: ' + path)
    if subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip() != state['git_commit']:
        raise ValueError('Execution commit changed')
    for arm in ['full', 'cost_own', 'cost_matched']:
        for suffix in ['disabled_audit', 'enabled_audit', 'enabled_metrics']:
            if not json.loads((cycle / (arm + '_' + suffix) / 'summary.json').read_text())['all_required_gates_passed']:
                raise ValueError('Prior completed arm failed fidelity')
    args.output.mkdir(exist_ok=False)
    with (args.output / 'original_failed_status.json').open('xb') as stream:
        stream.write(old_bytes)
    amendment = {
        'registered_utc': datetime.now(timezone.utc).isoformat(),
        'reason': 'BrokenPipeError starting Habitat forkserver worker before any episode decision.',
        'changes': 'Retry random_20261031 disabled under a new run/experiment ID, then complete remaining unchanged arms.',
        'no_scientific_protocol_change': True,
        'original_status_sha256': hashlib.sha256(old_bytes).hexdigest(),
        'failure_stderr_sha256': digest(error_path),
        'continuation_source_sha256': digest(__file__),
        'registration_sha256': digest(root / 'registration_001.json'),
        'original_failed_run': str(failed_run),
    }
    with (args.output / 'amendment.json').open('x') as stream:
        json.dump(amendment, stream, indent=2)
        stream.write('\n')
    state.update(status='running', continuation=str(args.output), original_failure_retained=True)

    def save():
        value = json.dumps(state, indent=2) + '\n'
        (args.output / 'status.json').write_text(value)
        state_path.write_text(value)

    def stage(name, script, options):
        command = [sys.executable, 'research/tools/' + script] + [str(x) for x in options]
        entry = {'name': name, 'command': command, 'started_utc': datetime.now(timezone.utc).isoformat(), 'continuation': str(args.output)}
        state['stages'].append(entry)
        save()
        print('START ' + name, flush=True)
        with (args.output / (name + '.stdout.log')).open('xb') as out, (args.output / (name + '.stderr.log')).open('xb') as err:
            code = subprocess.call(command, stdout=out, stderr=err)
        entry.update(returncode=code, finished_utc=datetime.now(timezone.utc).isoformat())
        if code:
            state['status'] = 'failed'
        save()
        if code:
            raise RuntimeError(name + ' failed; attempt retained')
        print('PASS ' + name, flush=True)

    for arm in ['random_20261031', 'random_20261032', 'random_20261033']:
        schedule = root / 'schedules_001' / (arm + '.json')
        if not json.loads(schedule.read_text())['events']:
            raise ValueError('Unexpected empty registered random arm')
        for mode in ['disabled', 'enabled']:
            name = arm + '_' + mode
            run_name = name + '_retry001' if name == 'random_20261031_disabled' else name
            run = cycle / run_name
            options = ['--schedule', schedule, '--source-capture', root / 'capture_001', '--output-dir', run,
                       '--exp-name', 'aaa_gv_prospective001_' + run_name]
            if mode == 'enabled':
                options += ['--enabled', '--control-gate', cycle / (arm + '_disabled_audit') / 'summary.json']
            stage(run_name, 'run_single_intervention.py', options)
            stage(name + '_audit', 'audit_single_intervention.py', ['--run-dir', run,
                  '--source-noninterference', root / 'noninterference_001', '--source-probe', root / 'probe_full001',
                  '--output-dir', cycle / (name + '_audit')])
            if mode == 'enabled':
                stage(name + '_metrics', 'verify_intervention_metrics.py', ['--run-dir', run, '--output-dir', cycle / (name + '_metrics')])
    state['all_arms_complete'] = True
    save()
    stage('analysis', 'analyze_preference_replication.py', ['--config', args.config, '--output', root / 'paired_analysis_001'])
    state['status'] = 'complete'
    save()


if __name__ == '__main__':
    main()
