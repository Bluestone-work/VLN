#!/usr/bin/env python3
"""Run the normal frozen baseline with optional full graph-option tracing."""
import argparse
import hashlib
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
    value = hashlib.sha256()
    with open(str(path), 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--exp-name', required=True)
    parser.add_argument('--trace', action='store_true')
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=False)
    if args.trace:
        os.environ['ETPNAV_OPTION_TRACE_DIR'] = str(args.output_dir.resolve() / 'traces')
        if cfg.get('capture_waypoint_provenance', False):
            os.environ['ETPNAV_DENSE_CANDIDATE_CAPTURE'] = '1'
    else:
        os.environ.pop('ETPNAV_OPTION_TRACE_DIR', None)
    from run import run_exp
    # Research-only registration: default run.py imports and baseline class stay unchanged.
    from vlnce_baselines.adaptive_action.option_calibration import OptionTraceEnv
    from vlnce_baselines.adaptive_action.graph_option_capture import GraphOptionTraceTrainer
    graph_capture = bool(cfg.get('capture_graph_options', False) and args.trace)
    if graph_capture:
        os.environ['ETPNAV_GRAPH_OPTION_DIR'] = str(args.output_dir.resolve() / 'graph_options')
    else:
        os.environ.pop('ETPNAV_GRAPH_OPTION_DIR', None)
        os.environ.pop('ETPNAV_DENSE_CANDIDATE_CAPTURE', None)
    overrides = {
        'TRAINER_NAME': 'SS-ETP-OptionCapture' if graph_capture else 'SS-ETP',
        'ENV_NAME': 'VLNCEOptionTraceEnv' if args.trace else 'VLNCEDaggerEnv',
        'SIMULATOR_GPU_IDS': '[0]', 'TORCH_GPU_IDS': '[0]', 'GPU_NUMBERS': '1',
        'NUM_ENVIRONMENTS': '1', 'EVAL.SPLIT': cfg['split'],
        'EVAL.EPISODE_COUNT': str(cfg['episodes']), 'TASK_CONFIG.SEED': str(cfg['seed']),
        'EVAL.CKPT_PATH_DIR': cfg['checkpoint'], 'IL.back_algo': cfg['back_algo'],
        'RL_TOPO.ENABLED': 'False',
        'ACTION_ABSTRACTION.LEVEL': cfg.get('action_abstraction', 'default'),
        'ACTION_ABSTRACTION.DIAGNOSTICS_ENABLED': 'False',
        'ACTION_ABSTRACTION.ORACLE_ENABLED': 'False',
        'ACTION_ABSTRACTION.GRAPH_SELECTION_ORACLE_ENABLED': 'False',
        'ACTION_ABSTRACTION.INCLUDE_RANKER_EMBEDDINGS': 'False',
        'TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING': str(cfg['allow_sliding'])}
    if cfg.get('dataset_path'):
        overrides['TASK_CONFIG.DATASET.DATA_PATH'] = cfg['dataset_path']
    source_paths = [args.config, Path(__file__), Path('vlnce_baselines/adaptive_action/option_calibration.py'),
                    Path('vlnce_baselines/common/environments.py'), Path('vlnce_baselines/ss_trainer_ETP.py'),
                    Path(cfg['base_config']), Path(cfg['checkpoint'])]
    source_paths.append(Path('vlnce_baselines/adaptive_action/graph_option_capture.py'))
    for key in ['dataset_path', 'sampling_manifest']:
        if cfg.get(key):
            source_paths.append(Path(cfg[key]))
    manifest = {'created_utc': datetime.now(timezone.utc).isoformat(), 'config': cfg,
                'exp_name': args.exp_name, 'overrides': overrides, 'trace': args.trace,
                'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip(),
                'git_status': subprocess.check_output(['git', 'status', '--porcelain']).decode(),
                'hashes': {str(p): digest(p) for p in source_paths}, 'python': sys.version,
                'hardware': subprocess.check_output(['nvidia-smi', '--query-gpu=name,uuid,driver_version', '--format=csv,noheader']).decode(),
                'command_argv': sys.argv, 'environment': {k: os.environ.get(k) for k in ['EGL_PLATFORM', 'CUDA_VISIBLE_DEVICES']}}
    (args.output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    opts = [item for pair in overrides.items() for item in pair]
    run_exp(exp_name=args.exp_name, exp_config=cfg['base_config'], run_type='eval', opts=opts, local_rank=0)


if __name__ == '__main__':
    main()
