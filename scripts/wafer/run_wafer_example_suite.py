#!/usr/bin/env python3
"""Run native TXDA tests in isolated processes, keeping exact nodeids and launch evidence."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
EXAMPLES = REPO / 'third_party/wafer/examples'
MANIFEST = REPO / 'test/wafer/suites/accepted.txt'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--suite', choices=('accepted', 'examples', 'ops', 'runtime', 'native_math', 'host'), default='accepted')
    parser.add_argument('--select', nargs='+', help='Paths relative to the suite root (repository root for accepted)')
    parser.add_argument('--nodeids-file', type=Path,
                        help='With --suite accepted, use an explicit nodeid list (e.g. remaining parameters)')
    parser.add_argument('--timeout', type=int, default=1800, help='Per-file timeout; any timeout stops device scheduling')
    args = parser.parse_args()
    if args.nodeids_file and args.suite != 'accepted':
        parser.error('--nodeids-file requires --suite accepted')
    manifest = args.nodeids_file.resolve() if args.nodeids_file else MANIFEST
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    # Never merge stale evidence from another source state into a fresh run.
    if (output / 'summary.json').exists():
        parser.error('Output already contains a run; choose a new directory')
    accepted = defaultdict(list)
    for node in manifest.read_text().splitlines():
        accepted[node.split('::', 1)[0]].append(node)
    root = {'accepted': REPO, 'examples': EXAMPLES, 'host': REPO / 'test/wafer'}.get(args.suite,
            REPO / 'test/wafer' / args.suite)
    if args.suite == 'accepted':
        files = [REPO / name for name in accepted]
    else:
        files = sorted(root.glob('test_*.py') if args.suite == 'host' else root.rglob('test_*.py'))
        files = [p for p in files if p.name != 'test_common.py']
    if args.select:
        requested = set(args.select)
        found = {str(p.relative_to(root)) for p in files}
        if requested - found:
            parser.error(f'Unknown files: {sorted(requested - found)}')
        files = [p for p in files if str(p.relative_to(root)) in requested]
    files.sort(key=lambda p: (2 if '/tle/' in str(p) else 1 if p.name == 'test_dot_scaled.py' else 0, str(p)))
    environment = os.environ.copy()
    environment['PYTHONPATH'] = os.pathsep.join(filter(None, (str(REPO / 'scripts/wafer'), environment.get('PYTHONPATH'))))
    if args.suite == 'host':
        # Host link tests import the source backend but link the delivered CRT.
        from triton.backends.dicp_triton import wafer
        environment.setdefault('WAFER_RUNTIME_LIB_DIR', str(Path(wafer.__file__).parent / 'lib'))
    if args.suite != 'host':
        for key, expected in dict(DICP_BACKEND='wafer', USE_SIM_MODE='0', WAFER_ENABLE_RUNTIME='1').items():
            if environment.get(key) != expected:
                parser.error(f'Activate the Wafer environment first: requires {key}={expected}')
    try:
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    except subprocess.CalledProcessError:
        revision = None
    summary = dict(suite=args.suite, revision=revision, python=sys.executable,
                   selection=str(manifest), selection_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                   files=[], blocked_reason=None)
    for index, source in enumerate(files, 1):
        relative = str(source.relative_to(REPO))
        directory = output / relative.removesuffix('.py')
        directory.mkdir(parents=True, exist_ok=True)
        record = dict(file=relative, source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
        if summary['blocked_reason']:
            record.update(status='not_run_after_device_error', reason=summary['blocked_reason'])
        else:
            events_path = directory / 'events.jsonl'
            events_path.write_text('')
            environment['WAFER_TEST_EVENTS'] = str(events_path)
            # These two accepted files have mode-2 evidence, not mode-0 evidence.
            precision = 2 if source.name in ('test_mod.py', 'test_device_print.py') else 0
            environment['PRECISION_MODE'] = str(precision)
            command = [sys.executable, '-m', 'pytest', '-q', '-p', 'wafer_pytest', str(source), '--tb=short', '-ra',
                       f'--junitxml={directory / "junit.xml"}']
            if args.suite != 'host':
                command.append('--wafer-hardware')
            if args.suite == 'accepted':
                selection = directory / 'nodeids.txt'
                selection.write_text('\n'.join(accepted[relative]) + '\n')
                command.append(f'--wafer-nodeids={selection}')
            print(f'[{index}/{len(files)}] {relative}', flush=True)
            started = time.monotonic()
            timed_out = False
            with (directory / 'pytest.log').open('w') as log:
                process = subprocess.Popen(command, cwd=REPO.parent, env=environment,
                                           stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    code = process.wait(timeout=args.timeout)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    code = process.returncode
            events = [json.loads(line) for line in events_path.read_text().splitlines() if line.strip()]
            tests = [e for e in events if e['event'] == 'test_result']
            counts = Counter(e['outcome'] for e in tests)
            launches = Counter(e['event'] for e in events)
            collected = next((e['nodeids'] for e in events if e['event'] == 'collection'), [])
            completed = {e['nodeid'] for e in tests}
            missing = sorted(set(collected) - completed)
            record.update(status='passed' if code == 0 and not missing else 'failed', returncode=code,
                          seconds=round(time.monotonic() - started, 3), precision_mode=precision,
                          collected=len(collected), outcomes=dict(counts), completed_launches=launches['launch_complete'],
                          passed_without_launch=[e['nodeid'] for e in tests if e['outcome'] == 'passed' and not e['launches']],
                          not_completed=missing, command=command, log=str(directory / 'pytest.log'))
            if timed_out or code < 0 or code == 3 or launches['device_error'] or launches['launch_start'] != launches['launch_complete']:
                summary['blocked_reason'] = f'Incomplete launch, device error or process timeout/crash in {relative}; inspect its log before continuing.'
            print(f"  {record['status']}: {dict(counts)}, launches={launches['launch_complete']}", flush=True)
        summary['files'].append(record)
        (directory / 'result.json').write_text(json.dumps(record, indent=2) + '\n')
        (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return int(any(row['status'] != 'passed' for row in summary['files']))


if __name__ == '__main__':
    raise SystemExit(main())
