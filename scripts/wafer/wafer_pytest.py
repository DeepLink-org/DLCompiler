"""Observe native Wafer tests through Triton's public hooks; never rewrite tensors."""
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time

import pytest

STATE = dict(nodeid=None, started=0, completed=0, blocked=False)
HOOKS = []


def record(event, **values):
    path = os.getenv('WAFER_TEST_EVENTS')
    if path:
        with open(path, 'a') as stream:
            stream.write(json.dumps(dict(event=event, time=time.time(), nodeid=STATE['nodeid'], **values)) + '\n')


def pytest_addoption(parser):
    parser.addoption('--wafer-hardware', action='store_true', help='Run native TXDA device tests')
    parser.addoption('--wafer-nodeids', help='Exact nodeids from the accepted suite; fail if any are absent')


def pytest_collection_modifyitems(config, items):
    selection = config.getoption('--wafer-nodeids')
    if selection:
        requested = set(Path(selection).read_text().splitlines())
        missing = requested - {item.nodeid for item in items}
        if missing:
            raise pytest.UsageError(f'Accepted cases missing from collection: {sorted(missing)}')
        excluded = [item for item in items if item.nodeid not in requested]
        items[:] = [item for item in items if item.nodeid in requested]
        config.hook.pytest_deselected(items=excluded)
        record('selection', excluded=[item.nodeid for item in excluded])
    record('collection', nodeids=[item.nodeid for item in items])


@pytest.fixture(scope='session')
def wafer_device(request):
    if not request.config.getoption('--wafer-hardware'):
        pytest.skip('requires --wafer-hardware')
    import torch
    import torch_txda  # noqa: F401
    import triton
    from triton import knobs
    for key, value in dict(DICP_BACKEND='wafer', USE_SIM_MODE='0', WAFER_ENABLE_RUNTIME='1').items():
        if os.getenv(key) != value:
            raise pytest.UsageError(f'requires {key}={value}')
    assert torch.txda.is_available()
    assert triton.runtime.driver.active.get_current_target().backend == 'wafer'
    assert triton.runtime.driver.active.get_active_torch_device().type == 'txda'

    def enter(metadata):
        STATE['started'] += 1
        record('launch_start', kernel=metadata.get()['name'])
        # A process killed by this timer leaves an unmatched launch_start.
        # The outer runner stops scheduling device work on that evidence.
        signal.setitimer(signal.ITIMER_REAL, float(os.getenv('WAFER_TEST_LAUNCH_TIMEOUT', '60')))

    def leave(metadata):
        signal.setitimer(signal.ITIMER_REAL, 0)
        STATE['completed'] += 1
        record('launch_complete', kernel=metadata.get()['name'])

    def loaded(module, function, name, group, kernel_hash):
        from audit_wafer_elf import audit_kernel
        for path in group.values():
            if str(path).endswith('.json'):
                data = json.loads(Path(path).read_text())
                if 'kernel_path' in data:
                    audit = audit_kernel(data['kernel_path'], data['device_log_abi'], os.getenv('WAFER_NOC_FIRMWARE_ELF'))
                    record('compiled', kernel=name, path=data['kernel_path'], audit=audit)
                    return
        raise RuntimeError(f'No Wafer ELF metadata found for {name}')

    for hook, fn in [(knobs.runtime.launch_enter_hook, enter), (knobs.runtime.launch_exit_hook, leave),
                     (knobs.runtime.kernel_load_end_hook, loaded)]:
        hook.add(fn)
        HOOKS.append((hook, fn))
    record('configured', target=str(triton.runtime.driver.active.get_current_target()),
           triton_path=triton.__file__, execution='native-txda')
    yield 'txda'
    for hook, fn in HOOKS:
        hook.remove(fn)
    HOOKS.clear()


@pytest.fixture
def device(wafer_device):
    return wafer_device


def pytest_runtest_logstart(nodeid, location):
    # This also runs before pytest's skip markers. A setup hook can be bypassed
    # by an early skip, leaving that result attributed to the previous case.
    if STATE['blocked']:
        pytest.exit('Device error: stop before another kernel launch', returncode=3)
    STATE.update(nodeid=nodeid, started=0, completed=0, since=time.time())
    record('test_start')


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if report.when == 'call' or report.failed or report.skipped:
        record('test_result', outcome=report.outcome, phase=report.when,
               launches=STATE['completed'], started=STATE['started'],
               detail=str(report.longrepr) if report.longrepr else None)
    if report.when == 'call' and STATE['started']:
        if STATE['started'] != STATE['completed']:
            STATE['blocked'] = True
        log = subprocess.run(['journalctl', '-k', '--since', f"@{STATE['since']:.6f}", '--no-pager', '-o', 'cat'],
                             capture_output=True, text=True, timeout=10)
        if log.returncode or re.search(r'XID|AP resetting|Clean Resource Timeout|NPU LSU.*Timeout', log.stdout):
            STATE['blocked'] = True
            record('device_error', detail=log.stdout + log.stderr)
        if STATE['blocked']:
            pytest.exit('Device error or incomplete launch: inspect event log', returncode=3)
