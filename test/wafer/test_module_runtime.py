"""SDK module ownership, native dispatch and GIL behavior without a device."""
import ctypes
import os
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


def test_module_owner_cleanup(wafer_modules, monkeypatch):
    _, _, runtime = wafer_modules
    state = {"device": 3}
    def load(pointer, binary, size):
        assert state["device"] == 0 and size == 3
        ctypes.cast(pointer, ctypes.POINTER(ctypes.c_void_p)).contents.value = 123
        return 0
    def get_function(pointer, module, name):
        assert module.value == 123 and name == b"kernel"
        ctypes.cast(pointer, ctypes.POINTER(ctypes.c_void_p)).contents.value = 456
        return 0
    def unload(module):
        assert state["device"] == 0 and module.value == 123
        return 0
    library = SimpleNamespace(txModuleLoad=Mock(side_effect=load),
                              txModuleGetFunction=Mock(side_effect=get_function),
                              txModuleUnload=Mock(side_effect=unload))
    sdk = SimpleNamespace(library=library, current_device=lambda: state["device"],
                          set_device=lambda device: state.update(device=device))
    monkeypatch.setattr(runtime, "_KuiperRuntime", lambda: sdk)
    owner = runtime._LoadedModule("kernel", b"ELF", 0)
    assert owner.function == 456 and state["device"] == 3
    owner.close()
    owner.close()
    assert library.txModuleUnload.call_count == 1 and state["device"] == 3
    library.txModuleGetFunction.side_effect = lambda *args: 0x42
    with pytest.raises(RuntimeError, match="txModuleGetFunction.*42"):
        runtime._LoadedModule("kernel", b"ELF", 0)
    assert library.txModuleUnload.call_count == 2 and state["device"] == 3


@pytest.mark.parametrize("mode", ["simt", "cluster"])
def test_native_module_dispatch_releases_gil(mode, wafer_modules, tmp_path, monkeypatch):
    if not os.getenv("KUIPER_ROOT"):
        pytest.skip("Kuiper headers required")
    _, _, runtime = wafer_modules
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "cache"))
    stubs = r'''
static txError_t test_module(txFunction_t function, dim3 grid, dim3 block,
    void *args, uint32_t length, uint32_t shared, txStream_t stream) {
  if (PyGILState_Check() || function != (txFunction_t)456 || grid.x != 4 ||
      length != 7 * sizeof(uint64_t) || ((uint64_t*)args)[0] != 42)
    return (txError_t)0x76;
  return (txError_t)0x75;
}
static txError_t test_cluster_module(txFunction_t function, dim3 cluster,
    dim3 grid, dim3 block, void *args, uint32_t length, uint32_t shared, txStream_t stream) {
  if (cluster.x != 1 || cluster.y != 1 || cluster.z != 1) return (txError_t)0x77;
  return test_module(function, grid, block, args, length, shared, stream);
}
#define txLaunchKernel test_module
#define txLaunchClusterKernel test_cluster_module
'''
    source = runtime.make_launcher({0: "i32"}, mode).replace(
        '#include "tx_runtime.h"', '#include "tx_runtime.h"\n' + stubs)
    launch = runtime.compile_launcher(source).launch
    metadata = SimpleNamespace(kernel_path=str(tmp_path / "not-read.so"), name="test")
    with pytest.raises(RuntimeError, match="0x75"):
        launch(4, 1, 1, None, 456, metadata, None, None, None, 42)
