"""Check generated native launch dispatch and arguments without device work."""
import os
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("mode,marker", [("simt", "0x71"), ("cluster", "0x72")])
def test_native_launch_mode(mode, marker, wafer_modules, tmp_path, monkeypatch):
    if not os.getenv("KUIPER_ROOT"):
        pytest.skip("Native launcher test requires Kuiper headers")
    _, _, runtime = wafer_modules
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "cache"))
    stubs = r'''
static txError_t test_simt(const char *name, uint64_t elf, uint64_t len,
    dim3 grid, dim3 block, void *args, uint32_t arglen, uint32_t shared, txStream_t stream) {
  if (strcmp(name, "test") || len != 4 || !elf || grid.x != 16 || grid.y != 1 || grid.z != 1 ||
      block.x != 1 || block.y != 1 || block.z != 1 || shared != 0 || stream != nullptr ||
      arglen != 7 * sizeof(uint64_t) || ((uint64_t *)args)[0] != 42 || ((uint64_t *)args)[1] != 16)
    return (txError_t)0x73;
  return (txError_t)0x71;
}
static txError_t test_cluster(const char *name, uint64_t elf, uint64_t len, dim3 cluster,
    dim3 grid, dim3 block, void *args, uint32_t arglen, uint32_t shared, txStream_t stream) {
  if (cluster.x != 1 || cluster.y != 1 || cluster.z != 1) return (txError_t)0x74;
  txError_t result = test_simt(name, elf, len, grid, block, args, arglen, shared, stream);
  return result == (txError_t)0x71 ? (txError_t)0x72 : result;
}
#define txLaunchKernelGGL test_simt
#define txLaunchClusterKernelGGL test_cluster
'''
    source = runtime.make_launcher({0: "i32"}, mode)
    source = source.replace('#include "tx_runtime.h"', '#include "tx_runtime.h"\n' + stubs)
    launch = runtime.compile_launcher(source).launch
    binary = tmp_path / "dummy.so"
    binary.write_bytes(b"test")
    metadata = SimpleNamespace(kernel_path=str(binary), name="test")
    with pytest.raises(RuntimeError, match=marker):
        launch(16, 1, 1, None, 0, metadata, None, None, None, 42)
    if mode == "cluster":
        for grid in [(17, 1, 1), (8, 2, 1), (8, 1, 2)]:
            with pytest.raises(ValueError, match="cluster grid"):
                launch(*grid, None, 0, metadata, None, None, None, 42)


def test_cluster_option_validation_and_cache(wafer_modules):
    _, compiler, _ = wafer_modules
    assert compiler.WaferOptions().hash() != compiler.WaferOptions(launch_mode="cluster").hash()
    with pytest.raises(ValueError, match="launch_mode"):
        compiler.WaferOptions(launch_mode="invalid")
    with pytest.raises(ValueError, match="one cluster"):
        compiler.WaferOptions(launch_mode="cluster", cluster_dims=(2, 1, 1))
