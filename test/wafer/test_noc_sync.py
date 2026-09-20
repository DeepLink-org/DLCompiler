"""Stress the production request/ack protocol on a host SPM model."""
from pathlib import Path
import shutil
import subprocess

import pytest


def test_noc_initializer_only_clears_protocol_words(tmp_path):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("A C++ compiler is required for the initializer test")
    path = Path(__file__).resolve().parents[2] / "third_party/wafer/crt/lib/Wafer/noc_init.c"
    source = path.read_text()
    program = r'''
#include <cstdint>
#include <cassert>
#define SINGLE_SPM_SYNC_ADDR 0x2f0320
static uint32_t spm[22];
static uintptr_t get_spm_memory_mapping(uintptr_t address) {
  assert(address == SINGLE_SPM_SYNC_ADDR);
  return (uintptr_t)&spm[1];
}
'''
    program += source[source.index("void __NoCRingInit"):]
    program += r'''
int main() {
  for (auto &word : spm) word = 0xdeadbeef;
  __NoCRingInit();
  __NoCRingInit();
  for (int i = 0; i < 22; ++i)
    assert(spm[i] == ((i == 1 || i == 2) ? 0 : 0xdeadbeef));
}
'''
    src, exe = tmp_path / "init.cpp", tmp_path / "init"
    src.write_text(program)
    subprocess.run([compiler, "-std=c++17", "-O2", str(src), "-o", str(exe)], check=True)
    subprocess.run([str(exe)], check=True, timeout=10)


@pytest.mark.parametrize("tiles", [3, 16])
def test_noc_sync_repeated_rounds_and_launches(tmp_path, tiles):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("A C++ compiler is required for the ring protocol test")
    path = Path(__file__).resolve().parents[2] / "third_party/wafer/crt/lib/Wafer/send.c"
    source = path.read_text()
    protocol = source[source.index("static void noc_memory_fence"):source.index("// Send to the next tile")]
    # MMIO accesses become atomic host-memory accesses, avoiding C++ data races.
    protocol = protocol.replace("volatile uint32_t", "std::atomic<uint32_t>")
    program = r'''
#include <atomic>
#include <cstdint>
#include <thread>
#include <vector>
#include <chrono>
#include <cassert>
#define SINGLE_SPM_SYNC_ADDR 0
static std::atomic<uint32_t> spm[16][2]{};
static thread_local int tile;
static uintptr_t get_spm_memory_mapping(uintptr_t) { return (uintptr_t)spm[tile]; }
static uintptr_t get_tile_spm_addr_base(int i, int, int) { return (uintptr_t)spm[i]; }
'''
    program += protocol
    program += f'''
int main() {{
  const int count = {tiles};
  for (int launch = 0; launch < 2; ++launch) {{
    std::vector<std::thread> workers;
    for (int i = 0; i < count; ++i) workers.emplace_back([i, count]() {{
      tile = i;
      for (int round = 0; round < 1000; ++round) {{
        if (i == 1 && round % 7 == 0)
          std::this_thread::sleep_for(std::chrono::microseconds(1));
        noc_ring_sync((i + count - 1) % count, (i + 1) % count);
      }}
    }});
    for (auto &worker : workers) worker.join();
    for (int i = 0; i < count; ++i) {{
      assert(spm[i][0] == 0);
      assert(spm[i][1] == 0);
    }}
  }}
}}
'''
    src, exe = tmp_path / "sync.cpp", tmp_path / "sync"
    src.write_text(program)
    subprocess.run([compiler, "-std=c++17", "-O2", "-pthread", str(src), "-o", str(exe)], check=True)
    subprocess.run([str(exe)], check=True, timeout=20)
