"""The peri receives addresses, never dereferenced seed contents."""
from pathlib import Path
import shutil
import subprocess

import pytest


def test_randgen_argument_addresses(tmp_path):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler required")
    root = Path(__file__).resolve().parents[2]
    implementation = (root / "third_party/wafer/crt/lib/Wafer/randgen.c").read_text().replace('#include "wafer.h"', '')
    source = r'''
#include <cstdint>
#include <cassert>
#define INTRNISIC_RUN_SWITCH
#define SYNCHRONOUS_INTRINSIC_SWITCH
using Data_Format = int;
constexpr int I_CGRA = 0;
struct TsmPeripheralInstr { int tag; int a[1]; int b[1]; };
struct TsmPeripheral {
  void RandGen(TsmPeripheralInstr*, uint64_t a, uint64_t b, uint64_t c,
               uint64_t d, uint64_t e, uint32_t bytes, Data_Format fmt) {
    assert(a == 0x1000 && b == 0x2000 && c == 0x3000 && d == 0x4000 && e == 0x5000);
    assert(bytes == 256 && fmt == 11);
  }
};
struct Intrinsic { TsmPeripheral* peripheral_pointer; };
Intrinsic* g_intrinsic() { static TsmPeripheral p; static Intrinsic i{&p}; return &i; }
void TsmExecute(TsmPeripheralInstr*) {}
'''
    path = tmp_path / "rand.cpp"
    path.write_text(source + implementation + '''
int main() { __RandGen((uint64_t*)0x1000, (uint64_t*)0x2000, (uint64_t*)0x3000,
                      (uint64_t*)0x4000, (uint64_t*)0x5000, 256, 11); }
''')
    exe = tmp_path / "rand"
    subprocess.run([compiler, "-std=c++17", str(path), "-o", str(exe)], check=True)
    subprocess.run([str(exe)], check=True, timeout=10)
