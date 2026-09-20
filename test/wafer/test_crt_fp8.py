"""Run the production C decoder with host-only SPM address translation."""
import ctypes
import math
from pathlib import Path
import shutil
import struct
import subprocess

import pytest


def test_e5m2_to_fp16_all_encodings(tmp_path):
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("A host C compiler is required for the CRT decoder test")
    source = Path(__file__).resolve().parents[2] / "third_party/wafer/crt/lib/Wafer/mxfp_fp16.c"
    text = source.read_text()
    start = text.index("void __FP8E5M2_FP16(")
    function = text[start:text.index("\n/**", start)]
    stub = "#include <stdint.h>\nstatic uint64_t get_spm_memory_mapping_wrapper(uint64_t p) { return p; }\n"
    host_source = tmp_path / "decoder.c"
    host_source.write_text(stub + function)
    library = tmp_path / "decoder.so"
    subprocess.run([compiler, "-O2", "-shared", "-fPIC", str(host_source), "-o", str(library)], check=True)
    decode = ctypes.CDLL(str(library)).__FP8E5M2_FP16
    decode.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_uint16), ctypes.c_uint32]
    decode.restype = None
    src = (ctypes.c_uint8 * 256)(*range(256))
    # Sentinels check that elem_count bounds the destination writes.
    dst = (ctypes.c_uint16 * 258)(*([0x1234] * 258))
    decode(src, dst, 256)
    assert list(dst)[256:] == [0x1234, 0x1234]
    for code, bits in enumerate(list(dst)[:256]):
        sign, exponent, fraction = code >> 7, (code >> 2) & 31, code & 3
        actual = struct.unpack("<e", struct.pack("<H", bits))[0]
        assert bits >> 15 == sign, hex(code)
        if exponent == 31:
            assert math.isnan(actual) if fraction else math.isinf(actual)
            assert bits & 1023 == fraction * 256, hex(code)
        else:
            magnitude = math.ldexp(fraction / 4, -14) if exponent == 0 else math.ldexp(1 + fraction / 4, exponent - 15)
            expected = -magnitude if sign else magnitude
            expected_bits = struct.unpack("<H", struct.pack("<e", expected))[0]
            assert bits == expected_bits, f"code={code:#04x}: got {bits:#06x}, expected {expected_bits:#06x}"
    before_empty = list(dst)
    decode(src, dst, 0)
    assert list(dst) == before_empty
