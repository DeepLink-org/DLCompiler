#!/usr/bin/env python3
"""Audit a DLCompiler runtime-linked Wafer kernel before submitting device work."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess


NOC_IMPORTS = frozenset({
    "direct_dte_attach", "direct_dte_release", "direct_dte_send_async",
    "direct_dte_wait_done", "direct_fsm_monitor_deinit", "direct_fsm_monitor_init",
    "direct_fsm_monitor_receive", "get_spm_memory_mapping", "get_tile_spm_addr_base",
    "set_direct_fsm_monitor_dst_addr",
})


def audit_kernel(path, log_abi="rcs", noc_firmware_elf=None):
    path = Path(path)
    data = path.read_bytes()
    if len(data) < 64 or data[:6] != b"\x7fELF\x02\x01":
        raise ValueError(f"Expected a little-endian ELF64 kernel: {path}")
    elf_type, machine = struct.unpack_from("<HH", data, 16)
    flags = struct.unpack_from("<I", data, 48)[0]
    if elf_type != 3 or machine != 243 or flags & 7 != 5:
        raise ValueError(
            f"Expected RISC-V shared ELF with RVC/double-float ABI: {path}"
        )
    binary_dir = Path(os.environ["LLVM_BINARY_DIR"])
    dynamic = subprocess.check_output(
        [str(binary_dir / "llvm-readelf"), "-l", "-d", "-S", str(path)], text=True
    )
    if "INTERP" in dynamic or "(NEEDED)" in dynamic:
        raise ValueError(
            f"Device ELF contains an interpreter or shared-library dependency: {path}"
        )
    if "ExportedDYNSYMTab" not in dynamic:
        raise ValueError(f"Device ELF has no ExportedDYNSYMTab: {path}")
    symbols = subprocess.check_output(
        [str(binary_dir / "llvm-nm"), "--undefined-only", "--format=posix", str(path)],
        text=True,
    )
    undefined = {line.split()[0] for line in symbols.splitlines() if line.strip()}
    allowed = {
        # SDK assert.h and the installed Kcore module export table agree on
        # __assert_func(file, line, function, expression), a non-returning call.
        "__assert_func",
        "__get_pid",
        "get_log_level",
        "monitor_write_log",
        "rt_free",
        "rt_malloc",
        "rt_thread_mdelay",
    }
    allowed.update(
        {"rcs_ep_log", "_rcs_ep_log", "rcs_kernel_printf", "rcs_kernel_vprintf"}
        if log_abi == "rcs"
        else {"tsm_ep_log", "_tsm_ep_log", "tx8_kernel_printf", "tx8_kernel_vprintf"}
    )
    firmware_evidence = None
    noc_imports = undefined & NOC_IMPORTS
    if noc_imports and noc_firmware_elf is not None:
        firmware = Path(noc_firmware_elf)
        # Check both the implementation and RT-Thread module export entry.
        # This is a reference ABI check; device execution still verifies the
        # firmware actually running on the card. No generic import override.
        defined_text = subprocess.check_output(
            [str(binary_dir / "llvm-nm"), "--defined-only", "--format=posix", str(firmware)],
            text=True,
        )
        defined = {line.split()[0] for line in defined_text.splitlines() if line.strip()}
        missing = {name for name in noc_imports
                   if name not in defined or "__rtmsym_" + name not in defined}
        if missing:
            raise ValueError(f"NoC firmware reference lacks module exports: {sorted(missing)}")
        allowed.update(noc_imports)
        firmware_evidence = {
            "path": str(firmware.resolve()),
            "sha256": hashlib.sha256(firmware.read_bytes()).hexdigest(),
            "module_exports": sorted(noc_imports),
        }
    unknown = undefined - allowed
    if unknown:
        raise ValueError(f"Unreviewed firmware imports in {path}: {sorted(unknown)}")
    return {
        "path": str(path),
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
        "flags": flags,
        "device_log_abi": log_abi,
        "firmware_imports": sorted(undefined),
        "noc_firmware_reference": firmware_evidence,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kernels", nargs="+", type=Path)
    parser.add_argument("--log-abi", choices=("wafer", "rcs"), default="rcs")
    parser.add_argument("--noc-firmware-elf", type=Path,
                        help="Reference firmware ELF used to verify the specific NoC module exports")
    args = parser.parse_args()
    print(
        json.dumps(
            [audit_kernel(path, args.log_abi, args.noc_firmware_elf) for path in args.kernels], indent=2
        )
    )
