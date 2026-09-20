"""NoC references must not authorize missing exports or unrelated imports."""
import importlib.util
from pathlib import Path
import struct

import pytest


@pytest.fixture
def audit(tmp_path, monkeypatch):
    source = Path(__file__).resolve().parents[2] / "scripts/wafer/audit_wafer_elf.py"
    spec = importlib.util.spec_from_file_location("wafer_elf_audit", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    header = bytearray(64)
    header[:6] = b"\x7fELF\x02\x01"
    struct.pack_into("<HH", header, 16, 3, 243)
    struct.pack_into("<I", header, 48, 5)
    kernel = tmp_path / "kernel.so"
    kernel.write_bytes(header)
    firmware = tmp_path / "firmware.elf"
    firmware.write_bytes(b"reference fixture")
    monkeypatch.setenv("LLVM_BINARY_DIR", str(tmp_path))
    return module, kernel, firmware


@pytest.mark.parametrize("definition,export", [(True, False), (False, True), (False, False)])
def test_noc_requires_implementation_and_module_export(audit, monkeypatch, definition, export):
    module, kernel, firmware = audit
    symbols = ("direct_dte_attach T 1 4\n" if definition else "")
    symbols += "__rtmsym_direct_dte_attach R 2 8\n" if export else ""
    outputs = iter(["ExportedDYNSYMTab", "direct_dte_attach U 0 0\n", symbols])
    monkeypatch.setattr(module.subprocess, "check_output", lambda *a, **k: next(outputs))
    with pytest.raises(ValueError, match="lacks module exports"):
        module.audit_kernel(kernel, noc_firmware_elf=firmware)


@pytest.mark.parametrize("unknown", [False, True])
def test_noc_reference_only_authorizes_reviewed_noc_imports(audit, monkeypatch, unknown):
    module, kernel, firmware = audit
    imports = "direct_dte_attach U 0 0\n" + ("RT_ASSERT U 0 0\n" if unknown else "")
    exports = "direct_dte_attach T 1 4\n__rtmsym_direct_dte_attach R 2 8\n"
    exports += "RT_ASSERT T 3 4\n__rtmsym_RT_ASSERT R 4 8\n"
    outputs = iter(["ExportedDYNSYMTab", imports, exports])
    monkeypatch.setattr(module.subprocess, "check_output", lambda *a, **k: next(outputs))
    if unknown:
        with pytest.raises(ValueError, match="Unreviewed firmware imports.*RT_ASSERT"):
            module.audit_kernel(kernel, noc_firmware_elf=firmware)
    else:
        result = module.audit_kernel(kernel, noc_firmware_elf=firmware)
        assert result["noc_firmware_reference"]["module_exports"] == ["direct_dte_attach"]


def test_noc_requires_explicit_reference(audit, monkeypatch):
    module, kernel, _ = audit
    outputs = iter(["ExportedDYNSYMTab", "direct_dte_attach U 0 0\n"])
    monkeypatch.setattr(module.subprocess, "check_output", lambda *a, **k: next(outputs))
    with pytest.raises(ValueError, match="Unreviewed firmware imports.*direct_dte_attach"):
        module.audit_kernel(kernel)


@pytest.mark.parametrize("abi,log_name", [("rcs", "_rcs_ep_log"), ("wafer", "_tsm_ep_log")])
def test_sdk_diagnostics_are_scoped_to_the_log_abi(audit, monkeypatch, abi, log_name):
    module, kernel, _ = audit
    monkeypatch.setattr(module.subprocess, "check_output", lambda args, **kw:
                        "ExportedDYNSYMTab" if "llvm-readelf" in str(args[0])
                        else f"{log_name} U 0 0\n__assert_func U 0 0\n")
    assert module.audit_kernel(kernel, log_abi=abi)["firmware_imports"] == sorted([log_name, "__assert_func"])
    other_abi = "wafer" if abi == "rcs" else "rcs"
    with pytest.raises(ValueError, match="Unreviewed firmware imports"):
        module.audit_kernel(kernel, log_abi=other_abi)
