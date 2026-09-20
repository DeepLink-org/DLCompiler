"""Check the diagnostic ABI without triggering a fatal assertion on the card."""
import os
from pathlib import Path
import shutil
import subprocess

import pytest


def test_assert_reports_and_terminates_with_ndebug(tmp_path):
    compiler = shutil.which("cc")
    if not compiler:
        pytest.skip("C compiler required")
    root = Path(__file__).resolve().parents[2]
    (tmp_path / "wafer.h").write_text("""
      #define INTRNISIC_RUN_SWITCH
      #define KCORE_LOG_ERROR 3
      void tsm_ep_log(const char *, const char *, unsigned, unsigned, const char *, ...);
      void __assert_func(const char *, int, const char *, const char *) __attribute__((noreturn));
    """)
    source = tmp_path / "check.c"
    source.write_text(r'''
      #include <stdarg.h>
      #include <stdio.h>
      #include <stdlib.h>
      #include <string.h>
      static int reported;
      void tsm_ep_log(const char *file, const char *func, unsigned line,
                      unsigned level, const char *format, ...) {
        va_list args;
        char text[256];
        va_start(args, format);
        vsnprintf(text, sizeof(text), format, args);
        va_end(args);
        if (level != 3 || strcmp(text,
            "kernel.py(line 17, col 9)::tile (2, 3, 4): bad value\n")) exit(21);
        reported = 1;
      }
      void __assert_func(const char *file, int line, const char *func, const char *expr) {
        if (!reported || strcmp(file, "kernel.py") || line != 17 ||
            strcmp(func, "__Assert") || strcmp(expr, "bad value")) exit(22);
        exit(0);
      }
      void __Assert(const char *, ...);
      int main(void) {
        __Assert("bad value", "kernel.py", 17, 9, 2, 3, 4);
        return 23;
      }
    ''')
    exe = tmp_path / "check"
    subprocess.run([
        compiler, "-O2", "-DNDEBUG", "-Werror=implicit-function-declaration",
        "-I", str(tmp_path), str(root / "third_party/wafer/crt/lib/Wafer/assert.c"),
        str(source), "-o", str(exe),
    ], check=True)
    subprocess.run([str(exe)], check=True, timeout=10)


@pytest.mark.parametrize("with_sdk_init", [False, True])
def test_assert_links_to_firmware_without_posix_imports(wafer_modules, monkeypatch, tmp_path, with_sdk_init):
    if not os.getenv("WAFER_DEPS_ROOT"):
        pytest.skip("Wafer SDK required for RISC-V link test")
    _, compiler, _ = wafer_modules
    linker, libraries = compiler._runtime_link_inputs()
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "cache"))
    source = tmp_path / "probe.c"
    # module_init pulls the real intrinsic/common archives and their libc
    # dependencies. An assertion-only object cannot expose missing libgloss.
    source.write_text('''#include <assert.h>
      extern void module_init(void *);
      void probe(void *args) {
    ''' + ('module_init(args);' if with_sdk_init else '') + '''
        __assert_func("probe", 1, "probe", "false");
      }
    ''')
    obj = tmp_path / "probe.o"
    subprocess.run([str(linker), "-fPIC", "-O2", "-march=rv64imfdc", "-mabi=lp64d",
                    "-c", str(source), "-o", str(obj)], check=True)
    original_libc = next(p for p in libraries if p.name == "libc.a")
    before = compiler.file_fingerprint(original_libc)
    metadata = {}
    compiler.object_to_binary(obj.read_bytes(), metadata, simulator=False, log_abi="rcs")
    nm = compiler._find_llvm_tool("llvm-nm")
    undefined = subprocess.check_output([str(nm), "--undefined-only", "--format=posix",
                                         metadata["kernel_path"]], text=True)
    imports = {line.split()[0] for line in undefined.splitlines()}
    assert "__assert_func" in imports
    allowed = {"__assert_func", "__get_pid", "get_log_level", "monitor_write_log",
               "rcs_ep_log", "rcs_kernel_printf", "rcs_kernel_vprintf",
               "rt_free", "rt_malloc", "rt_thread_mdelay"}
    assert imports <= allowed, imports - allowed
    if not with_sdk_init:
        assert imports == {"__assert_func"}
    assert compiler.file_fingerprint(original_libc) == before
