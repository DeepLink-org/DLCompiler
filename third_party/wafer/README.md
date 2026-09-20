# Local Build and Runtime Configuration

Repository-level build, environment, and packaging tools live in
`scripts/wafer/`. Standalone `verify_wafer_*` acceptance programs live in
`test/wafer/` alongside pytest tests, but are invoked explicitly rather than
collected by pytest. Device acceptance programs require the local hardware
environment. The packaging entry `setup_on_wafer.py` remains at repository root.

Prepare the pinned LLVM toolchain, Triton source checkout, Wafer SDK, and
matching Torch/Kuiper runtime locally. Dependency acquisition is outside this
project's installation interface. Missing dependencies must be supplied by the
user; the Wafer setup scripts do not download replacements.

Set `LLVM_SYSPATH` and `WAFER_DEPS_ROOT`, then run the repository-root
`scripts/wafer/setup_wafer_env.sh`, `scripts/wafer/compile_wafer.sh`, and `scripts/wafer/install_wafer.sh` entries with the
prepared Python environment. `scripts/wafer/migrate_wafer_env.sh --compiler-only` checks local
compiler prerequisites and writes the activation file; it does not install
dependencies. Initialize the pinned `third_party/triton` checkout locally before
building. Python build requirements must already be installed.

The project target is `GPUTarget("wafer", "wafer", 32)`. Device log configuration
accepts `WAFER_DEVICE_LOG_ABI=wafer` for the SDK's original logging ABI or `rcs`
for the RCS firmware ABI. SDK binary symbols and SDK directory layouts retain
their vendor-defined spelling. Torch device allocation continues to use the
matching `torch_txda` extension and its registered `txda` device.

Set `WAFER_BSP_INCLUDE_DIR` when the local SDK provides BSP headers in a
different directory. An explicit CMake value takes precedence over the
environment. Otherwise the vendor SDK's default BSP layout is used.

# Wafer Triton Plugin

## Current Status

This directory contains the Wafer external Triton plugin imported for the
DLCompiler Triton 3.5 migration. The import includes the bundled FLIR and TLE
sources and excludes prior build trees, generated IR, binaries, caches, and
machine-specific paths.

The plugin is adapted to LLVM/MLIR 22 and connected to the combined DLCompiler
and Wafer plugin build. A real Triton BF16 GEMM containing `tl.dot` is verified
through TTIR, CoreIR, TXIR, LLVM dialect IR, LLVM IR, and object generation.

The integration includes Kuiper linking, a device launcher, and native
`torch_txda` acceptance checks. Device execution requires a compatible local
runtime, firmware, and available device. Compiler-only checks do not establish
device correctness. `test/wafer/verify_wafer_runtime.py --compile-only --case all`
checks compilation, linking, and ELF structure without launching a kernel;
device acceptance must be run separately in the prepared runtime environment.

## Source Layout

- `backend/`: Python backend implementation imported for later adaptation
- `bin/`: Wafer command-line tools and dialect registration
- `include/` and `lib/`: dialects, analyses, and conversion passes
- `crt/` and `profiler/`: optional Wafer runtime components
- `third_party/flir/`: bundled FLIR sources
- `third_party/tle/`: bundled TLE sources

## Optional Wafer Dependencies

`WAFER_DEPS_ROOT` is optional at plugin configuration time. When it is unset or
points to a missing directory, the top-level CMake configuration skips both
`crt/` and `profiler/`.

The main compiler and dialect sources remain available without Wafer runtime
dependencies. A later build that enables CRT or the profiler must provide a
valid `WAFER_DEPS_ROOT` and a compatible LLVM toolchain containing Clang.

## LLVM Environment

Prepare the Triton-pinned LLVM/MLIR 22 environment from the repository root:

```bash
./scripts/wafer/setup_llvm22_env.sh
source llvm22_env.sh
```

Device code generation and linking require a matching LLVM RISC-V toolchain
and Wafer runtime libraries in addition to the compiler-only package.
