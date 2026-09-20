#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)

LLVM_COMMIT=7d5de3033187c8a3bb4d2e322f5462cdaf49808f
: "${LLVM_SYSPATH:?Provide the local LLVM directory}"
LLVM_ENV_FILE=${LLVM_ENV_FILE:-$SCRIPT_DIR/llvm22_env.sh}
TRITON_LLVM_HASH_FILE=$SCRIPT_DIR/third_party/triton/cmake/llvm-hash.txt

required_tools=(
    FileCheck
    ld.lld
    llvm-config
    llvm-tblgen
    mlir-opt
    mlir-tblgen
    mlir-translate
)

if [[ $(uname -s) != "Linux" || $(uname -m) != "x86_64" ]]; then
    echo "ERROR: the pinned prebuilt package supports Linux x86_64 only" >&2
    exit 1
fi

test -f "$TRITON_LLVM_HASH_FILE" || {
    echo "ERROR: initialize third_party/triton before preparing LLVM" >&2
    exit 1
}
TRITON_LLVM_COMMIT=$(tr -d '[:space:]' < "$TRITON_LLVM_HASH_FILE")
if [[ "$TRITON_LLVM_COMMIT" != "$LLVM_COMMIT" ]]; then
    echo "ERROR: LLVM pin does not match third_party/triton" >&2
    echo "  script: $LLVM_COMMIT" >&2
    echo "  triton: $TRITON_LLVM_COMMIT" >&2
    exit 1
fi

verify_toolchain() {
    local llvm_root=$1
    local tool

    test -d "$llvm_root" || {
        echo "ERROR: LLVM directory does not exist: $llvm_root" >&2
        return 1
    }

    for tool in "${required_tools[@]}"; do
        test -x "$llvm_root/bin/$tool" || {
            echo "ERROR: required LLVM tool is missing: $llvm_root/bin/$tool" >&2
            return 1
        }
    done

    local llvm_version
    local mlir_version
    llvm_version=$($llvm_root/bin/llvm-config --version)
    mlir_version=$($llvm_root/bin/mlir-opt --version | sed -n 's/^.*version //p' | head -n 1)
    if [[ "$llvm_version" != "22.0.0git" || "$mlir_version" != "22.0.0git" ]]; then
        echo "ERROR: expected LLVM/MLIR 22.0.0git, got LLVM=$llvm_version MLIR=$mlir_version" >&2
        return 1
    fi

    grep -q 'PACKAGE_VERSION "22.0.0git"' \
        "$llvm_root/lib/cmake/llvm/LLVMConfigVersion.cmake" || {
        echo "ERROR: LLVM CMake package is not version 22.0.0git" >&2
        return 1
    }
    grep -q 'PACKAGE_VERSION "22.0.0git"' \
        "$llvm_root/lib/cmake/mlir/MLIRConfigVersion.cmake" || {
        echo "ERROR: MLIR CMake package is not version 22.0.0git" >&2
        return 1
    }
}

verify_toolchain "$LLVM_SYSPATH"

cat > "$LLVM_ENV_FILE" <<EOF
#!/usr/bin/env bash

export LLVM_COMMIT="$LLVM_COMMIT"
export LLVM_SYSPATH="$LLVM_SYSPATH"
export LLVM_BINARY_DIR="\$LLVM_SYSPATH/bin"
export LLVM_DIR="\$LLVM_SYSPATH/lib/cmake/llvm"
export MLIR_DIR="\$LLVM_SYSPATH/lib/cmake/mlir"
export PATH="\$LLVM_BINARY_DIR:\${PATH:-}"
hash -r 2>/dev/null || true

for tool in llvm-config llvm-tblgen mlir-opt mlir-tblgen mlir-translate; do
    resolved=\$(command -v "\$tool" || true)
    expected="\$LLVM_BINARY_DIR/\$tool"
    if [[ "\$resolved" != "\$expected" ]]; then
        echo "ERROR: mixed LLVM toolchain: \$tool resolves to \$resolved, expected \$expected" >&2
        return 1 2>/dev/null || exit 1
    fi
done

if [[ \$(llvm-config --version) != "22.0.0git" ]]; then
    echo "ERROR: LLVM 22 environment activation failed" >&2
    return 1 2>/dev/null || exit 1
fi
EOF
chmod +x "$LLVM_ENV_FILE"

printf '\nLLVM/MLIR environment is ready.\n'
printf '  commit: %s\n' "$LLVM_COMMIT"
printf '  root:   %s\n' "$LLVM_SYSPATH"
printf '  env:    %s\n' "$LLVM_ENV_FILE"
printf '\nActivate with:\n  source %q\n' "$LLVM_ENV_FILE"
