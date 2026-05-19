# ============================================================================
# Compile all src/*.cpp → bc/*.aiv.bc (bitcode library for dl.custom())
#
# 编译器: ccec (Ascend CANN CCE compiler)
# 架构:   dav-c220-vec (Ascend 910B2 vector core)
#
# 用法:
#   bash compile_bc.sh             编译所有
#   bash compile_bc.sh add         只编译 add.cpp
#   bash compile_bc.sh softmax     只编译 softmax_ops.cpp
#   bash compile_bc.sh -f          强制重新编译全部
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC_DIR="$SCRIPT_DIR/src"
# Output to language/deeplink/bitcode/bc/ so .bc files are auto-installed with pip
BC_DIR="$SCRIPT_DIR/../../language/deeplink/bitcode/bc"

# ============================================================================
# 1. 检测 CANN 安装路径
# ============================================================================
detect_cann_path() {
    local cann_path=""

    # 优先级 1: CANN_PATH 环境变量
    if [ -n "${CANN_PATH:-}" ] && [ -d "$CANN_PATH" ]; then
        cann_path="$CANN_PATH"
    # 优先级 2: ASCEND_HOME_PATH 下的 cann 子目录
    elif [ -n "${ASCEND_HOME_PATH:-}" ] && [ -d "$ASCEND_HOME_PATH/cann" ]; then
        cann_path="$ASCEND_HOME_PATH/cann"
    elif [ -n "${ASCEND_HOME_PATH:-}" ] && [ -d "$ASCEND_HOME_PATH/cann-9.0.0" ]; then
        cann_path="$ASCEND_HOME_PATH/cann-9.0.0"
    # 优先级 3: 自动检测 /usr/local/Ascend/cann-*/
    else
        cann_path=$(ls -d /usr/local/Ascend/cann-*/ 2>/dev/null | head -1 || true)
        if [ -n "$cann_path" ]; then
            cann_path="${cann_path%/}"
        fi
    fi

    echo "$cann_path"
}

CANN_HOME=$(detect_cann_path)
if [ -z "$CANN_HOME" ]; then
    echo "ERROR: Cannot find CANN installation."
    echo "  Set CANN_PATH or ASCEND_HOME_PATH environment variable,"
    echo "  or install CANN toolkit under /usr/local/Ascend/"
    exit 1
fi
echo "CANN_HOME: $CANN_HOME"

# ============================================================================
# 2. 检测 ccec 编译器
# ============================================================================
CCEC="${CANN_HOME}/bin/ccec"
if [ ! -x "$CCEC" ]; then
    echo "ERROR: ccec not found at $CCEC"
    exit 1
fi
echo "CCEC: $CCEC"

# ============================================================================
# 3. 编译参数
# ============================================================================
# 架构: dav-c220-vec (Ascend 910B2), dav-c100-vec (Ascend 910B1)
AICORE_ARCH="${DLCOMPILER_AICORE_ARCH:-dav-c220-vec}"

CCEC_FLAGS="-x cce --cce-aicore-arch=${AICORE_ARCH} --cce-aicore-only -c -emit-llvm --std=c++17"

CCEC_INCLUDES="\
    -I ${CANN_HOME}/asc \
    -I ${CANN_HOME}/aarch64-linux/asc/include/basic_api \
    -I ${CANN_HOME}/aarch64-linux/asc/include/interface \
    -I ${CANN_HOME}/aarch64-linux/ascendc/include/highlevel_api \
    -I ${CANN_HOME}/aarch64-linux/ascendc/include/basic_api/impl \
    -I ${CANN_HOME}/aarch64-linux/ascendc/basic_api \
    -I ${CANN_HOME}/aarch64-linux/ascendc/basic_api/interface \
    -I ${CANN_HOME}/aarch64-linux/ascendc/highlevel_api/lib \
    -I ${CANN_HOME}/aarch64-linux/tiling"

echo "AICORE_ARCH: $AICORE_ARCH"
echo "INCLUDES:"
echo "$CCEC_INCLUDES" | tr ' ' '\n' | sed 's/^/  /'

# ============================================================================
# 4. 确保 bc/ 目录存在
# ============================================================================
mkdir -p "$BC_DIR"

# ============================================================================
# 5. 编译函数
# ============================================================================
compile_one() {
    local CPP="$1"
    local base_name="$(basename "${CPP%.cpp}")"
    local BC="$BC_DIR/${base_name}.aiv.bc"

    if [ -f "$BC" ] && [ "$FORCE" != "true" ]; then
        echo "Bitcode file $BC already exists, skipping."
        echo "  To recompile: rm -f $BC && bash compile_bc.sh"
        return
    fi

    echo "Compiling $CPP → $BC ..."
    ${CCEC} ${CCEC_FLAGS} ${CCEC_INCLUDES} "${CPP}" -o "${BC}"

    if [ $? -eq 0 ]; then
        echo "  OK: $BC"
        # Show exported custom symbols
        if command -v nm &> /dev/null; then
            echo "  Symbols:"
            nm -C "${BC}" 2>/dev/null | grep -i custom | sed 's/^/    /' || \
            nm "${BC}" 2>/dev/null | grep -i custom | sed 's/^/    /' || true
        fi
    else
        echo "  FAILED: $CPP"
        exit 1
    fi
}

# ============================================================================
# 6. 解析参数 → 确定编译目标
# ============================================================================
FORCE=false
TARGET=""

for arg in "$@"; do
    case "$arg" in
        -f|--force)
            FORCE=true
            ;;
        *)
            TARGET="$arg"
            ;;
    esac
done

cd "$SCRIPT_DIR"

if [ -n "$TARGET" ]; then
    case "$TARGET" in
        add)
            compile_one "$SRC_DIR/add.cpp"
            ;;
        softmax)
            compile_one "$SRC_DIR/softmax_ops.cpp"
            ;;
        all)
            for CPP in "$SRC_DIR"/*.cpp; do
                [ -f "$CPP" ] && compile_one "$CPP"
            done
            ;;
        *)
            echo "Unknown target: $TARGET"
            echo "  Options: add, softmax, softmax_full, all"
            exit 1
            ;;
    esac
else
    # 默认：编译所有
    for CPP in "$SRC_DIR"/*.cpp; do
        [ -f "$CPP" ] && compile_one "$CPP"
    done
fi

echo ""
echo "Done. Bitcode files in: $BC_DIR"
ls -lh "$BC_DIR"/*.aiv.bc 2>/dev/null || echo "  (no .aiv.bc files)"
